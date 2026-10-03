"""Train/evaluate V3.6 from paired deposit, method, and partition-count runs."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_hybrid_v36_selector import (  # noqa: E402
    DATASETS,
    _make_estimator,
)
from graph.load_graph import load_graph  # noqa: E402
from partition.hybrid_v36 import (  # noqa: E402
    FEATURE_NAMES,
    SCORE_WEIGHTS,
    candidate_features,
    metis_partition,
    rank_candidates,
    spatial_partition,
)

METHODS = ("Spatial", "METIS")
DEFAULT_PARTITIONS = (4, 8, 12)


def _load_graph(dataset: str):
    spec = DATASETS[dataset]
    nx_dim, ny_dim, nz_dim = spec["dims"]
    graph, frame = load_graph(
        str(spec["path"]), nx_dim=nx_dim, ny_dim=ny_dim, nz_dim=nz_dim
    )
    count = len(frame)
    graph.remove_nodes_from([node for node in (0, count + 1) if node in graph])
    values = frame["EPV"].to_numpy(dtype=np.float64)
    for node in graph:
        index = int(node) - 1
        graph.nodes[node].update({
            "x": float(index % nx_dim),
            "y": float((index // nx_dim) % ny_dim),
            "z": float(index // (nx_dim * ny_dim)),
            "value": float(values[index]),
        })
    return graph


def run(
    outcomes_path: Path | None = None,
    partitions: tuple[int, ...] = DEFAULT_PARTITIONS,
    results_dir: Path | None = None,
) -> dict:
    results_dir = results_dir or ROOT / "results"
    outcomes_path = outcomes_path or results_dir / "hybrid_v36_partition_grid_outcomes.csv"
    if not outcomes_path.is_absolute():
        outcomes_path = ROOT / outcomes_path
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Missing paired run outcomes: {outcomes_path}")
    if not partitions or any(k < 2 for k in partitions):
        raise ValueError("partition counts must each be at least 2")

    with outcomes_path.open(newline="", encoding="utf-8") as stream:
        outcomes = list(csv.DictReader(stream))

    outcome_index = {}
    for row in outcomes:
        dataset = row["dataset"]
        k = int(row["partitions"])
        method = row["method"].strip().lower()
        method = {"spatial": "Spatial", "metis": "METIS"}.get(method, method)
        if method not in METHODS:
            continue
        key = (dataset, k, method)
        if key in outcome_index:
            raise ValueError(f"Duplicate paired outcome for {key}")
        outcome_index[key] = float(row["final_npv"])

    records: list[dict] = []
    graphs = {}
    candidate_sets = {}
    for dataset in DATASETS:
        graph = _load_graph(dataset)
        graphs[dataset] = graph
        for k in partitions:
            key = (dataset, k)
            candidates = [
                ("Spatial", spatial_partition(graph, k)),
                ("METIS", metis_partition(graph, k)),
            ]
            if candidates[0][1] == candidates[1][1]:
                raise RuntimeError(
                    f"Spatial and METIS assignments are identical for {dataset}, k={k}; "
                    "the comparison would not contain two real partition methods"
                )
            expected = {(dataset, k, method) for method in METHODS}
            missing = expected - outcome_index.keys()
            if missing:
                raise ValueError(f"Missing paired outcomes: {sorted(missing)}")
            npvs = {method: outcome_index[(dataset, k, method)] for method in METHODS}
            best_npv = max(npvs.values())
            winners = {method for method, npv in npvs.items() if math.isclose(npv, best_npv)}
            candidate_sets[key] = candidates
            for method, labels in candidates:
                metrics = candidate_features(graph, labels, method)
                records.append({
                    "dataset": dataset,
                    "partitions": k,
                    "method": method,
                    **{name: metrics[name] for name in FEATURE_NAMES},
                    "observed_npv": npvs[method],
                    "best_choice": int(method in winners),
                })

    frame = pd.DataFrame(records)
    x = frame[FEATURE_NAMES]
    y = frame["best_choice"].to_numpy(dtype=int)
    groups = frame["dataset"].to_numpy()
    if len(set(y)) < 2:
        raise ValueError("paired outcomes need both winning and non-winning candidates")

    cv = LeaveOneGroupOut()
    holdout_pairs = []
    for train_indices, test_indices in cv.split(x, y, groups):
        estimator = _make_estimator()
        estimator.fit(x.iloc[train_indices], y[train_indices])
        test_frame = frame.iloc[test_indices]
        dataset = str(test_frame["dataset"].iloc[0])
        for k in partitions:
            ranked = rank_candidates(
                graphs[dataset], candidate_sets[(dataset, k)],
                {"version": "3.6", "model": estimator, "features": FEATURE_NAMES},
            )
            expected = set(test_frame.loc[
                (test_frame["partitions"] == k) & (test_frame["best_choice"] == 1),
                "method",
            ])
            npvs = {
                row["method"]: float(row["observed_npv"])
                for row in test_frame.loc[test_frame["partitions"] == k].to_dict("records")
            }
            best_npv = max(npvs.values())
            selected = ranked[0]["method"]
            ml_only = max(ranked, key=lambda row: row["ml_probability"])["method"]
            holdout_pairs.append({
                "dataset": dataset,
                "partitions": k,
                "selected_method": selected,
                "ml_only_method": ml_only,
                "observed_best_methods": sorted(expected),
                "selected_correct": selected in expected,
                "ml_only_correct": ml_only in expected,
                "selected_relative_npv_regret": (best_npv - npvs[selected]) / max(abs(best_npv), 1.0),
                "spatial_relative_npv_regret": (best_npv - npvs["Spatial"]) / max(abs(best_npv), 1.0),
                "metis_relative_npv_regret": (best_npv - npvs["METIS"]) / max(abs(best_npv), 1.0),
                "ranked": [
                    {"method": row["method"], "score": row["score"],
                     "ml_probability": row["ml_probability"],
                     "economic_quality": row["score_components"]["economic_quality"],
                     "balance_quality": row["score_components"]["balance_quality"],
                     "cut_quality": row["score_components"]["cut_quality"]}
                    for row in ranked
                ],
            })

    estimator = _make_estimator()
    estimator.fit(x, y)
    model_path = results_dir / "hybrid_v36_grid_selector.pkl"
    bundle = {
        "version": "3.6-grid-pilot",
        "model": estimator,
        "features": FEATURE_NAMES,
        "score_weights": SCORE_WEIGHTS,
        "candidate_methods": list(METHODS),
        "partition_counts": list(partitions),
        "label_definition": "highest final NPV within each dataset and matched partition count",
        "feature_policy": "pre-solve only; no dataset identity or post-solve outcomes",
    }
    joblib.dump(bundle, model_path)
    training_csv = results_dir / "hybrid_v36_grid_training_data.csv"
    frame.to_csv(training_csv, index=False)

    always_spatial = sum(r["observed_best_methods"] == ["Spatial"] for r in holdout_pairs)
    always_metis = sum(r["observed_best_methods"] == ["METIS"] for r in holdout_pairs)
    metrics = {
        "version": "3.6-grid-pilot",
        "dataset_count": len(DATASETS),
        "independent_dataset_groups": len(DATASETS),
        "partition_counts": list(partitions),
        "paired_settings": len(holdout_pairs),
        "candidate_count": len(frame),
        "features": FEATURE_NAMES,
        "score_weights": SCORE_WEIGHTS,
        "validation": "leave-one-dataset-out across matched method pairs at each partition count",
        "composite_selection_accuracy": float(np.mean([r["selected_correct"] for r in holdout_pairs])),
        "ml_only_selection_accuracy": float(np.mean([r["ml_only_correct"] for r in holdout_pairs])),
        "always_spatial_selection_accuracy": always_spatial / len(holdout_pairs),
        "always_metis_selection_accuracy": always_metis / len(holdout_pairs),
        "mean_selected_relative_npv_regret": float(np.mean([r["selected_relative_npv_regret"] for r in holdout_pairs])),
        "mean_spatial_relative_npv_regret": float(np.mean([r["spatial_relative_npv_regret"] for r in holdout_pairs])),
        "mean_metis_relative_npv_regret": float(np.mean([r["metis_relative_npv_regret"] for r in holdout_pairs])),
        "holdout_results": holdout_pairs,
        "training_csv": str(training_csv),
        "model": str(model_path),
        "limitations": [
            "There are more paired outcome settings, but only four independent deposits; leave-one-deposit-out remains the generalization check.",
            "Partition counts are related runs on the same deposits, not independent geological samples.",
            "Only Spatial and METIS are included in this expanded V3.6 comparison.",
        ],
    }
    metrics_path = results_dir / "hybrid_v36_grid_training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
