"""Build and grouped-validate the V3.6 economic partition selector."""
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph.load_graph import load_graph
from partition.hybrid_v36 import (
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    SCORE_WEIGHTS,
    candidate_features,
    metis_partition,
    rank_candidates,
    spatial_partition,
)

DATASETS = {
    "bauxitemed": {"path": ROOT.parent / "lg" / "bauxitemed.dat", "dims": (120, 120, 26)},
    "cucase": {"path": ROOT.parent / "lg" / "cucase.dat", "dims": (170, 215, 50)},
    "cupipe": {"path": ROOT.parent / "lg" / "cupipe.dat", "dims": (180, 180, 85)},
    "mclaughlingeo": {"path": ROOT.parent / "lg" / "mclaughlingeo.dat", "dims": (140, 296, 68)},
}
METHODS = ("Spatial", "METIS")


def _load_candidates(dataset: str, n_partitions: int):
    spec = DATASETS[dataset]
    if not spec["path"].exists():
        raise FileNotFoundError(f"Missing model input: {spec['path']}")
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
    candidates = [
        ("Spatial", spatial_partition(graph, n_partitions)),
        ("METIS", metis_partition(graph, n_partitions)),
    ]
    if candidates[0][1] == candidates[1][1]:
        raise RuntimeError(
            f"Spatial and METIS returned identical labels for {dataset}; "
            "check pymetis availability before training V3.6"
        )
    return graph, candidates


def _make_estimator():
    # Small, regularized model: four independent block models do not justify a
    # high-capacity classifier. Dataset identity is intentionally not a feature.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=36),
    )


def run(
    results_dir: Path | None = None,
    n_partitions: int = 8,
    outcomes_path: Path | None = None,
    model_path: Path | None = None,
) -> dict:
    results_dir = results_dir or ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = outcomes_path or results_dir / "final_hybrid_comparison.csv"
    outcomes_path = outcomes_path if outcomes_path.is_absolute() else ROOT / outcomes_path
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Missing observed candidate outcomes: {outcomes_path}")
    if importlib.util.find_spec("pymetis") is None:
        raise RuntimeError("V3.6 training requires pymetis; METIS fallbacks are not accepted")

    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes[outcomes["method"].isin(METHODS)].copy()
    rows: list[dict] = []
    graphs = {}
    candidates_by_dataset = {}

    for dataset in DATASETS:
        observed = outcomes[outcomes["dataset"] == dataset]
        if set(observed["method"]) != set(METHODS):
            raise ValueError(f"Need exactly Spatial and METIS outcomes for {dataset}")
        best_npv = float(observed["npv"].max())
        best_methods = set(observed.loc[observed["npv"] == best_npv, "method"])
        graph, candidates = _load_candidates(dataset, n_partitions)
        graphs[dataset] = graph
        candidates_by_dataset[dataset] = candidates
        for method, labels in candidates:
            features = candidate_features(graph, labels, method)
            rows.append({
                "dataset": dataset,
                "method": method,
                **{name: features[name] for name in FEATURE_NAMES},
                "observed_npv": float(observed.loc[observed["method"] == method, "npv"].iloc[0]),
                "best_choice": int(method in best_methods),
            })

    frame = pd.DataFrame(rows)
    x = frame[FEATURE_NAMES]
    y = frame["best_choice"].to_numpy(dtype=int)
    groups = frame["dataset"].to_numpy()
    if len(set(y)) < 2:
        raise ValueError("historical outcomes need both winning and non-winning methods")

    cv = LeaveOneGroupOut()
    holdout_results = []
    for train_indices, test_indices in cv.split(x, y, groups):
        estimator = _make_estimator()
        estimator.fit(x.iloc[train_indices], y[train_indices])
        test_dataset = str(groups[test_indices[0]])
        ranked = rank_candidates(
            graphs[test_dataset], candidates_by_dataset[test_dataset],
            {"version": "3.6", "model": estimator, "features": FEATURE_NAMES},
        )
        observed_winners = set(frame.loc[
            (frame["dataset"] == test_dataset) & (frame["best_choice"] == 1), "method"
        ])
        selected = ranked[0]["method"]
        ml_only = max(ranked, key=lambda item: item["ml_probability"])["method"]
        holdout_results.append({
            "dataset": test_dataset,
            "selected_method": selected,
            "ml_only_method": ml_only,
            "observed_best_methods": sorted(observed_winners),
            "selected_correct": selected in observed_winners,
            "ml_only_correct": ml_only in observed_winners,
            "ranked": [
                {"method": record["method"], "score": record["score"],
                 "ml_probability": record["ml_probability"],
                 "economic_quality": record["score_components"]["economic_quality"],
                 "balance_quality": record["score_components"]["balance_quality"],
                 "cut_quality": record["score_components"]["cut_quality"]}
                for record in ranked
            ],
        })

    final_estimator = _make_estimator()
    final_estimator.fit(x, y)
    model_path = model_path or DEFAULT_MODEL_PATH
    model_path = model_path if model_path.is_absolute() else ROOT / model_path
    bundle = {
        "version": "3.6",
        "model": final_estimator,
        "features": FEATURE_NAMES,
        "score_weights": SCORE_WEIGHTS,
        "candidate_methods": list(METHODS),
        "label_definition": "method(s) with the highest observed NPV among Spatial and METIS",
        "feature_policy": "pre-solve graph, partition, and boundary-economics features only; no dataset identity or realized NPV",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)

    csv_path = results_dir / "hybrid_v36_training_data.csv"
    frame.to_csv(csv_path, index=False)
    always_spatial = sum("Spatial" in row["observed_best_methods"] for row in holdout_results)
    always_metis = sum("METIS" in row["observed_best_methods"] for row in holdout_results)
    metrics = {
        "version": "3.6",
        "dataset_count": len(DATASETS),
        "candidate_count": len(frame),
        "n_partitions": n_partitions,
        "features": FEATURE_NAMES,
        "score_weights": SCORE_WEIGHTS,
        "label_definition": bundle["label_definition"],
        "validation": "leave-one-dataset-out; rank the two candidates in each held-out model",
        "composite_selection_accuracy": float(np.mean([r["selected_correct"] for r in holdout_results])),
        "ml_only_selection_accuracy": float(np.mean([r["ml_only_correct"] for r in holdout_results])),
        "always_spatial_selection_accuracy": always_spatial / len(holdout_results),
        "always_metis_selection_accuracy": always_metis / len(holdout_results),
        "holdout_results": holdout_results,
        "training_csv": str(csv_path),
        "model": str(model_path),
        "limitations": [
            "Only four independent block models are available for grouped validation.",
            "Historical labels come from saved NPV outcomes; candidate metrics are recomputed from current spatial/METIS partitions.",
            "The validation supports a pilot comparison, not generalization to new deposits.",
        ],
    }
    metrics_path = results_dir / "hybrid_v36_training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
