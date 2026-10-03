"""Grouped pilot for jointly selecting partition method and partition count.

Uses measured Spatial/METIS outcomes at k=4,8,12 and measured spectral k=4
outcomes. Labels and features are deposit-held-out. The saved model is a
research artifact and is not wired into the production experiment runner.
"""
from __future__ import annotations

import csv
import json
import math
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

from graph.load_graph import load_graph  # noqa: E402
from partition.hybrid_v36 import candidate_features, metis_partition, spatial_partition  # noqa: E402
from experiments.train_hybrid_v36_selector import DATASETS  # noqa: E402

METHODS = ("spatial", "metis", "spectral")
COUNTS = (4, 8, 12)
FEATURES = (
    "log_graph_nodes", "log_precedence_arcs", "log_partition_count",
    "is_metis", "is_spectral", "cut_fraction", "balance_ratio",
    "minimum_part_fraction", "maximum_part_fraction",
)


def _load_outcomes() -> dict[tuple[str, str, int], dict]:
    result = {}
    grid_path = ROOT / "results" / "hybrid_v36_partition_grid_outcomes.csv"
    with grid_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            method = row["method"].lower()
            if method not in {"spatial", "metis"}:
                continue
            key = (row["dataset"], method, int(row["partitions"]))
            result[key] = {
                "final_npv": float(row["final_npv"]),
                "partition_runtime_s": float(row["partition_runtime_s"]),
                "local_solve_runtime_s": float(row["local_solve_runtime_s"]),
                "refinement_runtime_s": float(row["refinement_runtime_s"]),
            }

    spectral_path = ROOT / "results" / "spectral_partition_extension_summary.csv"
    with spectral_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[(row["dataset"], "spectral", int(row["partitions"]))] = {
                "final_npv": float(row["final_npv"]),
                "partition_runtime_s": float(row["partition_runtime_s"]),
                "local_solve_runtime_s": float(row["local_solve_runtime_s"]),
                "refinement_runtime_s": float(row["refinement_runtime_s"]),
                "edge_cut": float(row["edge_cut"]),
                "balance": float(row["balance"]),
                "min_part": int(row["min_part"]),
                "max_part": int(row["max_part"]),
            }
    return result


def _partition_records(dataset: str, outcomes: dict) -> list[dict]:
    spec = DATASETS[dataset]
    nx_dim, ny_dim, nz_dim = spec["dims"]
    graph, frame = load_graph(str(spec["path"]), nx_dim=nx_dim, ny_dim=ny_dim, nz_dim=nz_dim)
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

    rows = []
    for k in COUNTS:
        for method in ("spatial", "metis"):
            key = (dataset, method, k)
            if key not in outcomes:
                raise ValueError(f"Missing measured paired outcome: {key}")
            labels = spatial_partition(graph, k) if method == "spatial" else metis_partition(graph, k)
            features = candidate_features(graph, labels, method)
            rows.append({
                "dataset": dataset, "method": method, "partitions": k,
                **features, **outcomes[key],
            })

    key = (dataset, "spectral", 4)
    if key not in outcomes:
        raise ValueError(f"Missing measured spectral outcome: {key}")
    spectral = outcomes[key]
    cut_count = spectral["edge_cut"] * graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
    k = 4
    # The saved spectral run supplies exact balance/cut telemetry. Economic
    # boundary exposure is intentionally excluded because labels were not
    # retained and must not be regenerated via a 5–7 minute embedding run.
    rows.append({
        "dataset": dataset, "method": "spectral", "partitions": k,
        "log_graph_nodes": math.log1p(n_nodes),
        "log_precedence_arcs": math.log1p(graph.number_of_edges()),
        "partition_count": float(k),
        "is_metis": 0.0, "cut_fraction": spectral["edge_cut"],
        "balance_ratio": spectral["balance"],
        "minimum_part_fraction": spectral["min_part"] / n_nodes,
        "maximum_part_fraction": spectral["max_part"] / n_nodes,
        **spectral,
    })
    del cut_count
    return rows


def _estimator():
    # Conservative linear classifier for only four independent deposit groups.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=37),
    )


def _config_baseline(train_frame: pd.DataFrame, test_frame: pd.DataFrame, mode: str,
                     comparison_frame: pd.DataFrame | None = None) -> dict:
    comparison_frame = test_frame if comparison_frame is None else comparison_frame
    best_npv = comparison_frame["observed_npv"].max()
    test_regret = {
        (row.method, int(row.partitions)): (best_npv - row.observed_npv) / max(abs(best_npv), 1.0)
        for row in comparison_frame.itertuples()
    }
    if mode == "mean_regret":
        train_best = train_frame.groupby("dataset")["observed_npv"].transform("max")
        tmp = train_frame.assign(regret=(train_best - train_frame["observed_npv"]) / train_best.abs().clip(lower=1.0))
        stats = tmp.groupby(["method", "partitions"])["regret"].mean()
        selected = min(stats.index, key=lambda key: (float(stats.loc[key]), key[0], key[1]))
    elif mode == "winner_count":
        train_best = train_frame.groupby("dataset")["observed_npv"].transform("max")
        tmp = train_frame.assign(winner=np.isclose(train_frame["observed_npv"], train_best, rtol=1e-12, atol=1e-9))
        stats = tmp.groupby(["method", "partitions"])["winner"].sum()
        selected = min(stats.index, key=lambda key: (-int(stats.loc[key]), key[0], key[1]))
    else:
        raise ValueError(mode)
    return {"selected_method": selected[0], "selected_partitions": int(selected[1]),
            "relative_npv_regret": float(test_regret[selected]),
            "correct": bool(test_regret[selected] <= 1e-12)}


def run(results_dir: Path | None = None) -> dict:
    results_dir = results_dir or ROOT / "results"
    outcomes = _load_outcomes()
    data_path = results_dir / "hybrid_v37_joint_training_data.csv"
    expected = {(dataset, method, k): result["final_npv"]
                for (dataset, method, k), result in outcomes.items()
                if dataset in DATASETS and method in METHODS and k in COUNTS}
    frame = None
    if data_path.exists():
        try:
            cached = pd.read_csv(data_path)
            cached_keys = {(str(row.dataset), str(row.method), int(row.partitions))
                           for row in cached.itertuples()}
            cache_matches = len(cached) == len(expected) and cached_keys == set(expected)
            if cache_matches:
                cache_matches = all(
                    math.isclose(float(row.final_npv), expected[(str(row.dataset), str(row.method), int(row.partitions))],
                                 rel_tol=1e-12, abs_tol=1e-9)
                    for row in cached.itertuples()
                )
            if cache_matches and set(FEATURES) <= set(cached.columns):
                frame = cached
                print(f"Reuse validated pre-solve feature table: {data_path.name}", flush=True)
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            frame = None

    if frame is None:
        rows = []
        for dataset in DATASETS:
            print(f"Deriving pre-solve candidate features: {dataset}", flush=True)
            rows.extend(_partition_records(dataset, outcomes))
        frame = pd.DataFrame(rows)
    # One best method/count candidate per deposit is the binary positive class.
    frame["observed_npv"] = frame["final_npv"].astype(float)
    frame["best_choice"] = frame.groupby("dataset")["observed_npv"].transform(
        lambda values: np.isclose(values, values.max(), rtol=1e-12, atol=1e-9)
    ).astype(int)
    frame["log_partition_count"] = np.log1p(frame["partition_count"].astype(float))
    frame["is_spectral"] = (frame["method"] == "spectral").astype(float)
    for feature in FEATURES:
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
    frame.to_csv(data_path, index=False)

    X = frame[list(FEATURES)]
    y = frame["best_choice"].to_numpy(dtype=int)
    groups = frame["dataset"].to_numpy()
    holdouts = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        train = frame.iloc[train_idx]
        test = frame.iloc[test_idx].copy()
        model = _estimator()
        model.fit(X.iloc[train_idx], y[train_idx])
        probabilities = model.predict_proba(X.iloc[test_idx])[:, list(model.classes_).index(1)]
        test["winner_probability"] = probabilities
        selected = test.sort_values(
            ["winner_probability", "cut_fraction", "method", "partitions"],
            ascending=[False, True, True, True], kind="stable",
        ).iloc[0]
        best_npv = float(test["observed_npv"].max())
        regret = (best_npv - float(selected["observed_npv"])) / max(abs(best_npv), 1.0)
        majority = _config_baseline(train, test, "winner_count")
        mean_regret = _config_baseline(train, test, "mean_regret")
        spatial_train = train[train["method"] == "spatial"]
        spatial_best = _config_baseline(
            spatial_train, test[test["method"] == "spatial"], "mean_regret",
            comparison_frame=test,
        )
        # Fixed, interpretable baselines evaluated on this held-out deposit.
        baseline_results = {
            "train_majority_winning_configuration": majority,
            "train_lowest_mean_regret_configuration": mean_regret,
            "train_selected_spatial_count": spatial_best,
        }
        for method, k in (("spatial", 4), ("spatial", 8), ("spatial", 12),
                          ("metis", 4), ("metis", 8), ("metis", 12), ("spectral", 4)):
            candidate = test[(test["method"] == method) & (test["partitions"] == k)].iloc[0]
            key = f"fixed_{method}_k{k}"
            baseline_results[key] = {
                "selected_method": method, "selected_partitions": k,
                "relative_npv_regret": float((best_npv - candidate["observed_npv"]) / max(abs(best_npv), 1.0)),
                "correct": bool(int(candidate["best_choice"]))
            }
        holdouts.append({
            "dataset": str(test["dataset"].iloc[0]),
            "selected_method": str(selected["method"]),
            "selected_partitions": int(selected["partitions"]),
            "observed_best_configs": [
                {"method": str(row.method), "partitions": int(row.partitions)}
                for row in test.loc[test["best_choice"] == 1].itertuples()
            ],
            "correct": bool(int(selected["best_choice"])),
            "relative_npv_regret": regret,
            "selected_reported_runtime_s": float(selected["partition_runtime_s"]
                + selected["local_solve_runtime_s"] + selected["refinement_runtime_s"]),
            "baselines": baseline_results,
        })

    # Train a separate all-data pilot artifact for future evaluation only.
    final_model = _estimator().fit(X, y)
    model_path = results_dir / "hybrid_v37_joint_selector_pilot.pkl"
    joblib.dump({
        "version": "3.7-joint-pilot",
        "model": final_model,
        "features": list(FEATURES),
        "candidate_methods": list(METHODS),
        "partition_counts": list(COUNTS),
        "label_definition": "highest observed final NPV among measured method/count settings per deposit",
        "not_for_deployment": True,
    }, model_path)
    baseline_names = [
        "train_majority_winning_configuration",
        "train_lowest_mean_regret_configuration",
        "train_selected_spatial_count",
        "fixed_spatial_k4", "fixed_spatial_k8", "fixed_spatial_k12",
        "fixed_metis_k4", "fixed_metis_k8", "fixed_metis_k12", "fixed_spectral_k4",
    ]
    baseline_summary = {}
    for name in baseline_names:
        results = [row["baselines"][name] for row in holdouts]
        baseline_summary[name] = {
            "accuracy": float(np.mean([result["correct"] for result in results])),
            "mean_relative_npv_regret": float(np.mean([result["relative_npv_regret"] for result in results])),
        }
    metrics = {
        "version": "3.7-joint-pilot",
        "candidate_count": len(frame),
        "independent_deposit_groups": len(DATASETS),
        "methods": list(METHODS),
        "partition_counts": list(COUNTS),
        "candidate_settings_per_deposit": int(frame.groupby("dataset").size().iloc[0]),
        "validation": "leave-one-deposit-out; candidate features are pre-solve graph and partition telemetry",
        "joint_selector_accuracy": float(np.mean([row["correct"] for row in holdouts])),
        "joint_selector_mean_relative_npv_regret": float(np.mean([row["relative_npv_regret"] for row in holdouts])),
        "joint_selector_mean_selected_reported_runtime_s": float(np.mean([row["selected_reported_runtime_s"] for row in holdouts])),
        "baseline_summary": baseline_summary,
        "holdouts": holdouts,
        "training_data": str(data_path),
        "pilot_model": str(model_path),
        "limitations": [
            "Only four independent deposits are available; counts and methods on one deposit are correlated candidate rows, not independent geology samples.",
            "Spectral has measured outcomes only at four partitions and its embedding emitted convergence-tolerance warnings on three large graphs.",
            "Historical k=8 Spatial/METIS candidate labels and their structural features are deterministically regenerated for this analysis.",
            "The fitted all-data artifact is not integrated into the solver because held-out results must beat fixed baselines before deployment.",
            "Reported selected runtimes include only the selected candidate's partition, solve, and repair time; a candidate-feature selector that generates every candidate would add their generation costs.",
        ],
    }
    metrics_path = results_dir / "hybrid_v37_joint_selector_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    run()
