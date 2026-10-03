"""Economic-aware ranking of spatial and METIS pit partitions (V3.6)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from .metis import metis_partition
from .spatial import spatial_partition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "hybrid_v36_selector.pkl"

FEATURE_NAMES = [
    "log_graph_nodes",
    "log_precedence_arcs",
    "partition_count",
    "is_metis",
    "cut_fraction",
    "balance_ratio",
    "minimum_part_fraction",
    "maximum_part_fraction",
    "boundary_abs_value_fraction",
    "boundary_positive_value_fraction",
    "boundary_negative_value_fraction",
]

SCORE_WEIGHTS = {
    "ml_probability": 0.40,
    "economic_quality": 0.30,
    "balance_quality": 0.15,
    "cut_quality": 0.15,
}


def candidate_features(graph: nx.DiGraph, labels: Mapping[int, int], method: str) -> dict:
    """Return only features available before the partitioned solve is run."""
    if set(labels) != set(graph.nodes):
        raise ValueError("partition labels must cover every graph node exactly once")
    if not graph:
        raise ValueError("cannot rank a partition of an empty graph")
    if any("value" not in graph.nodes[node] for node in graph):
        raise ValueError("every graph node needs a 'value' attribute for economic ranking")

    sizes: dict[int, int] = {}
    for part in labels.values():
        sizes[part] = sizes.get(part, 0) + 1
    if len(sizes) < 2 or any(size == 0 for size in sizes.values()):
        raise ValueError("candidate must contain at least two non-empty partitions")

    cut_edges = [(u, v) for u, v in graph.edges if labels[u] != labels[v]]
    boundary_nodes = {node for edge in cut_edges for node in edge}
    values = {node: float(graph.nodes[node]["value"]) for node in graph}
    total_abs = sum(abs(value) for value in values.values())
    total_positive = sum(max(value, 0.0) for value in values.values())
    total_negative = sum(max(-value, 0.0) for value in values.values())
    boundary_abs = sum(abs(values[node]) for node in boundary_nodes)
    boundary_positive = sum(max(values[node], 0.0) for node in boundary_nodes)
    boundary_negative = sum(max(-values[node], 0.0) for node in boundary_nodes)
    n_nodes = graph.number_of_nodes()
    n_arcs = graph.number_of_edges()
    balance = max(sizes.values()) / min(sizes.values())

    return {
        "log_graph_nodes": math.log1p(n_nodes),
        "log_precedence_arcs": math.log1p(n_arcs),
        "partition_count": float(len(sizes)),
        "is_metis": float(method.lower().startswith("metis")),
        "cut_fraction": len(cut_edges) / max(n_arcs, 1),
        "balance_ratio": float(balance),
        "minimum_part_fraction": min(sizes.values()) / n_nodes,
        "maximum_part_fraction": max(sizes.values()) / n_nodes,
        "boundary_abs_value_fraction": boundary_abs / max(total_abs, 1e-12),
        "boundary_positive_value_fraction": boundary_positive / max(total_positive, 1e-12),
        "boundary_negative_value_fraction": boundary_negative / max(total_negative, 1e-12),
        "cut_arcs": len(cut_edges),
        "total_precedence_arcs": n_arcs,
        "boundary_nodes": len(boundary_nodes),
        "economic_quality": 1.0 - boundary_abs / max(total_abs, 1e-12),
        "balance_quality": 1.0 / max(balance, 1.0),
        "cut_quality": 1.0 - len(cut_edges) / max(n_arcs, 1),
    }


def _relative_quality(values: Sequence[float]) -> list[float]:
    low, high = min(values), max(values)
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-15):
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def rank_candidates(
    graph: nx.DiGraph,
    candidates: Sequence[tuple[str, Mapping[int, int]]],
    model,
    score_weights: Mapping[str, float] | None = None,
) -> list[dict]:
    """Rank candidates by ML winner probability and relative economic/graph quality."""
    if not candidates:
        raise ValueError("at least one partition candidate is required")
    weights = dict(SCORE_WEIGHTS if score_weights is None else score_weights)
    if set(weights) != set(SCORE_WEIGHTS):
        raise ValueError(f"score weights must have exactly these keys: {sorted(SCORE_WEIGHTS)}")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("score weights must sum to 1")

    model_features = model["features"] if isinstance(model, dict) else FEATURE_NAMES
    estimator = model["model"] if isinstance(model, dict) else model
    if list(model_features) != FEATURE_NAMES:
        raise ValueError("V3.6 model feature schema does not match this selector")

    records = []
    for name, labels in candidates:
        metrics = candidate_features(graph, labels, name)
        row = {feature: metrics[feature] for feature in FEATURE_NAMES}
        records.append({"method": name, "labels": labels, "metrics": metrics, "row": row})

    probabilities = estimator.predict_proba(
        pd.DataFrame([record["row"] for record in records], columns=FEATURE_NAMES)
    )
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("V3.6 classifier has no positive best-candidate class")
    positive_index = classes.index(1)
    for record, probability in zip(records, probabilities):
        record["ml_probability"] = float(probability[positive_index])

    normalized = {
        "economic_quality": _relative_quality([r["metrics"]["economic_quality"] for r in records]),
        "balance_quality": _relative_quality([r["metrics"]["balance_quality"] for r in records]),
        "cut_quality": _relative_quality([r["metrics"]["cut_quality"] for r in records]),
    }
    for index, record in enumerate(records):
        record["score_components"] = {
            "ml_probability": record["ml_probability"],
            **{name: values[index] for name, values in normalized.items()},
        }
        record["score"] = sum(weights[name] * record["score_components"][name] for name in weights)

    records.sort(key=lambda r: (
        -r["score"],
        r["metrics"]["cut_fraction"],
        r["metrics"]["balance_ratio"],
        r["method"],
    ))
    return records


def hybrid_v36_partition(
    graph: nx.DiGraph,
    n_partitions: int = 8,
    dataset_id: int | None = None,
    model_path: str | Path | None = None,
) -> dict[int, int]:
    """Generate spatial/METIS candidates and return the highest-ranked labels.

    ``dataset_id`` is accepted to match the experiment runner interface but is
    deliberately unused: dataset identity is not a V3.6 model feature.
    """
    del dataset_id
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"V3.6 selector model not found: {path}")
    bundle = joblib.load(path)
    if bundle.get("version") != "3.6":
        raise ValueError(f"unsupported V3.6 selector model version: {bundle.get('version')!r}")

    candidates = [
        ("Spatial", spatial_partition(graph, n_partitions)),
        ("METIS", metis_partition(graph, n_partitions)),
    ]
    ranked = rank_candidates(graph, candidates, bundle)
    for record in ranked:
        m = record["metrics"]
        print(
            f"Hybrid V3.6 {record['method']}: score={record['score']:.4f} "
            f"ML={record['ml_probability']:.4f} economic={m['economic_quality']:.4f} "
            f"balance={m['balance_ratio']:.4f} cut={m['cut_fraction']:.5f}"
        )
    print(f"Hybrid V3.6 selected: {ranked[0]['method']}")
    graph.graph["hybrid_v36_selection"] = {
        "selected_method": ranked[0]["method"],
        "score_weights": SCORE_WEIGHTS,
        "ranked_candidates": [
            {
                "method": record["method"],
                "score": record["score"],
                "ml_probability": record["ml_probability"],
                "score_components": record["score_components"],
                "cut_arcs": record["metrics"]["cut_arcs"],
                "cut_fraction": record["metrics"]["cut_fraction"],
                "balance_ratio": record["metrics"]["balance_ratio"],
                "boundary_nodes": record["metrics"]["boundary_nodes"],
                "boundary_abs_value_fraction": record["metrics"]["boundary_abs_value_fraction"],
            }
            for record in ranked
        ],
    }
    return dict(ranked[0]["labels"])
