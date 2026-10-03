"""Partition quality metrics used consistently across methods."""

from __future__ import annotations

from collections import Counter
from typing import Dict

import networkx as nx


def evaluate_partition(
    graph: nx.DiGraph,
    labels: Dict[int, int],
) -> Dict[str, float]:
    """Return edge-cut, balance and partition-size statistics."""
    if set(labels) != set(graph.nodes):
        raise ValueError(
            "Partition labels must cover every graph node exactly once."
        )

    total_edges = graph.number_of_edges()
    cut_edges = sum(
        1
        for u, v in graph.edges
        if labels[u] != labels[v]
    )

    sizes = Counter(labels.values())
    k = len(sizes)
    mean = len(labels) / k if k else 0.0

    return {
        "edge_cut": (
            cut_edges / total_edges
            if total_edges
            else 0.0
        ),
        "cut_edges": cut_edges,
        "total_edges": total_edges,
        "balance": (
            max(sizes.values()) / mean
            if mean
            else float("inf")
        ),
        # Retained for compatibility with existing callers.
        "violations": cut_edges,
        "n_partitions": k,
        "min_part": (
            min(sizes.values()) if sizes else 0
        ),
        "max_part": (
            max(sizes.values()) if sizes else 0
        ),
    }
