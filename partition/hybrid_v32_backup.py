"""
Hybrid V3.2 partitioning.

Strategy
--------
Generate Spatial and METIS candidate partitions and select between them using
a richer pre-solve graph/economic risk model.

V3.2 adds:
    - normalized precedence cut
    - node-count imbalance
    - economic-value imbalance
    - boundary-node fraction
    - boundary economic exposure
    - economically weighted precedence cut

Unlike V3.1, the metrics are normalized relative to the two candidates.
This prevents one metric's numerical scale from dominating the selector.

Important:
    V3.2 is still a pre-solve selector. It does NOT claim to predict final
    repaired NPV. That requires a later pilot/surrogate stage.
"""

from typing import Dict, Tuple

import networkx as nx

from .metis import metis_partition
from .spatial import spatial_partition


# ---------------------------------------------------------------------------
# V3.2 weights
# ---------------------------------------------------------------------------

CUT_WEIGHT = 0.25
SIZE_WEIGHT = 0.10
VALUE_WEIGHT = 0.10
BOUNDARY_WEIGHT = 0.15
EXPOSURE_WEIGHT = 0.15
ECONOMIC_CUT_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Basic statistics
# ---------------------------------------------------------------------------

def _partition_stats(
    graph: nx.DiGraph,
    labels: Dict[int, int],
) -> Tuple[Dict[int, int], Dict[int, float]]:
    sizes = {}
    values = {}

    for node, part in labels.items():
        sizes[part] = sizes.get(part, 0) + 1

        value = float(graph.nodes[node].get("value", 0.0))
        values[part] = values.get(part, 0.0) + value

    return sizes, values


def _size_penalty(sizes: Dict[int, int]) -> float:
    if not sizes:
        return 0.0

    mean_size = sum(sizes.values()) / len(sizes)

    if mean_size <= 0.0:
        return 0.0

    return sum(
        ((size - mean_size) / mean_size) ** 2
        for size in sizes.values()
    ) / len(sizes)


def _value_penalty(values: Dict[int, float]) -> float:
    if not values:
        return 0.0

    mean_value = sum(values.values()) / len(values)

    scale = max(abs(mean_value), 1.0)

    return sum(
        ((value - mean_value) / scale) ** 2
        for value in values.values()
    ) / len(values)


# ---------------------------------------------------------------------------
# Boundary / precedence metrics
# ---------------------------------------------------------------------------

def _boundary_metrics(
    graph: nx.DiGraph,
    labels: Dict[int, int],
):
    """
    Calculate structural and economic exposure at partition boundaries.

    Returns
    -------
    cut_count:
        Number of precedence arcs crossing partitions.

    boundary_nodes:
        Number of nodes touching at least one cross-partition arc.

    weighted_cut:
        Cross-partition precedence cut weighted by endpoint economic exposure.

    boundary_abs_value:
        Sum of absolute economic value of boundary nodes.

    total_abs_value:
        Sum of absolute economic value of all nodes.
    """

    boundary_nodes = set()
    cut_count = 0
    weighted_cut = 0.0

    for u, v in graph.edges:
        if labels[u] == labels[v]:
            continue

        cut_count += 1
        boundary_nodes.add(u)
        boundary_nodes.add(v)

        u_value = abs(float(graph.nodes[u].get("value", 0.0)))
        v_value = abs(float(graph.nodes[v].get("value", 0.0)))

        weighted_cut += u_value + v_value

    total_abs_value = sum(
        abs(float(graph.nodes[node].get("value", 0.0)))
        for node in graph.nodes
    )

    boundary_abs_value = sum(
        abs(float(graph.nodes[node].get("value", 0.0)))
        for node in boundary_nodes
    )

    return {
        "cut_count": cut_count,
        "boundary_nodes": len(boundary_nodes),
        "weighted_cut": weighted_cut,
        "boundary_abs_value": boundary_abs_value,
        "total_abs_value": total_abs_value,
    }


def _raw_metrics(
    graph: nx.DiGraph,
    labels: Dict[int, int],
) -> Dict[str, float]:

    sizes, values = _partition_stats(graph, labels)

    boundary = _boundary_metrics(graph, labels)

    total_edges = graph.number_of_edges()
    total_nodes = graph.number_of_nodes()

    cut_fraction = (
        boundary["cut_count"] / total_edges
        if total_edges
        else 0.0
    )

    boundary_fraction = (
        boundary["boundary_nodes"] / total_nodes
        if total_nodes
        else 0.0
    )

    boundary_exposure = (
        boundary["boundary_abs_value"]
        / boundary["total_abs_value"]
        if boundary["total_abs_value"] > 0.0
        else 0.0
    )

    economic_cut_fraction = (
        boundary["weighted_cut"]
        / max(boundary["total_abs_value"], 1.0)
    )

    return {
        "cut_fraction": cut_fraction,
        "size_penalty": _size_penalty(sizes),
        "value_penalty": _value_penalty(values),
        "boundary_fraction": boundary_fraction,
        "boundary_exposure": boundary_exposure,
        "economic_cut_fraction": economic_cut_fraction,
        "cut_count": boundary["cut_count"],
        "boundary_nodes": boundary["boundary_nodes"],
    }


# ---------------------------------------------------------------------------
# Candidate-relative normalization
# ---------------------------------------------------------------------------

def _normalize_pair(
    a: float,
    b: float,
) -> Tuple[float, float]:
    """
    Normalize two candidate values to [0, 1].

    Lower is always better.

    If both candidates have the same value, both receive 0.0 because the
    metric provides no information for choosing between them.
    """

    low = min(a, b)
    high = max(a, b)

    if high - low <= 1e-15:
        return 0.0, 0.0

    return (
        (a - low) / (high - low),
        (b - low) / (high - low),
    )


def _score_candidates(
    spatial: Dict[str, float],
    metis: Dict[str, float],
):
    """
    Score Spatial and METIS using candidate-relative normalized metrics.
    """

    metric_names = (
        "cut_fraction",
        "size_penalty",
        "value_penalty",
        "boundary_fraction",
        "boundary_exposure",
        "economic_cut_fraction",
    )

    weights = {
        "cut_fraction": CUT_WEIGHT,
        "size_penalty": SIZE_WEIGHT,
        "value_penalty": VALUE_WEIGHT,
        "boundary_fraction": BOUNDARY_WEIGHT,
        "boundary_exposure": EXPOSURE_WEIGHT,
        "economic_cut_fraction": ECONOMIC_CUT_WEIGHT,
    }

    spatial_score = 0.0
    metis_score = 0.0

    normalized = {}

    for name in metric_names:
        s_norm, m_norm = _normalize_pair(
            spatial[name],
            metis[name],
        )

        normalized[name] = {
            "spatial": s_norm,
            "metis": m_norm,
        }

        spatial_score += weights[name] * s_norm
        metis_score += weights[name] * m_norm

    return spatial_score, metis_score, normalized


# ---------------------------------------------------------------------------
# Balance helper
# ---------------------------------------------------------------------------

def _balance_ratio(
    graph: nx.DiGraph,
    labels: Dict[int, int],
) -> float:

    sizes = {}

    for node, part in labels.items():
        sizes[part] = sizes.get(part, 0) + 1

    if not sizes:
        return 0.0

    minimum = min(sizes.values())
    maximum = max(sizes.values())

    if minimum <= 0:
        return float("inf")

    return maximum / minimum


# ---------------------------------------------------------------------------
# Hybrid V3.2
# ---------------------------------------------------------------------------

def hybrid_partition(
    graph: nx.DiGraph,
    n_partitions: int = 9,
) -> Dict[int, int]:
    """
    Generate Spatial and METIS candidates and select the lower-risk
    partition according to the V3.2 pre-solve score.

    Interface intentionally remains identical to V3.1.
    """

    print(
        "  Hybrid V3.2: generating Spatial candidate..."
    )

    spatial_labels = spatial_partition(
        graph,
        n_partitions,
    )

    print(
        "  Hybrid V3.2: evaluating Spatial candidate..."
    )

    spatial_metrics = _raw_metrics(
        graph,
        spatial_labels,
    )

    print(
        "  Hybrid V3.2: generating METIS candidate..."
    )

    metis_labels = metis_partition(
        graph,
        n_partitions,
    )

    print(
        "  Hybrid V3.2: evaluating METIS candidate..."
    )

    metis_metrics = _raw_metrics(
        graph,
        metis_labels,
    )

    (
        spatial_score,
        metis_score,
        normalized,
    ) = _score_candidates(
        spatial_metrics,
        metis_metrics,
    )

    spatial_balance = _balance_ratio(
        graph,
        spatial_labels,
    )

    metis_balance = _balance_ratio(
        graph,
        metis_labels,
    )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    print(
        f"  Spatial V3.2:"
        f" score={spatial_score:.6f}"
        f" | cut={spatial_metrics['cut_count']:,}"
        f" | balance={spatial_balance:.4f}"
        f" | boundary={spatial_metrics['boundary_nodes']:,}"
        f" | exposure={spatial_metrics['boundary_exposure']:.4f}"
    )

    print(
        f"  METIS V3.2:"
        f" score={metis_score:.6f}"
        f" | cut={metis_metrics['cut_count']:,}"
        f" | balance={metis_balance:.4f}"
        f" | boundary={metis_metrics['boundary_nodes']:,}"
        f" | exposure={metis_metrics['boundary_exposure']:.4f}"
    )

    print(
        "  Hybrid V3.2 normalized comparison:"
    )

    for metric, values in normalized.items():
        print(
            f"    {metric}: "
            f"Spatial={values['spatial']:.4f}, "
            f"METIS={values['metis']:.4f}"
        )

    # -----------------------------------------------------------------------
    # Selection
    # -----------------------------------------------------------------------

    if spatial_score <= metis_score:
        print(
            "  Hybrid V3.2: "
            f"selected Spatial (score={spatial_score:.6f})"
        )
        return spatial_labels

    print(
        "  Hybrid V3.2: "
        f"selected METIS (score={metis_score:.6f})"
    )

    return metis_labels