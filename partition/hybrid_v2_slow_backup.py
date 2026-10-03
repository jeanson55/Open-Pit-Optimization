"""Precedence-aware hybrid partitioning.

Hybrid V2.1
-----------
Spatially seeded, precedence-aware boundary refinement.

Unlike the earlier V2 implementation, candidate moves are evaluated
incrementally. Moving one node can only change the cut status of arcs
incident to that node, so the algorithm does not repeatedly scan the
entire graph.

The method is deterministic and preserves the common partitioner interface:

    labels = hybrid_partition(graph, n_partitions=9)
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping

import networkx as nx

from .spatial import spatial_partition


BALANCE_TOLERANCE = 1.15
MAX_SWEEPS = 8

CRITICALITY_WEIGHT = 2.0
CRITICALITY_CAP = 5.0

# Small soft economic-balance term.
VALUE_BALANCE_WEIGHT = 0.15


def _safe_scale(values) -> float:
    nonzero = sorted(
        abs(float(v))
        for v in values
        if float(v) != 0.0
    )

    if not nonzero:
        return 1.0

    return max(nonzero[len(nonzero) // 2], 1e-12)


def _node_criticality(
    graph: nx.DiGraph,
    value_scale: float,
) -> Dict[int, float]:
    """Economic criticality assigned to each precedence source."""
    criticality = {}

    for node in graph.nodes:
        value = float(
            graph.nodes[node].get("value", 0.0)
        )

        if value > 0.0:
            bonus = min(
                value / value_scale,
                CRITICALITY_CAP,
            )

            criticality[node] = (
                1.0 + CRITICALITY_WEIGHT * bonus
            )
        else:
            criticality[node] = 1.0

    return criticality


def _boundary_nodes(
    graph: nx.DiGraph,
    labels: Mapping[int, int],
) -> set:
    """Nodes incident to at least one cross-partition precedence arc."""
    boundary = set()

    for u, v in graph.edges:
        if labels[u] != labels[v]:
            boundary.add(u)
            boundary.add(v)

    return boundary


def _candidate_parts(
    graph: nx.DiGraph,
    node: int,
    labels: Mapping[int, int],
) -> set:
    """Partitions adjacent to node through precedence relationships."""
    candidates = set()

    for neighbour in graph.successors(node):
        part = labels[neighbour]

        if part != labels[node]:
            candidates.add(part)

    for neighbour in graph.predecessors(node):
        part = labels[neighbour]

        if part != labels[node]:
            candidates.add(part)

    return candidates


def _weighted_arc_cost(
    graph: nx.DiGraph,
    source: int,
    criticality: Mapping[int, float],
) -> float:
    """Return the weighted cost of one cross-partition precedence arc."""
    return float(criticality[source])


def _node_cut_cost(
    graph: nx.DiGraph,
    node: int,
    labels: Mapping[int, int],
    criticality: Mapping[int, float],
) -> float:
    """Weighted cut contribution of all arcs incident to node.

    An arc is counted once when it crosses the partition boundary.
    """
    own = labels[node]
    cost = 0.0

    # Outgoing arcs: node is the source.
    for neighbour in graph.successors(node):
        if labels[neighbour] != own:
            cost += _weighted_arc_cost(
                graph,
                node,
                criticality,
            )

    # Incoming arcs: neighbour is the source.
    for neighbour in graph.predecessors(node):
        if labels[neighbour] != own:
            cost += _weighted_arc_cost(
                graph,
                neighbour,
                criticality,
            )

    return cost


def _move_delta(
    graph: nx.DiGraph,
    node: int,
    target: int,
    labels: Mapping[int, int],
    criticality: Mapping[int, float],
) -> float:
    """Return change in weighted cut caused by moving node.

    Negative = improvement.
    Positive = deterioration.

    Only arcs incident to ``node`` are examined.
    """
    source = labels[node]

    if source == target:
        return 0.0

    delta = 0.0

    # ------------------------------------------------------------
    # Outgoing arcs: node -> neighbour
    # ------------------------------------------------------------
    for neighbour in graph.successors(node):
        neighbour_part = labels[neighbour]
        weight = criticality[node]

        before = (
            weight
            if source != neighbour_part
            else 0.0
        )

        after = (
            weight
            if target != neighbour_part
            else 0.0
        )

        delta += after - before

    # ------------------------------------------------------------
    # Incoming arcs: neighbour -> node
    # ------------------------------------------------------------
    for neighbour in graph.predecessors(node):
        neighbour_part = labels[neighbour]
        weight = criticality[neighbour]

        before = (
            weight
            if neighbour_part != source
            else 0.0
        )

        after = (
            weight
            if neighbour_part != target
            else 0.0
        )

        delta += after - before

    return delta


def _part_value_sums(
    graph: nx.DiGraph,
    labels: Mapping[int, int],
) -> Dict[int, float]:
    sums = Counter()

    for node, part in labels.items():
        sums[part] += float(
            graph.nodes[node].get("value", 0.0)
        )

    return dict(sums)


def _balance_penalty(
    part_values: Mapping[int, float],
    total_abs_value: float,
) -> float:
    """Normalised economic-balance penalty."""
    if not part_values or total_abs_value <= 0.0:
        return 0.0

    values = list(part_values.values())
    mean = sum(values) / len(values)

    return sum(
        (value - mean) ** 2
        for value in values
    ) / (total_abs_value ** 2)


def _balance_delta(
    graph: nx.DiGraph,
    node: int,
    source: int,
    target: int,
    part_values: Mapping[int, float],
    total_abs_value: float,
) -> float:
    """Exact change in economic-balance penalty after a move."""
    if total_abs_value <= 0.0:
        return 0.0

    value = float(
        graph.nodes[node].get("value", 0.0)
    )

    before = _balance_penalty(
        part_values,
        total_abs_value,
    )

    updated = dict(part_values)
    updated[source] -= value
    updated[target] += value

    after = _balance_penalty(
        updated,
        total_abs_value,
    )

    return after - before


def _move_is_balanced(
    sizes: Mapping[int, int],
    source: int,
    target: int,
    n_nodes: int,
    n_partitions: int,
    tolerance: float,
) -> bool:
    mean = n_nodes / n_partitions

    if sizes[target] + 1 > mean * tolerance:
        return False

    if sizes[source] <= 1:
        return False

    return True


def hybrid_partition(
    graph: nx.DiGraph,
    n_partitions: int = 9,
    balance_tolerance: float = BALANCE_TOLERANCE,
    max_sweeps: int = MAX_SWEEPS,
) -> Dict[int, int]:
    """Create a spatially seeded, precedence-aware partition.

    Hybrid V2.1 uses local incremental move evaluation instead of repeatedly
    scanning the entire graph.
    """
    if n_partitions < 1:
        raise ValueError(
            "n_partitions must be >= 1"
        )

    if not graph.nodes:
        return {}

    if n_partitions > len(graph):
        raise ValueError(
            "n_partitions cannot exceed graph node count"
        )

    # ============================================================
    # Phase 1: spatial seed
    # ============================================================
    labels = spatial_partition(
        graph,
        n_partitions,
    )

    # ============================================================
    # Economic criticality
    # ============================================================
    values = {
        node: float(
            graph.nodes[node].get("value", 0.0)
        )
        for node in graph.nodes
    }

    value_scale = _safe_scale(
        values.values()
    )

    criticality = _node_criticality(
        graph,
        value_scale,
    )

    total_abs_value = sum(
        abs(value)
        for value in values.values()
    )

    sizes = Counter(
        labels.values()
    )

    part_values = _part_value_sums(
        graph,
        labels,
    )

    # Deterministic ordering.
    node_order = sorted(
        graph.nodes
    )

    # ============================================================
    # Phase 2: boundary refinement
    # ============================================================
    for sweep in range(max_sweeps):

        boundary = _boundary_nodes(
            graph,
            labels,
        )

        if not boundary:
            break

        moved = False

        for node in node_order:

            if node not in boundary:
                continue

            source = labels[node]

            candidates = sorted(
                _candidate_parts(
                    graph,
                    node,
                    labels,
                )
            )

            if not candidates:
                continue

            best_target = source
            best_delta = 0.0

            for target in candidates:

                if target == source:
                    continue

                if not _move_is_balanced(
                    sizes,
                    source,
                    target,
                    len(labels),
                    n_partitions,
                    balance_tolerance,
                ):
                    continue

                # ------------------------------------------------
                # Incremental precedence-cut change.
                # ------------------------------------------------
                cut_delta = _move_delta(
                    graph,
                    node,
                    target,
                    labels,
                    criticality,
                )

                # ------------------------------------------------
                # Small economic-balance correction.
                # ------------------------------------------------
                balance_delta = _balance_delta(
                    graph,
                    node,
                    source,
                    target,
                    part_values,
                    total_abs_value,
                )

                total_delta = (
                    cut_delta
                    + VALUE_BALANCE_WEIGHT
                    * balance_delta
                )

                if total_delta < best_delta - 1e-12:
                    best_delta = total_delta
                    best_target = target

            # ----------------------------------------------------
            # Accept only strictly improving moves.
            # ----------------------------------------------------
            if best_target != source:

                value = values[node]

                labels[node] = best_target

                sizes[source] -= 1
                sizes[best_target] += 1

                part_values[source] -= value
                part_values[best_target] += value

                moved = True

        if not moved:
            break

    return {
        int(node): int(part)
        for node, part in labels.items()
    }
