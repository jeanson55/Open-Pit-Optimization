"""Partitioned local closure solve, merge, and boundary repair."""

from __future__ import annotations

import time
from typing import Dict, Set, Tuple

import networkx as nx

from graph.constraints import check_constraints
from .hpf import run_hpf
from .lg_repair import repair_with_metrics


def _violation_cluster_metrics(graph: nx.DiGraph, violations: list) -> dict:
    clusters = nx.Graph()
    clusters.add_edges_from(violations)
    components = list(nx.connected_components(clusters))
    component_of = {node: i for i, comp in enumerate(components) for node in comp}
    arc_counts = [0] * len(components)
    for u, v in violations:
        arc_counts[component_of[u]] += 1
    sorted_arc_counts = sorted(arc_counts, reverse=True)
    return {
        "violation_cluster_count": len(components),
        "largest_violation_cluster_nodes": max(map(len, components), default=0),
        "largest_violation_cluster_arcs": sorted_arc_counts[0] if sorted_arc_counts else 0,
        "violation_cluster_arcs_top10": sorted_arc_counts[:10],
    }


def solve_partitioned(graph: nx.DiGraph, partition_labels: Dict[int, int],
                      repair_mode: str = "add", local_solver=None
                      ) -> Tuple[Set[int], dict]:
    """Solve each part independently, merge, then refine to global feasibility.

    ``graph`` must contain precedence arcs only and each node must have a
    ``value`` attribute. Returns ``(final_solution, metrics)``.
    """
    if set(partition_labels) != set(graph.nodes):
        missing = set(graph.nodes) - set(partition_labels)
        extra = set(partition_labels) - set(graph.nodes)
        raise ValueError(f"partition labels must cover graph nodes (missing={len(missing)}, extra={len(extra)})")
    missing_values = [n for n in graph.nodes if "value" not in graph.nodes[n]]
    if missing_values:
        raise ValueError(f"graph nodes missing value attributes (count={len(missing_values)})")

    start = time.perf_counter()
    merged = run_hpf(graph, partition_labels, local_solver=local_solver)
    local_solve_runtime = time.perf_counter() - start

    initial_violations = check_constraints(graph, merged)
    values = nx.get_node_attributes(graph, "value")
    initial_npv = sum(values[n] for n in merged)

    refine_start = time.perf_counter()
    final_solution, cascade = repair_with_metrics(graph, merged, mode=repair_mode)
    refinement_runtime = time.perf_counter() - refine_start
    final_violations = check_constraints(graph, final_solution)
    if final_violations:
        raise RuntimeError(f"repair left {len(final_violations)} precedence violations")

    metrics = {
        "local_solve_runtime_s": local_solve_runtime,
        "initial_blocks_mined": len(merged),
        "initial_npv": float(initial_npv),
        "initial_violations": len(initial_violations),
        "refinement_runtime_s": refinement_runtime,
        "final_blocks_mined": len(final_solution),
        "final_violations": len(final_violations),
        "final_npv": float(sum(values[n] for n in final_solution)),
        "repair_mode": repair_mode,
        **cascade,
        **_violation_cluster_metrics(graph, initial_violations),
    }
    return final_solution, metrics
