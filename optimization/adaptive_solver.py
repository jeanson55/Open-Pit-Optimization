"""Adaptive per-partition LG / Pseudoflow solve with global repair.

This version preserves the original optimization workflow while adding an
optional, non-invasive trace of block-level states for visualization.

Default behaviour is unchanged:
    final_solution, metrics = solve_adaptive(...)

When ``return_trace=True``:
    final_solution, metrics, trace = solve_adaptive(..., return_trace=True)

The trace contains actual graph node IDs, not reconstructed/aggregate states.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional

import networkx as nx
import numpy as np

from graph.constraints import check_constraints
from gpit.lg_literal import LiteralLG
from optimization.lg_repair import repair_with_metrics
from optimization.partitioned import _violation_cluster_metrics


FEATURE_NAMES = (
    "node_count", "arc_count", "arcs_per_node", "mean_in_degree",
    "mean_out_degree", "max_out_degree", "out_degree_cv",
    "positive_value_fraction", "negative_value_fraction", "value_mean",
    "value_std", "absolute_value_sum", "near_zero_value_fraction",
)


class RuntimeChoiceModel:
    """Predict both solver runtimes and return the faster solver label."""

    def __init__(self, lg_regressor: Any, hpf_regressor: Any):
        self.lg_regressor = lg_regressor
        self.hpf_regressor = hpf_regressor

    def predict(self, features):
        lg_log_seconds = np.asarray(
            self.lg_regressor.predict(features), dtype=np.float64
        )
        hpf_log_seconds = np.asarray(
            self.hpf_regressor.predict(features), dtype=np.float64
        )
        return np.where(lg_log_seconds < hpf_log_seconds, "LG", "HPF")


def partition_features(graph: nx.DiGraph, values: Mapping[int, float]) -> dict:
    """Return topology and economic descriptors available before solving."""
    nodes = list(graph.nodes)
    n = len(nodes)
    m = graph.number_of_edges()

    weights = np.asarray(
        [values[node] for node in nodes], dtype=np.float64
    )
    out_degree = np.asarray(
        [graph.out_degree(node) for node in nodes], dtype=np.float64
    )
    in_degree = np.asarray(
        [graph.in_degree(node) for node in nodes], dtype=np.float64
    )

    mean_out = float(out_degree.mean()) if n else 0.0
    std_out = float(out_degree.std()) if n else 0.0
    scale = float(np.mean(np.abs(weights))) if n else 0.0

    return {
        "node_count": n,
        "arc_count": m,
        "arcs_per_node": m / n if n else 0.0,
        "mean_in_degree": float(in_degree.mean()) if n else 0.0,
        "mean_out_degree": mean_out,
        "max_out_degree": int(out_degree.max()) if n else 0,
        "out_degree_cv": std_out / mean_out if mean_out else 0.0,
        "positive_value_fraction": float(np.mean(weights > 0)) if n else 0.0,
        "negative_value_fraction": float(np.mean(weights < 0)) if n else 0.0,
        "value_mean": float(weights.mean()) if n else 0.0,
        "value_std": float(weights.std()) if n else 0.0,
        "absolute_value_sum": float(np.abs(weights).sum()),
        "near_zero_value_fraction": (
            float(np.mean(np.abs(weights) < max(0.005, scale * 1e-6)))
            if n else 0.0
        ),
    }


def _solve_hpf(
    graph: nx.DiGraph,
    values: Mapping[int, float],
    required=(),
    forbidden=(),
) -> set:
    """Run the installed pseudoflow engine on a compact arbitrary DAG."""
    import pseudoflow

    nodes = list(graph.nodes)
    n = len(nodes)
    required = set(required)
    forbidden = set(forbidden)

    if required & forbidden:
        raise ValueError("a boundary node cannot be both required and forbidden")
    if not (required | forbidden) <= set(nodes):
        raise ValueError(
            "required and forbidden nodes must be inside the local graph"
        )

    index = {node: i + 1 for i, node in enumerate(nodes)}
    source, sink = 0, n + 1

    network = nx.DiGraph()
    network.add_nodes_from((source, sink))

    rounded = {
        node: float(np.round(float(values[node]), 2))
        for node in nodes
    }
    hard_capacity = max(
        99e99,
        sum(abs(v) for v in rounded.values()) + 1.0,
    )

    for node, value in rounded.items():
        i = index[node]
        if value >= 0:
            network.add_edge(source, i, const=value, mult=1)
        else:
            network.add_edge(i, sink, const=-value, mult=-1)

    for node in required:
        network.add_edge(
            source, index[node], const=hard_capacity, mult=1
        )
    for node in forbidden:
        network.add_edge(
            index[node], sink, const=hard_capacity, mult=-1
        )

    for u, v in graph.edges:
        network.add_edge(
            index[u], index[v], const=hard_capacity, mult=1
        )

    _, cuts, _ = pseudoflow.hpf(
        network,
        source,
        sink,
        const_cap="const",
        mult_cap="mult",
        lambdaRange=[0],
        roundNegativeCapacity=False,
    )

    return {
        node
        for node in nodes
        if cuts.get(index[node]) == [1]
    }


def _refine_boundary(
    graph: nx.DiGraph,
    partition_labels: Mapping[int, int],
    merged: set,
    repair_mode: str,
    max_iterations: int = 8,
    return_trace: bool = False,
):
    """Repair and reoptimize the mutable cross-partition boundary.

    When ``return_trace`` is enabled, all block sets are captured directly
    from the computation. No block-level state is inferred from aggregate
    metrics.
    """
    if return_trace:
        repaired, cascade, repair_trace = repair_with_metrics(
            graph, merged, mode=repair_mode, return_trace=True
        )
    else:
        repaired, cascade = repair_with_metrics(
            graph, merged, mode=repair_mode
        )
        repair_trace = None

    if repair_mode != "add":
        raise ValueError(
            "boundary refinement currently requires add-mode repair"
        )

    values = nx.get_node_attributes(graph, "value")
    incumbent_npv = float(
        sum(values[node] for node in repaired)
    )

    boundary = {
        node
        for u, v in graph.edges
        if partition_labels[u] != partition_labels[v]
        for node in (u, v)
    }
    boundary.update(repaired - merged)

    trace = {
        "merged": set(merged),
        "repaired": set(repaired),
        "boundary_initial": set(boundary),
        "boundary_final": set(boundary),
        "boundary_history": [set(boundary)],
        "boundary_iterations": [],
    }
    if repair_trace is not None:
        trace.update(repair_trace)

    if not boundary:
        info = {
            **cascade,
            "boundary_nodes": 0,
            "boundary_arcs": 0,
            "boundary_refinement_iterations": 0,
            "boundary_solver_runtime_s": 0.0,
            "boundary_reoptimized_added": 0,
            "boundary_reoptimized_removed": 0,
            "boundary_seed_npv": incumbent_npv,
            "boundary_npv_gain": 0.0,
        }
        return (repaired, info, trace) if return_trace else (repaired, info)

    candidate = set(repaired)
    elapsed = 0.0
    iterations = 0

    for iteration in range(max_iterations):
        inside = boundary
        outside_selected = candidate - inside

        required, forbidden = set(), set()
        for u, v in graph.edges:
            u_in, v_in = u in inside, v in inside

            if not u_in and v_in and u in outside_selected:
                required.add(v)
            elif u_in and not v_in and v not in candidate:
                forbidden.add(u)

        conflict = required & forbidden
        if conflict:
            raise RuntimeError(
                "boundary pins conflict for nodes "
                f"{sorted(conflict, key=str)[:10]}"
            )

        local = graph.subgraph(inside)
        started = time.perf_counter()

        local_solution = _solve_hpf(
            local,
            {node: values[node] for node in inside},
            required=required,
            forbidden=forbidden,
        )

        elapsed += time.perf_counter() - started
        iterations += 1

        updated = outside_selected | local_solution
        violations = check_constraints(graph, updated)

        iteration_record = {
            "iteration": iteration + 1,
            "boundary_nodes": len(boundary),
            "required_nodes": set(required),
            "forbidden_nodes": set(forbidden),
            "candidate_before": set(candidate),
            "candidate_after": set(updated),
            "violations": list(violations),
        }

        if not violations:
            candidate = updated
            trace["boundary_iterations"].append(iteration_record)
            break

        before = len(boundary)
        boundary.update(
            node for edge in violations for node in edge
        )
        candidate = updated

        iteration_record["boundary_nodes_after"] = len(boundary)
        trace["boundary_iterations"].append(iteration_record)
        trace["boundary_history"].append(set(boundary))

        if len(boundary) == before:
            break

    remaining = check_constraints(graph, candidate)

    if remaining:
        prior_depth = int(cascade.get("cascade_depth", 0))
        if return_trace:
            candidate, extra, extra_trace = repair_with_metrics(
                graph, candidate, mode="add", return_trace=True
            )
            trace["repair_added"].update(extra_trace["repair_added"])
            for level, nodes in extra_trace["repair_added_by_level"].items():
                trace["repair_added_by_level"][prior_depth + int(level)] = set(nodes)
            for node, level in extra_trace["repair_level_by_node"].items():
                trace["repair_level_by_node"][node] = prior_depth + int(level)
            trace["repair_final_solution"] = set(candidate)
        else:
            candidate, extra = repair_with_metrics(
                graph, candidate, mode="add"
            )
        cascade = {
            **cascade,
            "cascade_added_blocks": (
                cascade["cascade_added_blocks"]
                + extra["cascade_added_blocks"]
            ),
            "cascade_removed_blocks": 0,
            "cascade_depth": (
                cascade["cascade_depth"]
                + extra["cascade_depth"]
            ),
        }

    candidate_npv = float(
        sum(values[node] for node in candidate)
    )

    if candidate_npv + 1e-9 < incumbent_npv:
        candidate = repaired
        candidate_npv = incumbent_npv

    trace["boundary_final"] = set(boundary)
    trace["boundary_refined"] = set(candidate)
    trace["boundary_reoptimized_added"] = set(candidate - repaired)
    trace["boundary_reoptimized_removed"] = set(repaired - candidate)

    info = {
        **cascade,
        "boundary_nodes": len(boundary),
        "boundary_arcs": graph.subgraph(boundary).number_of_edges(),
        "boundary_refinement_iterations": iterations,
        "boundary_solver_runtime_s": elapsed,
        "boundary_reoptimized_added": len(candidate - repaired),
        "boundary_reoptimized_removed": len(repaired - candidate),
        "boundary_seed_npv": incumbent_npv,
        "boundary_npv_gain": candidate_npv - incumbent_npv,
    }

    return (candidate, info, trace) if return_trace else (candidate, info)


def _choose_solver(
    features: Mapping[str, float],
    policy: str,
    model: Optional[Any],
    lg_node_limit: int,
) -> str:
    if model is not None:
        import pandas as pd

        row = pd.DataFrame(
            [{
                name: features[name]
                for name in FEATURE_NAMES
            }],
            columns=FEATURE_NAMES,
        )
        choice = str(model.predict(row)[0]).upper()

        if choice not in {"LG", "HPF"}:
            raise ValueError(
                f"ML policy returned unsupported solver {choice!r}"
            )

    elif policy == "heuristic":
        estimate = (
            features["node_count"]
            + features["arc_count"]
        )
        choice = (
            "LG"
            if (
                features["node_count"] <= 128
                and estimate <= 2048
            )
            else "HPF"
        )
    else:
        raise ValueError(
            "policy must be 'heuristic', or pass a fitted model"
        )

    if choice == "LG" and features["node_count"] > lg_node_limit:
        return "HPF"

    return choice


def solve_adaptive(
    graph: nx.DiGraph,
    partition_labels: Dict[int, int],
    repair_mode: str = "add",
    policy: str = "heuristic",
    model: Optional[Any] = None,
    lg_node_limit: int = 4096,
    refinement: str = "boundary",
    return_trace: bool = False,
):
    """Solve labeled partitions with adaptive LG/HPF.

    Default return:
        ``(final_solution, metrics)``

    Visualization return:
        ``(final_solution, metrics, trace)``

    The trace records actual node IDs for the local, merged, repaired,
    boundary and final states.
    """
    if set(partition_labels) != set(graph.nodes):
        raise ValueError(
            "partition labels must cover every graph node exactly once"
        )

    if any(
        "value" not in graph.nodes[node]
        for node in graph.nodes
    ):
        raise ValueError(
            "every graph node must have a 'value' attribute"
        )

    if 0 in graph:
        raise ValueError(
            "node 0 is reserved as the HPF source"
        )

    started = time.perf_counter()
    merged, decisions = set(), []
    partition_traces = {}

    for part in sorted(set(partition_labels.values())):
        members = [
            node
            for node in graph
            if partition_labels[node] == part
        ]
        subgraph = graph.subgraph(members)
        values = {
            node: float(graph.nodes[node]["value"])
            for node in members
        }

        features = partition_features(
            subgraph, values
        )
        solver = _choose_solver(
            features,
            policy,
            model,
            lg_node_limit,
        )

        solve_start = time.perf_counter()

        if solver == "LG":
            precedence = {
                node: list(subgraph.successors(node))
                for node in members
            }
            engine = LiteralLG(
                values=values,
                precedence=precedence,
            )
            engine.solve()
            local_solution = engine.closure()
        else:
            local_solution = _solve_hpf(
                subgraph, values
            )

        solve_seconds = (
            time.perf_counter() - solve_start
        )

        merged.update(local_solution)

        decisions.append({
            "partition": int(part),
            "solver": solver,
            "features": features,
            "local_solve_runtime_s": solve_seconds,
            "local_blocks": len(local_solution),
            "local_npv": float(
                sum(
                    values[node]
                    for node in local_solution
                )
            ),
        })

        if return_trace:
            partition_traces[int(part)] = {
                "members": set(members),
                "local_solution": set(local_solution),
            }

    initial_violations = check_constraints(
        graph, merged
    )
    values_all = nx.get_node_attributes(
        graph, "value"
    )
    initial_npv = float(
        sum(values_all[node] for node in merged)
    )

    refine_started = time.perf_counter()

    if refinement == "boundary":
        if return_trace:
            (
                final_solution,
                cascade,
                refinement_trace,
            ) = _refine_boundary(
                graph,
                partition_labels,
                merged,
                repair_mode,
                return_trace=True,
            )
        else:
            final_solution, cascade = _refine_boundary(
                graph,
                partition_labels,
                merged,
                repair_mode,
            )
            refinement_trace = None

    elif refinement == "add":
        if return_trace:
            (
                final_solution,
                cascade,
                repair_trace,
            ) = repair_with_metrics(
                graph,
                merged,
                mode=repair_mode,
                return_trace=True,
            )
            refinement_trace = repair_trace
        else:
            final_solution, cascade = (
                repair_with_metrics(
                    graph,
                    merged,
                    mode=repair_mode,
                )
            )
            refinement_trace = None
    else:
        raise ValueError(
            "refinement must be 'boundary' or 'add'"
        )

    refine_seconds = (
        time.perf_counter() - refine_started
    )

    final_violations = check_constraints(
        graph, final_solution
    )

    if final_violations:
        raise RuntimeError(
            "global repair left "
            f"{len(final_violations)} "
            "precedence violations"
        )

    metrics = {
        "policy": (
            "ml" if model is not None
            else policy
        ),
        "partitions": decisions,
        "solver_counts": {
            name: sum(
                d["solver"] == name
                for d in decisions
            )
            for name in ("LG", "HPF")
        },
        "local_solve_runtime_s": sum(
            d["local_solve_runtime_s"]
            for d in decisions
        ),
        "total_runtime_s": (
            time.perf_counter() - started
        ),
        "initial_blocks_mined": len(merged),
        "initial_npv": initial_npv,
        "initial_violations": len(
            initial_violations
        ),
        "refinement_runtime_s": refine_seconds,
        "final_blocks_mined": len(final_solution),
        "final_violations": len(final_violations),
        "final_npv": float(
            sum(
                values_all[node]
                for node in final_solution
            )
        ),
        "repair_mode": repair_mode,
        "refinement": refinement,
        **cascade,
        **_violation_cluster_metrics(
            graph,
            initial_violations,
        ),
    }

    if not return_trace:
        return final_solution, metrics

    trace = {
        "partition_labels": dict(partition_labels),
        "partition_traces": partition_traces,
        "merged": set(merged),
        "initial_violations": list(initial_violations),
        "repaired": set(
            refinement_trace.get(
                "repaired",
                refinement_trace.get(
                    "repair_final_solution", merged
                ),
            )
        ),
        "final_solution": set(final_solution),
        "refinement": refinement,
        "cascade": dict(cascade),
    }

    if refinement_trace:
        trace.update(refinement_trace)

    return final_solution, metrics, trace
