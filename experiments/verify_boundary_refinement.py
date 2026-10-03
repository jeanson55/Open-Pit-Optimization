"""Exhaustive small-graph checks for HPF closure and boundary refinement.

This is a deterministic research verification experiment, not a proof for
arbitrary graphs. It checks exact maximum-closure values on small DAGs and
checks that global boundary refinement preserves feasibility and does not
reduce the feasible add-only incumbent.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.constraints import check_constraints
from optimization.adaptive_solver import _refine_boundary, _solve_hpf
from optimization.lg_repair import repair_with_metrics


def _is_feasible(graph: nx.DiGraph, selected: set[int], required=(), forbidden=()) -> bool:
    return (set(required) <= selected and not (set(forbidden) & selected)
            and all(u not in selected or v in selected for u, v in graph.edges))


def _enumerate_optimum(graph: nx.DiGraph, required=(), forbidden=()):
    nodes = list(graph.nodes)
    values = nx.get_node_attributes(graph, "value")
    best_value = float("-inf")
    best_set = None
    for mask in range(1 << len(nodes)):
        chosen = {nodes[i] for i in range(len(nodes)) if mask & (1 << i)}
        if _is_feasible(graph, chosen, required, forbidden):
            objective = sum(values[node] for node in chosen)
            if objective > best_value:
                best_value, best_set = objective, chosen
    if best_set is None:
        raise AssertionError("verification case unexpectedly has no feasible closure")
    return best_value, best_set


def _random_dag(rng: random.Random, node_count: int, case_id: int) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in range(1, node_count + 1):
        graph.add_node(node, value=rng.randint(-900, 900) / 100.0)
    # Earlier nodes require later nodes, giving a DAG with realistic precedence.
    for u in range(1, node_count + 1):
        for v in range(u + 1, node_count + 1):
            if rng.random() < 0.24:
                graph.add_edge(u, v)
    graph.graph["verification_case"] = case_id
    return graph


def run(cases: int = 100, seed: int = 3603) -> dict:
    rng = random.Random(seed)
    closure_checks = 0
    refinement_checks = 0
    max_closure_error = 0.0

    for case_id in range(cases):
        graph = _random_dag(rng, rng.randint(4, 10), case_id)
        # Check HPF against exhaustive enumeration, including compatible hard pins.
        _, feasible_seed = _enumerate_optimum(graph)
        required = {node for node in feasible_seed if rng.random() < 0.12}
        forbidden = {node for node in set(graph) - feasible_seed if rng.random() < 0.12}
        exact_value, _ = _enumerate_optimum(graph, required, forbidden)
        solved = _solve_hpf(graph, nx.get_node_attributes(graph, "value"), required, forbidden)
        solved_value = sum(graph.nodes[node]["value"] for node in solved)
        if not _is_feasible(graph, solved, required, forbidden):
            raise AssertionError(f"HPF returned an infeasible closure in case {case_id}")
        error = abs(solved_value - exact_value)
        max_closure_error = max(max_closure_error, error)
        if error > 0.011:
            raise AssertionError(
                f"HPF disagreed with exhaustive optimum in case {case_id}: "
                f"{solved_value} vs {exact_value}"
            )
        closure_checks += 1

        labels = {node: rng.randrange(1, min(4, len(graph)) + 1) for node in graph}
        arbitrary_seed = {node for node in graph if rng.random() < 0.5}
        repaired, _ = repair_with_metrics(graph, arbitrary_seed, mode="add")
        refined, _ = _refine_boundary(graph, labels, arbitrary_seed, repair_mode="add")
        if check_constraints(graph, refined):
            raise AssertionError(f"boundary refinement left violations in case {case_id}")
        repaired_value = sum(graph.nodes[node]["value"] for node in repaired)
        refined_value = sum(graph.nodes[node]["value"] for node in refined)
        if refined_value + 1e-9 < repaired_value:
            raise AssertionError(
                f"boundary refinement lost incumbent NPV in case {case_id}: "
                f"{refined_value} < {repaired_value}"
            )
        refinement_checks += 1

    return {
        "seed": seed,
        "cases": cases,
        "exhaustive_hpf_closure_checks": closure_checks,
        "boundary_feasibility_and_nonregression_checks": refinement_checks,
        "maximum_hpf_objective_error": max_closure_error,
        "value_rounding_tolerance": 0.011,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3603)
    parser.add_argument("--output", type=Path,
                        default=Path("results/boundary_refinement_verification.json"))
    args = parser.parse_args()
    result = run(args.cases, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
