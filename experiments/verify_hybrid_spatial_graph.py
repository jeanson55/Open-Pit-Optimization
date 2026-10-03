"""Dependency-light correctness checks for Hybrid V4 graph refinement."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partition.hybrid_spatial_graph import (
    _cut_count,
    _move_cut_delta,
    _swap_cut_delta,
    refine_spatial_labels,
)


class TinyDiGraph:
    """Minimal directed-graph interface used by the refinement routine."""

    def __init__(self, nodes, edges):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self._successors = defaultdict(list)
        self._predecessors = defaultdict(list)
        for source, target in self.edges:
            self._successors[source].append(target)
            self._predecessors[target].append(source)

    def successors(self, node):
        return iter(self._successors[node])

    def predecessors(self, node):
        return iter(self._predecessors[node])


def run(cases: int = 100, seed: int = 7341) -> dict:
    rng = random.Random(seed)

    # A hand-built case where moving a boundary node removes a net cut arc.
    graph = TinyDiGraph(
        range(1, 7),
        [(2, 4), (3, 4), (3, 5), (1, 2), (5, 6)],
    )
    labels = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
    initial_cut = _cut_count(graph, labels)
    refined, diagnostics = refine_spatial_labels(
        graph, labels, balance_tolerance=1.5, max_sweeps=5
    )
    if _cut_count(graph, refined) >= initial_cut:
        raise AssertionError("known graph-affinity move did not reduce the cut")
    if diagnostics["final_cut_arcs"] > diagnostics["initial_cut_arcs"]:
        raise AssertionError("reported hybrid cut increased")

    # Single-node moves are blocked by the exact-size cap, but exchanging
    # these two boundary nodes lowers the cut while preserving part sizes.
    swap_graph = TinyDiGraph(
        range(1, 7),
        [(3, 4), (3, 5), (3, 6), (1, 3), (4, 1), (4, 2), (6, 4)],
    )
    swap_labels = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1}
    swap_before = _cut_count(swap_graph, swap_labels)
    if _swap_cut_delta(swap_graph, 3, 4, swap_labels) >= 0:
        raise AssertionError("known improving boundary swap was not detected")
    swapped, swap_metrics = refine_spatial_labels(
        swap_graph, swap_labels, balance_tolerance=1.0, max_sweeps=4
    )
    if _cut_count(swap_graph, swapped) >= swap_before:
        raise AssertionError("pair-swap refinement failed to escape single-move trap")
    if swap_metrics["boundary_pair_swaps"] < 1:
        raise AssertionError("expected the known graph case to use a pair swap")

    delta_checks = 0
    final_cut_checks = 0
    deterministic_checks = 0
    swap_delta_checks = 0
    for _ in range(cases):
        n = rng.randint(8, 32)
        nodes = list(range(n))
        edges = [
            (u, v)
            for u in nodes
            for v in range(u + 1, n)
            if rng.random() < 0.12
        ]
        graph = TinyDiGraph(nodes, edges)
        k = rng.randint(2, min(5, n))
        labels = {node: node % k for node in nodes}
        # Verify local move deltas against brute-force full cut recomputation.
        for node in nodes:
            old = labels[node]
            for new in range(k):
                if new == old:
                    continue
                predicted = _move_cut_delta(graph, node, old, new, labels)
                trial = dict(labels)
                trial[node] = new
                actual = _cut_count(graph, trial) - _cut_count(graph, labels)
                if predicted != actual:
                    raise AssertionError(
                        f"cut delta mismatch: predicted {predicted}, actual {actual}"
                    )
                delta_checks += 1

        cross_part_pairs = [
            (nodes[i], nodes[j])
            for i in range(n)
            for j in range(i + 1, n)
            if labels[nodes[i]] != labels[nodes[j]]
        ]
        sample_pairs = rng.sample(
            cross_part_pairs, min(50, len(cross_part_pairs))
        )
        for first, second in sample_pairs:
            predicted = _swap_cut_delta(graph, first, second, labels)
            trial = dict(labels)
            trial[first], trial[second] = trial[second], trial[first]
            actual = _cut_count(graph, trial) - _cut_count(graph, labels)
            if predicted != actual:
                raise AssertionError(
                    f"swap delta mismatch: predicted {predicted}, actual {actual}"
                )
            swap_delta_checks += 1

        before = _cut_count(graph, labels)
        result_a, metrics_a = refine_spatial_labels(
            graph, labels, balance_tolerance=1.5, max_sweeps=6
        )
        result_b, metrics_b = refine_spatial_labels(
            graph, labels, balance_tolerance=1.5, max_sweeps=6
        )
        after = _cut_count(graph, result_a)
        if after > before:
            raise AssertionError("refinement increased the directed cut")
        if result_a != result_b or metrics_a != metrics_b:
            raise AssertionError("refinement is not deterministic")
        final_cut_checks += 1
        deterministic_checks += 1

    return {
        "seed": seed,
        "random_graph_cases": cases,
        "move_delta_matches_full_recomputation": delta_checks,
        "swap_delta_matches_full_recomputation": swap_delta_checks,
        "cases_with_nonincreasing_cut": final_cut_checks,
        "deterministic_cases": deterministic_checks,
        "known_cut_reduction": diagnostics["cut_arcs_removed"],
        "known_pair_swap_cut_reduction": swap_before - _cut_count(swap_graph, swapped),
        "known_pair_swaps_applied": swap_metrics["boundary_pair_swaps"],
        "status": "passed",
    }


def main():
    result = run()
    print(json.dumps(result, indent=2))
    output = Path(__file__).resolve().parents[1] / "results" / "hybrid_spatial_graph_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
