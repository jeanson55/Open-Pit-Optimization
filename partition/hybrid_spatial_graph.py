"""Spatially seeded graph-boundary refinement partitioner (Hybrid V4).

The spatial grid provides local, contiguous starting regions. The refinement
then considers only blocks on a graph cut and moves a block only to a part
used by one of its precedence neighbors. Moves strictly reduce the directed
precedence cut, or preserve it while improving size balance, subject to a
hard maximum part size. This is a deterministic, dependency-free alternative
to selecting one of two complete candidate labelings.
"""
from __future__ import annotations

from collections import Counter
from typing import Mapping

def _cut_count(graph, labels: Mapping[int, int]) -> int:
    return sum(1 for u, v in graph.edges if labels[u] != labels[v])


def _boundary_nodes(graph, labels: Mapping[int, int]) -> set:
    boundary = set()
    for u, v in graph.edges:
        if labels[u] != labels[v]:
            boundary.add(u)
            boundary.add(v)
    return boundary


def _size_penalty(sizes: Mapping[int, int], mean: float) -> float:
    if not sizes or mean <= 0:
        return 0.0
    return sum((size - mean) ** 2 for size in sizes.values()) / (len(sizes) * mean**2)


def _move_cut_delta(graph, node, old_part, new_part, labels) -> int:
    """Exact directed cut-arc change from moving one node."""
    delta = 0
    for neighbor in graph.successors(node):
        delta += int(new_part != labels[neighbor]) - int(old_part != labels[neighbor])
    for neighbor in graph.predecessors(node):
        delta += int(new_part != labels[neighbor]) - int(old_part != labels[neighbor])
    return delta


def _swap_cut_delta(graph, first, second, labels) -> int:
    """Exact cut change from exchanging two boundary blocks' labels."""
    first_part = labels[first]
    second_part = labels[second]
    if first_part == second_part:
        return 0

    affected_arcs = set()
    affected_arcs.update((first, v) for v in graph.successors(first))
    affected_arcs.update((u, first) for u in graph.predecessors(first))
    affected_arcs.update((second, v) for v in graph.successors(second))
    affected_arcs.update((u, second) for u in graph.predecessors(second))

    def after(node):
        if node == first:
            return second_part
        if node == second:
            return first_part
        return labels[node]

    return sum(
        int(after(u) != after(v)) - int(labels[u] != labels[v])
        for u, v in affected_arcs
    )


def refine_spatial_labels(
    graph,
    initial_labels: Mapping[int, int],
    *,
    balance_tolerance: float = 1.15,
    max_sweeps: int = 10,
) -> tuple[dict[int, int], dict]:
    """Refine spatial labels using local graph affinity and a size cap."""
    if balance_tolerance < 1.0:
        raise ValueError("balance_tolerance must be at least 1.0")
    if max_sweeps < 0:
        raise ValueError("max_sweeps must be nonnegative")
    if set(initial_labels) != set(graph.nodes):
        raise ValueError("initial labels must cover every graph node exactly once")

    labels = dict(initial_labels)
    sizes = Counter(labels.values())
    if len(sizes) < 2:
        return labels, {
            "initial_cut_arcs": _cut_count(graph, labels),
            "final_cut_arcs": _cut_count(graph, labels),
            "cut_arcs_removed": 0,
            "moved_blocks": 0,
            "sweeps": 0,
            "initial_size_balance_max_over_mean": 1.0,
            "final_size_balance_max_over_mean": 1.0,
            "balance_tolerance": balance_tolerance,
        }

    part_count = len(sizes)
    node_count = len(labels)
    mean_size = node_count / part_count
    max_size = balance_tolerance * mean_size
    initial_cut = _cut_count(graph, labels)
    current_size_penalty = _size_penalty(sizes, mean_size)
    initial_max_ratio = max(sizes.values()) / mean_size
    total_moves = 0
    total_swaps = 0
    completed_sweeps = 0

    for sweep in range(max_sweeps):
        boundary = sorted(_boundary_nodes(graph, labels))
        sweep_moves = 0

        for node in boundary:
            old_part = labels[node]
            if sizes[old_part] <= 1:
                continue

            candidate_parts = {
                labels[neighbor]
                for neighbor in graph.successors(node)
            }
            candidate_parts.update(
                labels[neighbor]
                for neighbor in graph.predecessors(node)
            )
            candidate_parts.discard(old_part)
            if not candidate_parts:
                continue

            best = None
            old_size = sizes[old_part]
            old_term = (old_size - mean_size) ** 2
            for new_part in sorted(candidate_parts):
                new_size = sizes[new_part]
                if new_size + 1 > max_size:
                    continue

                cut_delta = _move_cut_delta(
                    graph, node, old_part, new_part, labels
                )
                size_delta = (
                    (old_size - 1 - mean_size) ** 2
                    + (new_size + 1 - mean_size) ** 2
                    - old_term
                    - (new_size - mean_size) ** 2
                ) / (part_count * mean_size**2)

                # Lexicographic objective: first reduce graph cut; when the
                # cut is unchanged, accept only a strict balance improvement.
                if cut_delta < 0 or (cut_delta == 0 and size_delta < -1e-12):
                    candidate = (cut_delta, size_delta, new_part)
                    if best is None or candidate < best:
                        best = candidate

            if best is None:
                continue

            cut_delta, size_delta, new_part = best
            labels[node] = new_part
            sizes[old_part] -= 1
            sizes[new_part] += 1
            current_size_penalty += size_delta
            total_moves += 1
            sweep_moves += 1

        # Single-node refinement can be trapped where either side of a cut
        # has no improving one-block move. Equal-size swaps cross that local
        # minimum while preserving every part's block count exactly.
        node_order = {node: index for index, node in enumerate(graph.nodes)}
        cut_pairs = set()
        for u, v in graph.edges:
            if labels[u] != labels[v]:
                cut_pairs.add((u, v) if node_order[u] < node_order[v] else (v, u))
        for first, second in sorted(
            cut_pairs,
            key=lambda pair: (node_order[pair[0]], node_order[pair[1]]),
        ):
            if labels[first] == labels[second]:
                continue
            if _swap_cut_delta(graph, first, second, labels) < 0:
                labels[first], labels[second] = labels[second], labels[first]
                total_swaps += 1
                sweep_moves += 1

        completed_sweeps = sweep + 1
        if sweep_moves == 0:
            break

    final_cut = _cut_count(graph, labels)
    # Recompute this diagnostic to avoid exposing accumulated float drift.
    current_size_penalty = _size_penalty(sizes, mean_size)
    _ = current_size_penalty
    diagnostics = {
        "initial_cut_arcs": initial_cut,
        "final_cut_arcs": final_cut,
        "cut_arcs_removed": initial_cut - final_cut,
        "moved_blocks": total_moves,
        "boundary_pair_swaps": total_swaps,
        "sweeps": completed_sweeps,
        "initial_size_balance_max_over_mean": initial_max_ratio,
        "final_size_balance_max_over_mean": max(sizes.values()) / mean_size,
        "balance_tolerance": balance_tolerance,
    }
    if final_cut > initial_cut:
        raise AssertionError("hybrid graph refinement increased the precedence cut")
    return labels, diagnostics


def hybrid_spatial_graph_partition(
    graph,
    n_partitions: int = 9,
    *,
    balance_tolerance: float = 1.15,
    max_sweeps: int = 10,
) -> dict[int, int]:
    """Return spatial labels refined by local precedence-graph connectivity."""
    from .spatial import spatial_partition

    labels = spatial_partition(graph, n_partitions)
    refined, diagnostics = refine_spatial_labels(
        graph,
        labels,
        balance_tolerance=balance_tolerance,
        max_sweeps=max_sweeps,
    )
    diagnostics["algorithm"] = "spatial_seed_graph_cut_local_moves_and_swaps_v4_1"
    graph.graph["hybrid_spatial_graph_diagnostics"] = diagnostics
    print(
        "  Hybrid V4.1 spatial-graph refinement: "
        f"cut {diagnostics['initial_cut_arcs']:,} -> "
        f"{diagnostics['final_cut_arcs']:,}; "
        f"moved={diagnostics['moved_blocks']:,}; "
        f"balance(max/mean) "
        f"{diagnostics['initial_size_balance_max_over_mean']:.3f} -> "
        f"{diagnostics['final_size_balance_max_over_mean']:.3f}"
    )
    return refined
