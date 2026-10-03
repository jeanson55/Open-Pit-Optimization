"""Precedence repair with optional block-level cascade tracing.

The default API remains compatible with the original implementation:
    repaired, metrics = repair_with_metrics(...)

When ``return_trace=True`` the function also returns the actual blocks added
or removed at each dependency wave.
"""

from typing import Iterable, Set, Tuple

import networkx as nx


def repair_with_metrics(
    graph: nx.DiGraph,
    solution: Iterable,
    mode: str = "add",
    return_trace: bool = False,
):
    """Repair precedence violations and report cascade metrics.

    For add mode, an arc ``u -> v`` means selecting ``u`` requires ``v``.
    The repair proceeds in dependency waves. ``repair_level_by_node`` records
    the exact wave at which every newly added block entered the repaired set.

    For prune mode, the analogous mapping records removal waves.
    """
    sol = set(solution)

    if mode == "add":
        added: Set[int] = set()
        added_by_level = {}
        level_by_node = {}
        frontier = {
            v
            for u, v in graph.edges
            if u in sol and v not in sol
        }
        depth = 0

        while frontier:
            frontier.difference_update(sol)
            frontier.difference_update(added)

            if not frontier:
                break

            depth += 1
            wave = set(frontier)
            added.update(wave)
            added_by_level[depth] = wave

            for node in wave:
                level_by_node[node] = depth

            frontier = {
                successor
                for node in wave
                for successor in graph.successors(node)
                if (
                    successor not in sol
                    and successor not in added
                )
            }

        repaired = sol | added

        metrics = {
            "cascade_added_blocks": len(added),
            "cascade_removed_blocks": 0,
            "cascade_depth": depth,
        }

        if not return_trace:
            return repaired, metrics

        trace = {
            "repair_mode": "add",
            "repair_original_solution": set(sol),
            "repair_added": set(added),
            "repair_added_by_level": added_by_level,
            "repair_level_by_node": level_by_node,
            "repair_removed": set(),
            "repair_removed_by_level": {},
            "repair_final_solution": set(repaired),
        }
        return repaired, metrics, trace

    if mode == "prune":
        removed: Set[int] = set()
        removed_by_level = {}
        level_by_node = {}
        depth = 0

        while True:
            bad = {
                u
                for u, v in graph.edges
                if u in sol and v not in sol
            }

            if not bad:
                break

            depth += 1
            wave = set(bad)
            sol.difference_update(wave)
            removed.update(wave)
            removed_by_level[depth] = wave

            for node in wave:
                level_by_node[node] = depth

        metrics = {
            "cascade_added_blocks": 0,
            "cascade_removed_blocks": len(removed),
            "cascade_depth": depth,
        }

        if not return_trace:
            return sol, metrics

        trace = {
            "repair_mode": "prune",
            "repair_original_solution": set(sol),
            "repair_added": set(),
            "repair_added_by_level": {},
            "repair_level_by_node": {},
            "repair_removed": set(removed),
            "repair_removed_by_level": removed_by_level,
            "repair_level_by_node": level_by_node,
            "repair_final_solution": set(sol),
        }
        return sol, metrics, trace

    raise ValueError(
        f"unknown repair mode: {mode!r}"
    )


def repair(
    graph: nx.DiGraph,
    solution: Iterable,
    mode: str = "add",
) -> Set[int]:
    """Backward-compatible repair API."""
    repaired, _ = repair_with_metrics(
        graph,
        solution,
        mode=mode,
    )
    return repaired
