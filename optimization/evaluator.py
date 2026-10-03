"""Economic evaluation of a candidate pit.

Calculates NPV, tonnage, ore recovery and runtime of a solution.
"""

import time
from typing import Dict, Iterable, Optional

import networkx as nx


def evaluate(graph: nx.DiGraph, solution: Iterable,
             block_tonnage: float = 2080.0,
             runtime: Optional[float] = None) -> Dict[str, float]:
    """Evaluate a (closed) pit solution."""
    sol = set(solution)
    values = nx.get_node_attributes(graph, "value")
    grades = nx.get_node_attributes(graph, "grade")

    npv = sum(values[n] for n in sol)
    ore_t = sum(block_tonnage * grades[n] for n in sol)     # in-situ product
    total_t = block_tonnage * len(sol)

    result = {
        "npv": npv,
        "tonnage": total_t,
        "ore_recovery": ore_t,
        "avg_grade": ore_t / total_t if total_t else 0.0,
        "blocks_mined": len(sol),
        "blocks_total": graph.number_of_nodes(),
    }
    if runtime is not None:
        result["runtime"] = runtime
    return result


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0
        return False
