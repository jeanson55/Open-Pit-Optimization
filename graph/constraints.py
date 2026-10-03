"""Precedence checks for gPit's directed block graph.

An arc ``u -> v`` means that selecting block ``u`` requires block ``v``.
A feasible pit is therefore closed in the direction of the arcs.
"""

from typing import Iterable, Set

import networkx as nx


def check_constraints(G: nx.DiGraph, solution: Iterable) -> list:
    """Return arcs ``u -> v`` where ``u`` is selected but required ``v`` is not."""
    sol = set(solution)
    return [(u, v) for u, v in G.edges if u in sol and v not in sol]


def is_closed(G: nx.DiGraph, solution: Iterable) -> bool:
    return len(check_constraints(G, solution)) == 0


def closure_of(G: nx.DiGraph, seeds: Iterable) -> Set:
    """Return the smallest feasible set containing seeds and their requirements."""
    closure = set(seeds)
    frontier = list(closure)
    while frontier:
        required = {v for u in frontier for v in G.successors(u)} - closure
        if not required:
            break
        closure.update(required)
        frontier = list(required)
    return closure
