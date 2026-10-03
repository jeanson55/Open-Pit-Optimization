"""Maximum-weight closure via minimum s-t cut.

This is the core engine behind both the HPF local solves and LG-style
boundary repair.  If your existing HPF–LG implementation already provides a
closure solver, swap it in here — the rest of the pipeline only calls
`maximum_closure`.
"""

from typing import Dict, Iterable, Set

import networkx as nx

INF = 1e12          # "infinite" arc capacity (must exceed sum of |values|)
SOURCE, SINK = "__source__", "__sink__"


def maximum_closure(graph: nx.DiGraph,
                    values: Dict[int, float],
                    required: Iterable[int] = ()) -> Set[int]:
    """Maximise sum of node values subject to precedence closure.

    Parameters
    ----------
    graph  : precedence digraph (u -> v means u must be taken before v)
    values : net value of each node (negative = waste / cost)
    required : nodes forced into the closure (their outgoing requirements
               are pulled in automatically)

    Returns the optimal closed node set.
    """
    H = nx.DiGraph()
    H.add_nodes_from([SOURCE, SINK])
    H.add_nodes_from(values)
    for n, val in values.items():
        if val >= 0:
            H.add_edge(SOURCE, n, capacity=val)
        else:
            H.add_edge(n, SINK, capacity=-val)
    for u, v in graph.edges:
        if u in values and v in values:
            H.add_edge(u, v, capacity=INF)

    _, (reachable, _) = nx.minimum_cut(H, SOURCE, SINK)
    closure = (set(reachable) - {SOURCE}) | set(required)
    # Required nodes may drag in their successor requirements outside the cut.
    missing = {v for u, v in graph.edges if u in closure and v not in closure}
    while missing:
        closure |= missing
        missing = {v for u, v in graph.edges if u in closure and v not in closure}
    return closure

