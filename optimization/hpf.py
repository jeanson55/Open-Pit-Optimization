"""HPF (hierarchical partition-first) optimizer.

Pipeline per method call:
    partition labels
        -> per-partition local closure optimization
        -> merged candidate pit (boundary issues handled by lg_repair)

Example:
    solution = run_hpf(graph, partition_labels)

NOTE
----
`run_hpf` is deliberately a thin orchestrator.  Drop your existing HPF
implementation into this function (or have it call your solver module);
every experiment in `experiments/` only depends on this signature.
"""

from typing import Dict, Iterable, Set

import networkx as nx

from .closure import maximum_closure


def run_hpf(graph: nx.DiGraph, partition_labels: Dict[int, int],
            local_solver=None) -> Set[int]:
    """Solve each partition locally and merge the results.

    local_solver : callable(subgraph, values) -> node set.
                   Defaults to the min-cut closure solver in closure.py.
    """
    if local_solver is None:
        local_solver = lambda sub, vals: maximum_closure(sub, vals)

    solution: Set[int] = set()
    for part in sorted(set(partition_labels.values())):
        members = [n for n in graph.nodes if partition_labels.get(n) == part]
        if not members:
            continue
        sub = graph.subgraph(members)
        values = {n: graph.nodes[n]["value"] for n in members}
        solution |= set(local_solver(sub, values))
    return solution
