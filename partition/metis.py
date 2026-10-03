"""METIS partitioning (Step 5).

Requires the optional `pymetis` package.  Falls back to the spatial baseline
with a warning when it is not installed, so experiment scripts keep running.

Interface:
    labels = metis_partition(graph, n_partitions=9)
"""

import warnings
from typing import Dict

import networkx as nx


def metis_partition(graph: nx.DiGraph, n_partitions: int = 9) -> Dict[int, int]:
    try:
        import pymetis  # noqa: F401
    except ImportError:
        warnings.warn(
            "pymetis not installed; falling back to spatial_partition. "
            "Install with: pip install pymetis"
        )
        from .spatial import spatial_partition
        return spatial_partition(graph, n_partitions)

    import pymetis

    nodes = sorted(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    adjacency = []
    for n in nodes:                      # METIS needs an *undirected* adjacency
        nbrs = {index[m] for m in graph.successors(n)} | {index[m] for m in graph.predecessors(n)}
        adjacency.append(sorted(nbrs))

    _, parts = pymetis.part_graph(n_partitions, adjacency=adjacency)
    return {n: int(parts[index[n]]) for n in nodes}
