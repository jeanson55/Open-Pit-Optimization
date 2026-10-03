"""Proposed hybrid partition method (Step 7).

Working baseline provided so the comparison runs end-to-end:
  Phase 1 — spatial grid seeding (gives balanced, spatially coherent parts)
  Phase 2 — greedy boundary refinement that moves boundary blocks to the
            neighbouring partition when doing so reduces the edge cut
            without breaking the balance tolerance

Replace / extend Phase 2 with the proposed method as the research develops.

Interface:
    labels = hybrid_partition(graph, n_partitions=9)
"""

from collections import Counter, defaultdict
from typing import Dict

import networkx as nx

from .spatial import spatial_partition

# move a block only if the receiving part stays below this x the mean size
BALANCE_TOLERANCE = 1.10
MAX_SWEEPS = 25


def _undirected(graph: nx.DiGraph) -> nx.Graph:
    U = nx.Graph()
    U.add_nodes_from(graph.nodes)
    U.add_edges_from((u, v) for u, v in graph.edges)
    U.add_edges_from((v, u) for u, v in graph.edges)
    return U


def hybrid_partition(graph: nx.DiGraph, n_partitions: int = 9,
                     balance_tolerance: float = BALANCE_TOLERANCE,
                     max_sweeps: int = MAX_SWEEPS) -> Dict[int, int]:
    labels = spatial_partition(graph, n_partitions)
    U = _undirected(graph)
    nodes = len(labels)
    mean = nodes / n_partitions

    for _ in range(max_sweeps):
        sizes = Counter(labels.values())
        moved = False
        for u in list(U.nodes):
            own = labels[u]
            # candidate parts = parts of this block's neighbours
            cand = {labels[v] for v in U.neighbors(u) if labels[v] != own}
            if not cand:
                continue
            # gain of moving u to part p: cut edges removed - cut edges added
            for p in cand:
                if sizes[p] + 1 > mean * balance_tolerance:
                    continue
                cut_now = sum(1 for v in U.neighbors(u) if labels[v] != own)
                cut_new = sum(1 for v in U.neighbors(u)
                              if (labels[v] != p and labels[v] != own))
                # neighbours already in p stop being cut; neighbours in own
                # become cut.  net gain = (#nbrs in p) - (#nbrs in own)
                gain = sum(1 for v in U.neighbors(u) if labels[v] == p) \
                     - sum(1 for v in U.neighbors(u) if labels[v] == own)
                _ = (cut_now, cut_new)  # kept for clarity
                if gain > 0:
                    labels[u] = p
                    sizes[own] -= 1
                    sizes[p] += 1
                    moved = True
                    break
        if not moved:
            break
    return labels
