"""Spectral clustering partition (Step 6).

Embeds the (symmetrised) precedence graph via the normalised Laplacian and
clusters the eigenvectors with k-means.

Interface:
    labels = spectral_partition(graph, n_partitions=9)
"""

import warnings
from typing import Dict

import networkx as nx


def spectral_partition(graph: nx.DiGraph, n_partitions: int = 9) -> Dict[int, int]:
    try:
        import numpy as np
        from scipy.sparse import csgraph
        from scipy.sparse.linalg import eigsh, lobpcg
        from sklearn.cluster import KMeans
    except ImportError:
        warnings.warn(
            "spectral partitioning needs numpy/scipy/scikit-learn; "
            "falling back to spatial_partition"
        )
        from .spatial import spatial_partition
        return spatial_partition(graph, n_partitions)

    nodes = sorted(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    import scipy.sparse as sp
    rows, cols = [], []
    for u, v in graph.edges:
        rows += [index[u], index[v]]
        cols += [index[v], index[u]]
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))

    lap = csgraph.laplacian(adj, normed=True)
    k = min(n_partitions, n - 2)
    if n > 50_000:
        # Shift-invert factorization of the singular graph Laplacian has high
        # fill-in on large block models. LOBPCG finds the same low-frequency
        # eigenspace with sparse matrix products and bounded memory.
        rng = np.random.default_rng(0)
        initial = rng.standard_normal((n, k))
        _, vecs = lobpcg(lap, initial, largest=False, tol=1e-3, maxiter=150)
    else:
        _, vecs = eigsh(lap, k=k, which="SM", tol=1e-6)
    km = KMeans(n_clusters=n_partitions, n_init=10, random_state=0).fit(vecs)
    centers = km.cluster_centers_.copy()

    # Enforce balanced capacities while preserving each node's preference for
    # its nearest spectral centroid. Assign high-confidence nodes first; nodes
    # whose two best choices are close fill remaining slots.
    target_sizes = np.full(n_partitions, n // n_partitions, dtype=int)
    target_sizes[:n % n_partitions] += 1
    labels = np.full(n, -1, dtype=int)
    for _ in range(4):
        distances = ((vecs[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        choices = np.argsort(distances, axis=1)
        ordered_distances = np.take_along_axis(distances, choices, axis=1)
        confidence = ordered_distances[:, 1] - ordered_distances[:, 0]
        node_order = np.argsort(-confidence, kind="stable")
        remaining = target_sizes.copy()
        labels.fill(-1)
        for row in node_order:
            for part in choices[row]:
                if remaining[part]:
                    labels[row] = part
                    remaining[part] -= 1
                    break
        centers = np.vstack([vecs[labels == part].mean(axis=0)
                             for part in range(n_partitions)])
    return {node: int(labels[index[node]]) for node in nodes}
