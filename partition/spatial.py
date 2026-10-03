"""Spatial grid partitioning (current baseline).

Divides the deposit into a regular 2D (x, y) grid of cells and assigns every
block to the cell that contains it.  Balance is approximate and edge cuts
follow grid lines — the weaknesses this research aims to beat.

Interface (identical for every partition method):
    labels = spatial_partition(graph, n_partitions=9)
    # -> {block_id: partition_id, ...}
"""

import math
from typing import Dict

import networkx as nx
import pandas as pd


def _blocks_dataframe(graph_or_blocks) -> pd.DataFrame:
    if isinstance(graph_or_blocks, nx.Graph):
        rows = [(n, d["x"], d["y"], d["z"]) for n, d in graph_or_blocks.nodes(data=True)]
        return pd.DataFrame(rows, columns=["block_id", "x", "y", "z"])
    return graph_or_blocks


def spatial_partition(graph_or_blocks, n_partitions: int = 9) -> Dict[int, int]:
    blocks = _blocks_dataframe(graph_or_blocks)

    cols = math.ceil(math.sqrt(n_partitions))
    rows = math.ceil(n_partitions / cols)

    xmin, xmax = blocks["x"].min(), blocks["x"].max()
    ymin, ymax = blocks["y"].min(), blocks["y"].max()
    xstep = (xmax - xmin) / cols if xmax > xmin else 1.0
    ystep = (ymax - ymin) / rows if ymax > ymin else 1.0

    labels = {}
    for b in blocks.itertuples(index=False):
        cx = min(int((b.x - xmin) / xstep), cols - 1) if xstep else 0
        cy = min(int((b.y - ymin) / ystep), rows - 1) if ystep else 0
        part = cy * cols + cx
        labels[int(b.block_id)] = min(part, n_partitions - 1)
    return labels
