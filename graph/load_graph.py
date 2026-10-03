"""Load an open-pit block model and construct its precedence graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from gpit import precedence as prec
from gpit.pseudoflow import Pseudoflow


def load_graph(
    dat_file,
    nx_dim=120,
    ny_dim=120,
    nz_dim=26,
    precedence_template="1x5",
):
    """Load EPV values and construct the 1x5 slope-precedence graph."""
    flat = np.loadtxt(
        dat_file,
        dtype=np.float64,
    )

    df = pd.DataFrame({"EPV": flat})

    expected = nx_dim * ny_dim * nz_dim
    if len(df) != expected:
        raise ValueError(
            f"Expected {expected:,} block values for "
            f"{nx_dim}x{ny_dim}x{nz_dim}, "
            f"but found {len(df):,}."
        )

    if precedence_template != "1x5":
        raise ValueError(
            f"Unsupported precedence template: "
            f"{precedence_template!r}"
        )

    bmparms = {
        "nx": nx_dim,
        "ny": ny_dim,
        "nz": nz_dim,
        "xorg": 0,
        "yorg": 0,
        "zorg": 0,
        "xs": 1,
        "ys": 1,
        "zs": 1,
        "volume": 1,
    }

    graph = nx.DiGraph()

    ext = Pseudoflow(df, bmparms)
    ext.EPV = np.asarray(df["EPV"])
    graph = ext.CreateExtArcs(graph)

    for ind_z in range(nz_dim - 1):
        pos_z = nz_dim - ind_z - 2

        for pos_y in range(
            ind_z + 1,
            ny_dim - ind_z - 1,
        ):
            for pos_x in range(
                ind_z + 1,
                nx_dim - ind_z - 1,
            ):
                graph = prec.IA1x5(
                    pos_x,
                    pos_y,
                    pos_z,
                    nx_dim,
                    ny_dim,
                    graph,
                )

    print(
        f"Graph loaded: "
        f"{graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )

    return graph, df
