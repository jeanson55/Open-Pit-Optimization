import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from gpit import precedence as prec
from gpit.pseudoflow import Pseudoflow

from partition.spatial import spatial_partition
from partition.metis import metis_partition
from partition.hybrid import hybrid_partition


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path("../lg")

MODELS = {
    "bauxitemed": (120, 120, 26),
    "cucase": (70, 70, 30),
    "cupipe": (100, 100, 50),
    "mclaughlingeo": (90, 90, 50),
}

PARTITIONS = 8

OUTPUT = Path(
    "results/hybrid_v32_multimodel_summary.json"
)


# ============================================================
# GRAPH BUILDER
# ============================================================

def build_graph(path, nx_, ny_, nz_):

    flat = np.loadtxt(path, dtype=np.float64)

    df = pd.DataFrame({
        "EPV": flat
    })

    bmparms = {
        "nx": nx_,
        "ny": ny_,
        "nz": nz_,
        "xorg": 0,
        "yorg": 0,
        "zorg": 0,
        "xs": 1,
        "ys": 1,
        "zs": 1,
        "volume": 1,
    }

    G = nx.DiGraph()

    ext = Pseudoflow(df, bmparms)
    ext.EPV = np.asarray(df["EPV"])

    G = ext.CreateExtArcs(G)


    for ind_z in range(nz_ - 1):

        pos_z = nz_ - ind_z - 2

        for pos_y in range(ind_z, ny_ - ind_z):

            for pos_x in range(ind_z, nx_ - ind_z):

                G = prec.IA1x5(
                    pos_x,
                    pos_y,
                    pos_z,
                    nx_,
                    ny_,
                    G,
                )


    sink = nx_ * ny_ * nz_ + 1

    real_nodes = [
        n for n in G.nodes
        if n not in (0, sink)
    ]

    H = G.subgraph(real_nodes).copy()

    # ------------------------------------------------------------
    # Add spatial coordinates required by partition methods
    # ------------------------------------------------------------

    for node in H.nodes:

        idx = int(node) - 1

        x = idx % nx_
        y = (idx // nx_) % ny_
        z = idx // (nx_ * ny_)

        H.nodes[node]["x"] = float(x)
        H.nodes[node]["y"] = float(y)
        H.nodes[node]["z"] = float(z)

        # add economic value
        if 0 <= idx < len(flat):
            H.nodes[node]["value"] = float(flat[idx])
        else:
            H.nodes[node]["value"] = 0.0

    return H


# ============================================================
# METRICS
# ============================================================

def partition_metrics(G, labels):

    sizes = {}

    for node, part in labels.items():

        sizes[part] = sizes.get(part,0)+1


    balance = (
        max(sizes.values())
        /
        min(sizes.values())
    )


    cut = sum(
        1
        for u,v in G.edges
        if labels[u] != labels[v]
    )


    cut_percent = (
        100 * cut / G.number_of_edges()
    )


    return {
        "balance": balance,
        "cut": cut,
        "cut_percent": cut_percent,
        "min_size": min(sizes.values()),
        "max_size": max(sizes.values()),
    }



# ============================================================
# RUN
# ============================================================


def run():

    results = {}


    print("="*100)
    print("HYBRID V3.2 MULTI MODEL BENCHMARK")
    print("="*100)


    for name, dims in MODELS.items():

        nx_, ny_, nz_ = dims

        print("\n")
        print("="*100)
        print(
            f"{name.upper()} "
            f"({nx_} x {ny_} x {nz_})"
        )
        print("="*100)


        data = BASE / f"{name}.dat"


        print("\nBuilding graph...")

        start = time.perf_counter()

        G = build_graph(
            data,
            nx_,
            ny_,
            nz_,
        )

        build_time = (
            time.perf_counter()
            -
            start
        )


        print(
            f"Nodes : {G.number_of_nodes():,}"
        )

        print(
            f"Edges : {G.number_of_edges():,}"
        )

        print(
            f"Build time : {build_time:.2f}s"
        )


        model_result = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "build_time": build_time,
        }


        # -----------------------------
        # Spatial
        # -----------------------------

        print("\nSpatial")

        t=time.perf_counter()

        spatial = spatial_partition(
            G,
            PARTITIONS
        )

        spatial_time=time.perf_counter()-t


        sm = partition_metrics(
            G,
            spatial
        )


        print(sm)


        model_result["spatial"]={
            "runtime":spatial_time,
            **sm
        }



        # -----------------------------
        # METIS
        # -----------------------------

        print("\nMETIS")

        t=time.perf_counter()

        metis = metis_partition(
            G,
            PARTITIONS
        )

        metis_time=time.perf_counter()-t


        mm=partition_metrics(
            G,
            metis
        )


        print(mm)


        model_result["metis"]={
            "runtime":metis_time,
            **mm
        }



        # -----------------------------
        # Hybrid V3.2
        # -----------------------------

        print("\nHybrid V3.2")

        t=time.perf_counter()

        hybrid = hybrid_partition(
            G,
            PARTITIONS
        )

        hybrid_time=time.perf_counter()-t


        hm=partition_metrics(
            G,
            hybrid
        )


        print(hm)


        same_spatial = (
            hybrid == spatial
        )

        same_metis = (
            hybrid == metis
        )


        if same_spatial:
            selected="Spatial"

        elif same_metis:
            selected="METIS"

        else:
            selected="OTHER"


        print(
            "Selected:",
            selected
        )


        model_result["hybrid"]={
            "runtime":hybrid_time,
            "selected":selected,
            **hm
        }


        results[name]=model_result



    OUTPUT.parent.mkdir(
        exist_ok=True
    )

    OUTPUT.write_text(
        json.dumps(
            results,
            indent=2
        )
    )


    print("\n")
    print("="*100)
    print("FINAL SUMMARY")
    print("="*100)


    for name,r in results.items():

        print(
            f"{name:18s}"
            f" Hybrid={r['hybrid']['selected']:7s}"
            f" "
            f"Spatial cut={r['spatial']['cut']:8d}"
            f" "
            f"METIS cut={r['metis']['cut']:8d}"
        )


    print("\nSaved:")
    print(OUTPUT)



if __name__ == "__main__":
    run()