import time
import numpy as np
import pandas as pd
import networkx as nx


from gpit import precedence as prec
from gpit.pseudoflow import Pseudoflow


from partition.hybrid_v34 import hybrid_v34_partition



PATH = "../lg/bauxitemed.dat"

NX = 120
NY = 120
NZ = 26

K = 8



def build_graph():

    flat = np.loadtxt(
        PATH,
        dtype=np.float64
    )


    df = pd.DataFrame(
        {
            "EPV": flat
        }
    )


    bmparms = {

        "nx": NX,
        "ny": NY,
        "nz": NZ,

        "xorg":0,
        "yorg":0,
        "zorg":0,

        "xs":1,
        "ys":1,
        "zs":1,

        "volume":1,

    }


    G = nx.DiGraph()


    ext = Pseudoflow(
        df,
        bmparms
    )


    ext.EPV = np.asarray(
        df["EPV"]
    )


    G = ext.CreateExtArcs(G)


    for ind_z in range(NZ-1):

        pos_z = NZ-ind_z-2


        for y in range(
            ind_z,
            NY-ind_z
        ):

            for x in range(
                ind_z,
                NX-ind_z
            ):

                G = prec.IA1x5(
                    x,
                    y,
                    pos_z,
                    NX,
                    NY,
                    G
                )


    return G



def main():


    print("="*80)
    print("HYBRID V3.4 REAL GRAPH TEST")
    print("="*80)



    t0=time.perf_counter()


    G = build_graph()


    print(
        "Graph:",
        G.number_of_nodes(),
        "nodes",
        G.number_of_edges(),
        "edges"
    )


    print(
        "Build time:",
        round(
            time.perf_counter()-t0,
            2
        ),
        "s"
    )



    # remove source/sink

    sink = NX*NY*NZ+1

    G_real = G.copy()

    if 0 in G_real:
        G_real.remove_node(0)

    if sink in G_real:
        G_real.remove_node(sink)

    # -------------------------------------------------
    # Add block coordinates for Spatial partition
    # -------------------------------------------------

    print("Adding coordinates...")

    for node in G_real.nodes:
        idx = node - 1

        z = idx // (NX * NY)

        rem = idx % (NX * NY)

        y = rem // NX

        x = rem % NX

        G_real.nodes[node]["x"] = x
        G_real.nodes[node]["y"] = y
        G_real.nodes[node]["z"] = z

    missing = sum(
        1
        for _, d in G_real.nodes(data=True)
        if not all(
            k in d
            for k in ("x", "y", "z")
        )
    )

    print(
        "Missing coordinates:",
        missing
    )


    print(
        "\nRunning Hybrid V3.4..."
    )


    labels = hybrid_v34_partition(
        G_real,
        K
    )


    sizes={}


    for p in labels.values():

        sizes[p]=sizes.get(p,0)+1



    print("\nRESULT")
    print("="*80)

    print(
        "Partitions:",
        len(sizes)
    )


    print(
        "Partition sizes:"
    )


    for p,s in sorted(sizes.items()):

        print(
            p,
            ":",
            s
        )



if __name__=="__main__":
    main()