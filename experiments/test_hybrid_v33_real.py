import time
import numpy as np
import pandas as pd
import networkx as nx

from gpit import precedence as prec
from gpit.pseudoflow import Pseudoflow

from partition.hybrid_v33 import hybrid_v33_partition


DATA = "../lg/bauxitemed.dat"

NX = 120
NY = 120
NZ = 26

K = 8


def build_graph():

    print("Loading data...")

    flat = np.loadtxt(
        DATA,
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

        "xorg": 0,
        "yorg": 0,
        "zorg": 0,

        "xs": 1,
        "ys": 1,
        "zs": 1,

        "volume": 1,
    }


    G = nx.DiGraph()


    print("Creating gPit precedence graph...")


    ext = Pseudoflow(
        df,
        bmparms
    )


    ext.EPV = np.asarray(
        df["EPV"]
    )


    G = ext.CreateExtArcs(G)


    for ind_z in range(NZ - 1):

        pos_z = NZ - ind_z - 2


        for pos_y in range(
            ind_z + 1,
            NY - ind_z - 1
        ):

            for pos_x in range(
                ind_z + 1,
                NX - ind_z - 1
            ):

                G = prec.IA1x5(
                    pos_x,
                    pos_y,
                    pos_z,
                    NX,
                    NY,
                    G
                )


    print(
        "Raw graph:",
        G.number_of_nodes(),
        "nodes",
        G.number_of_edges(),
        "edges"
    )


    # --------------------------------------------------
    # Convert gPit nodes to partition-compatible nodes
    # --------------------------------------------------

    sink = NX * NY * NZ + 1


    remove = []


    for node in list(G.nodes):


        # remove source and sink
        if node == 0 or node == sink:
            remove.append(node)
            continue



        flat_index = node - 1



        # remove artificial nodes
        if (
            flat_index < 0
            or flat_index >= len(flat)
        ):
            remove.append(node)
            continue



        z = flat_index // (NX * NY)

        rem = flat_index % (NX * NY)

        y = rem // NX

        x = rem % NX



        G.nodes[node]["x"] = x
        G.nodes[node]["y"] = y
        G.nodes[node]["z"] = z

        G.nodes[node]["value"] = float(
            flat[flat_index]
        )



    G.remove_nodes_from(remove)



    print(
        "Partition graph:",
        G.number_of_nodes(),
        "nodes",
        G.number_of_edges(),
        "edges"
    )



    # verify

    missing = []


    for n,d in G.nodes(data=True):

        for key in (
            "x",
            "y",
            "z"
        ):

            if key not in d:
                missing.append(n)
                break



    print(
        "Missing coordinates:",
        len(missing)
    )


    if missing:
        raise RuntimeError(
            "Some nodes have no coordinates"
        )


    return G



def main():


    print("="*80)
    print(
        "HYBRID V3.3 REAL GRAPH TEST"
    )
    print("="*80)


    t0=time.perf_counter()


    G = build_graph()


    print(
        "Build time:",
        round(
            time.perf_counter()-t0,
            2
        ),
        "s"
    )



    print()
    print(
        "Running Hybrid V3.3..."
    )


    t=time.perf_counter()


    labels = hybrid_v33_partition(
        G,
        K
    )


    runtime = time.perf_counter()-t



    print()
    print("="*80)
    print("RESULT")
    print("="*80)



    print(
        "Partitions:",
        len(set(labels.values()))
    )


    print(
        "Runtime:",
        round(runtime,3),
        "s"
    )


    sizes={}


    for p in labels.values():

        sizes[p]=sizes.get(p,0)+1



    print(
        "Partition sizes:"
    )


    for p,s in sorted(
        sizes.items()
    ):

        print(
            p,
            ":",
            s
        )



if __name__=="__main__":

    main()