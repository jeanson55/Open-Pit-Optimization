"""
Hybrid V3.3 ML partition selector.

Generates Spatial and METIS candidates,
then uses trained selector model to choose
the expected best NPV partition.
"""

from pathlib import Path
import joblib
import networkx as nx

from .spatial import spatial_partition
from .metis import metis_partition


MODEL_PATH = Path(
    "results/hybrid_v33_selector.pkl"
)


def partition_features(
    graph,
    labels,
    method,
):

    sizes = {}

    for p in labels.values():
        sizes[p] = sizes.get(p,0)+1


    edge_cut = 0

    for u,v in graph.edges():

        if labels[u] != labels[v]:
            edge_cut += 1


    values = [
        abs(float(d.get("value",0)))
        for _,d in graph.nodes(data=True)
    ]


    return {
        "method": method,
        "partition_runtime":0,
        "solver_runtime":0,
        "total_runtime":0,
        "initial_violations":0,
        "edge_cut":edge_cut,
        "balance":
            max(sizes.values()) /
            min(sizes.values()),
    }



def hybrid_v33_partition(
        graph,
        n_partitions=8
):

    print("Hybrid V3.3: generating candidates")


    spatial = spatial_partition(
        graph,
        n_partitions
    )

    metis = metis_partition(
        graph,
        n_partitions
    )


    candidates = [
        ("Spatial", spatial),
        ("METIS", metis),
    ]


    bundle = joblib.load(
        MODEL_PATH
    )


    model = bundle["model"]
    dataset_encoder = bundle["dataset_encoder"]
    method_encoder = bundle["method_encoder"]
    features = bundle["features"]


    # dataset inference
    dataset="unknown"


    results=[]


    for name,labels in candidates:


        row = partition_features(
            graph,
            labels,
            name
        )


        row["dataset_id"] = 0


        row["method_id"] = (
            method_encoder.transform(
                [name]
            )[0]
        )

        import pandas as pd

        X = pd.DataFrame(
            [
                row
            ]
        )[features]

        pred = model.predict(
            X
        )[0]


        results.append(
            (
                name,
                labels,
                pred
            )
        )


        print(
            f"{name}: prediction={pred}"
        )


    # choose predicted winner

    for name,labels,pred in results:

        if pred==1:

            print(
                "Hybrid V3.3 selected:",
                name
            )

            return labels


    # fallback

    print(
        "Hybrid V3.3 fallback METIS"
    )

    return metis