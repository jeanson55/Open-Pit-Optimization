"""
Hybrid V3.3 ML Ranking Selector

Generates Spatial and METIS candidates,
extracts partition characteristics,
uses trained ML model to rank candidates.
"""

from pathlib import Path

import pandas as pd
import joblib
import networkx as nx


from .spatial import spatial_partition
from .metis import metis_partition


MODEL_PATH = Path(
    "results/hybrid_v33_selector.pkl"
)



def _partition_metrics(graph, labels):

    sizes = {}

    for p in labels.values():
        sizes[p] = sizes.get(p, 0) + 1


    cut = 0

    for u,v in graph.edges():

        if labels[u] != labels[v]:
            cut += 1


    balance = (
        max(sizes.values())
        /
        min(sizes.values())
    )


    boundary = set()

    for u,v in graph.edges():

        if labels[u] != labels[v]:
            boundary.add(u)
            boundary.add(v)


    return {
        "edge_cut": cut,
        "balance": balance,
        "boundary": len(boundary),
    }




def hybrid_v33_partition(
        graph,
        n_partitions=8
):

    print(
        "Hybrid V3.3: generating candidates"
    )


    candidates = []


    spatial = spatial_partition(
        graph,
        n_partitions
    )


    candidates.append(
        (
            "Spatial",
            spatial
        )
    )


    metis = metis_partition(
        graph,
        n_partitions
    )


    candidates.append(
        (
            "METIS",
            metis
        )
    )



    bundle = joblib.load(
        MODEL_PATH
    )


    model = bundle["model"]

    features = bundle["features"]



    scored=[]



    print(
        "Hybrid V3.3: ranking candidates"
    )



    for name,labels in candidates:


        metrics = _partition_metrics(
            graph,
            labels
        )


        row = {

            "dataset_id":0,

            "method_id":
                bundle["method_encoder"]
                .transform(
                    [name]
                )[0],


            "total_runtime":
                0,


            "partition_runtime":
                0,


            "solver_runtime":
                0,


            "initial_violations":
                0,


            "edge_cut":
                metrics["edge_cut"],


            "balance":
                metrics["balance"],
        }


        X = pd.DataFrame(
            [row]
        )[features]


        probability = model.predict_proba(
            X
        )


        # probability of best class
        score = probability[:,1][0]



        print(
            f"{name}: score={score:.4f} "
            f"cut={metrics['edge_cut']} "
            f"balance={metrics['balance']:.3f}"
        )


        scored.append(
            (
                score,
                name,
                labels
            )
        )



    scored.sort(
        reverse=True,
        key=lambda x:x[0]
    )


    best_score,best_name,best_labels = scored[0]


    print(
        f"Hybrid V3.3 selected: {best_name} "
        f"(score={best_score:.4f})"
    )


    return best_labels