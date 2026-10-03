"""
Hybrid V3.4.2 ML Adaptive Partition Selector

Uses Random Forest prediction plus
partition quality metrics to select
between Spatial and METIS.
"""

from pathlib import Path
import time

import pandas as pd
import joblib


from .spatial import spatial_partition
from .metis import metis_partition


MODEL_PATH = Path(
    "results/hybrid_v342_selector.pkl"
)



def partition_metrics(graph, labels):

    sizes = {}

    for p in labels.values():
        sizes[p] = sizes.get(p, 0) + 1


    cut = 0

    for u, v in graph.edges():

        if labels[u] != labels[v]:
            cut += 1


    return {
        "cut": cut,
        "balance": max(sizes.values()) / min(sizes.values()),
        "min_size": min(sizes.values()),
        "max_size": max(sizes.values())
    }



def detect_dataset(graph):

    nodes = graph.number_of_nodes()

    if nodes < 300000:
        return 0

    elif nodes < 700000:
        return 1

    elif nodes < 1100000:
        return 2

    return 3



def hybrid_v34_partition(
        graph,
        n_partitions=8,
        dataset_id=None
):

    print(
        "Hybrid V3.4: generating candidates"
    )


    if dataset_id is None:
        dataset_id = detect_dataset(graph)



    candidates=[]


    start=time.perf_counter()

    spatial = spatial_partition(
        graph,
        n_partitions
    )

    candidates.append(
        (
            "Spatial",
            spatial,
            time.perf_counter()-start
        )
    )


    start=time.perf_counter()

    metis = metis_partition(
        graph,
        n_partitions
    )

    candidates.append(
        (
            "METIS",
            metis,
            time.perf_counter()-start
        )
    )



    bundle = joblib.load(
        MODEL_PATH
    )

    model = bundle["model"]
    features = bundle["features"]



    print(
        "Hybrid V3.4: ML ranking"
    )



    ranked=[]



    for name, labels, runtime in candidates:


        metrics = partition_metrics(
            graph,
            labels
        )


        method_id = {
            "Spatial":0,
            "METIS":1
        }[name]


        economic_score = (
            metrics["cut"]
            /
            max(metrics["balance"],1)
        )


        row = {

            "dataset_id":dataset_id,

            "method_id":method_id,

            "runtime":runtime,

            "violations":metrics["cut"],

            "economic_score":economic_score
        }


        X = pd.DataFrame(
            [row]
        )[features]


        probability = model.predict_proba(
            X
        )[0][1]



        quality_score = (
            1 /
            max(metrics["cut"],1)
        )


        balance_score = (
            1 /
            max(metrics["balance"],1)
        )


        # geological continuity preference
        spatial_bonus = 0

        if name == "Spatial":
            spatial_bonus = 0.15



        final_score = (

            0.55 * probability

            +
            0.20 * quality_score

            +
            0.10 * balance_score

            +
            spatial_bonus

        )



        print(
            f"{name}: "
            f"ML={probability:.4f} "
            f"cut={metrics['cut']} "
            f"balance={metrics['balance']:.3f} "
            f"FINAL={final_score:.4f}"
        )



        ranked.append(
            (
                final_score,
                name,
                labels
            )
        )



    ranked.sort(
        reverse=True,
        key=lambda x:x[0]
    )



    score, selected_name, selected_labels = ranked[0]


    print(
        f"Hybrid V3.4 selected: "
        f"{selected_name} "
        f"(score={score:.4f})"
    )


    return selected_labels