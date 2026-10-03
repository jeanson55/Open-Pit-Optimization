"""
Hybrid V3.5 ML Adaptive Partition Selector

Uses trained Random Forest model to select
between Spatial and METIS partitions.
"""

from pathlib import Path
import pandas as pd
import joblib


from .spatial import spatial_partition
from .metis import metis_partition


MODEL_PATH = Path(
    "results/hybrid_v35_selector.pkl"
)


DATASET_IDS = {

    "bauxitemed":0,

    "cucase":1,

    "cupipe":2,

    "mclaughlingeo":3,

}



def detect_dataset(graph):

    n = graph.number_of_nodes()


    if n < 300000:
        return 0

    elif n < 700000:
        return 1

    elif n < 1100000:
        return 2

    else:
        return 3




def partition_metrics(graph, labels):


    sizes={}


    for p in labels.values():

        sizes[p] = sizes.get(p,0)+1



    cut=0


    for u,v in graph.edges():

        if labels[u] != labels[v]:

            cut += 1



    balance = (

        max(sizes.values())

        /

        min(sizes.values())

    )


    return {

        "cut":cut,

        "balance":balance

    }





def hybrid_v35_partition(

        graph,

        n_partitions=8,

        dataset_id=None

):


    print(
        "Hybrid V3.5: generating candidates"
    )



    if dataset_id is None:

        dataset_id = detect_dataset(graph)



    candidates=[]



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



    print(

        "Hybrid V3.5: ML ranking"

    )



    results=[]



    for name,labels in candidates:


        metrics = partition_metrics(

            graph,

            labels

        )


        method_id = {

            "Spatial":0,

            "METIS":1

        }[name]



        nodes = graph.number_of_nodes()



        # estimate economic indicators

        npv_per_node = (

            1.0 /
            max(metrics["cut"],1)

        )



        economic_score = (

            npv_per_node /

            (1 + metrics["balance"])

        )



        row = {


            "dataset_id":

                dataset_id,


            "method_id":

                method_id,


            "runtime":

                0.0,


            "violations":

                metrics["cut"],


            "npv_per_node":

                npv_per_node,


            "economic_score":

                economic_score

        }



        X = pd.DataFrame(

            [row]

        )[features]



        probability = model.predict_proba(

            X

        )[0][1]



        print(

            f"{name}: "

            f"probability={probability:.4f} "

            f"cut={metrics['cut']} "

            f"balance={metrics['balance']:.3f}"

        )



        results.append(

            (

                probability,

                name,

                labels

            )

        )



    results.sort(

        key=lambda x:x[0],

        reverse=True

    )



    probability,name,labels = results[0]



    print(

        f"Hybrid V3.5 selected: {name} "

        f"(score={probability:.4f})"

    )



    return labels