"""
Hybrid V3.5 Training Dataset Builder

Creates training data for the final economic partition selector.

Sources:
- partition_analysis_summary.json
- adaptive_*_hybrid_v34_fixed_add.json

Target:
best_choice

Preference:
1. Highest NPV
2. Lowest violations if NPV is equal
"""

import json
from pathlib import Path

import pandas as pd


OUTPUT = Path(
    "results/hybrid_v35_training_data.csv"
)



def load_json(path):

    with open(path) as f:
        return json.load(f)



def main():

    rows = []


    # -------------------------------------------------
    # Previous benchmark results
    # -------------------------------------------------

    benchmark = load_json(
        "results/partition_analysis_summary.json"
    )


    for dataset, info in benchmark.items():

        for item in info["results"]:

            rows.append({

                "dataset": dataset,

                "method": item["method"],

                "npv":
                    item["npv"],

                "runtime":
                    item["runtime"],

                "violations":
                    item["violations"]

            })



    # -------------------------------------------------
    # Hybrid V3.4 results
    # -------------------------------------------------

    for file in Path("results").glob(
        "adaptive_*_hybrid_v34_fixed_add.json"
    ):

        data = load_json(file)


        rows.append({

            "dataset":
                data["dataset"],

            "method":
                "Hybrid_V34",

            "npv":
                data["final_npv"],

            "runtime":
                data["total_runtime_s"],

            "violations":
                data.get(
                    "final_violations",
                    0
                )

        })



    df = pd.DataFrame(rows)



    # -------------------------------------------------
    # Feature engineering
    # -------------------------------------------------

    dataset_map = {

        name:i

        for i,name in enumerate(
            df["dataset"].unique()
        )

    }


    method_map = {

        name:i

        for i,name in enumerate(
            df["method"].unique()
        )

    }



    df["dataset_id"] = (
        df["dataset"]
        .map(dataset_map)
    )


    df["method_id"] = (
        df["method"]
        .map(method_map)
    )



    # estimated graph size proxy
    nodes = {

        "bauxitemed":240502,
        "cucase":1045952,
        "cupipe":988042,
        "mclaughlingeo":1241546

    }


    df["nodes"] = (
        df["dataset"]
        .map(nodes)
    )


    df["npv_per_node"] = (
        df["npv"]
        /
        df["nodes"]
    )



    df["economic_score"] = (

        df["npv_per_node"]

        /
        (
            1
            +
            df["violations"]
        )

    )



    # -------------------------------------------------
    # Label generation
    # -------------------------------------------------

    df["best_choice"] = 0



    for dataset in df["dataset"].unique():

        mask = (
            df["dataset"]
            ==
            dataset
        )


        local = df.loc[
            mask
        ].sort_values(

            [
                "npv",
                "violations"

            ],

            ascending=[
                False,
                True
            ]

        )


        winner = local.index[0]


        df.loc[
            winner,
            "best_choice"
        ] = 1



    df.to_csv(
        OUTPUT,
        index=False
    )


    print("="*90)

    print(
        "HYBRID V3.5 TRAINING DATASET"
    )

    print("="*90)

    print(df)


    print()

    print(
        "Rows:",
        len(df)
    )

    print(
        "Saved:",
        OUTPUT
    )



if __name__ == "__main__":

    main()