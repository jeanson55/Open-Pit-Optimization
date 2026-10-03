import json
import pandas as pd
from pathlib import Path


INPUT = Path(
    "results/partition_analysis_summary.json"
)

OUTPUT = Path(
    "results/hybrid_v34_training_data.csv"
)


METHOD_MAP = {
    "Spatial": 0,
    "METIS": 1,
    "Hybrid_V31": 2,
    "Hybrid_V32": 3,
}


DATASET_NODES = {

    "bauxitemed": 240502,

    "cucase": 1045952,

    "cupipe": 988042,

    "mclaughlingeo": 1241546,

}



def main():

    with open(INPUT) as f:

        data = json.load(f)



    rows = []



    for dataset, info in data.items():

        results = info["results"]


        dataset_id = list(data.keys()).index(dataset)



        for item in results:


            method = item["method"]



            npv = item["npv"]

            runtime = item["runtime"]

            violations = item["violations"]



            nodes = DATASET_NODES[dataset]



            rows.append({

                "dataset": dataset,


                "method": method,


                "dataset_id": dataset_id,


                "method_id":
                    METHOD_MAP[method],


                "final_npv": npv,


                "runtime": runtime,


                "violations": violations,


                # New economic features

                "economic_score":
                    npv / max(violations, 1),


                "npv_per_node":
                    npv / nodes,


                "nodes":
                    nodes,

            })



    df = pd.DataFrame(rows)



    # determine winning method

    df["best_choice"] = 0



    for dataset in df["dataset"].unique():


        mask = (

            df["dataset"] == dataset

        )



        best_npv = df.loc[

            mask,

            "final_npv"

        ].max()



        df.loc[

            mask &

            (df["final_npv"] == best_npv),

            "best_choice"

        ] = 1




    df.to_csv(

        OUTPUT,

        index=False

    )



    print("="*80)

    print(
        "HYBRID V3.4 ECONOMIC TRAINING DATASET CREATED"
    )

    print("="*80)



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