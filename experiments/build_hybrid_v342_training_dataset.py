import json
import pandas as pd
from pathlib import Path


INPUT = Path(
    "results/partition_analysis_summary.json"
)

OUTPUT = Path(
    "results/hybrid_v342_training_data.csv"
)


METHOD_MAP = {
    "Spatial": 0,
    "METIS": 1,
    "Hybrid_V31": 2,
    "Hybrid_V32": 3,
}


def main():

    with open(INPUT) as f:
        data = json.load(f)


    rows = []


    for dataset, info in data.items():

        dataset_id = list(data.keys()).index(dataset)

        for item in info["results"]:

            method = item["method"]


            # synthetic partition features
            # extracted from existing benchmark results

            npv = item["npv"]

            runtime = item["runtime"]

            violations = item["violations"]


            rows.append({

                "dataset": dataset,

                "method": method,

                "dataset_id": dataset_id,

                "method_id": METHOD_MAP[method],

                "final_npv": npv,

                "runtime": runtime,

                "violations": violations,

                # proxy features
                "economic_score":
                    npv / max(runtime,1),

            })


    df = pd.DataFrame(rows)



    df["best_choice"] = 0


    for dataset in df["dataset"].unique():

        mask = df.dataset == dataset

        best = df.loc[
            mask,
            "final_npv"
        ].max()


        df.loc[
            mask &
            (df.final_npv == best),
            "best_choice"
        ] = 1



    df.to_csv(
        OUTPUT,
        index=False
    )


    print("="*80)
    print("HYBRID V3.4.2 TRAINING DATASET")
    print("="*80)

    print(df)

    print()
    print("Saved:", OUTPUT)



if __name__=="__main__":
    main()