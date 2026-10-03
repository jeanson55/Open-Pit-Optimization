"""
Final Hybrid Model Comparison

Compares:
- Hybrid V3.1
- Hybrid V3.2
- Hybrid V3.4
- Hybrid V3.5
"""


import json
import pandas as pd
from pathlib import Path



RESULTS = Path("results")

OUTPUT = RESULTS / "final_hybrid_comparison.csv"



def load_json(path):

    with open(path) as f:
        return json.load(f)



def main():


    rows = []


    # -------------------------------
    # Historical results V3.1/V3.2
    # -------------------------------

    summary_file = RESULTS / "partition_analysis_summary.json"


    if summary_file.exists():

        data = load_json(summary_file)


        for dataset, info in data.items():

            for item in info["results"]:

                rows.append({

                    "dataset":
                        dataset,

                    "method":
                        item["method"],

                    "npv":
                        item["npv"],

                    "runtime":
                        item["runtime"],

                    "violations":
                        item["violations"]

                })



    # -------------------------------
    # V3.4 results
    # -------------------------------

    v34_file = RESULTS / "hybrid_v34_comparison_report.csv"


    if v34_file.exists():

        df = pd.read_csv(v34_file)


        for _,r in df.iterrows():

            if r["method"] == "Hybrid_V34":

                rows.append({

                    "dataset":
                        r["dataset"],

                    "method":
                        "Hybrid_V34",

                    "npv":
                        r["npv"],

                    "runtime":
                        r["runtime"],

                    "violations":
                        r["violations"]

                })



    # -------------------------------
    # V3.5 results
    # -------------------------------

    v35_file = RESULTS / "hybrid_v35_final_summary.json"


    if v35_file.exists():

        data = load_json(v35_file)


        for item in data:

            rows.append({

                "dataset":
                    item["dataset"],

                "method":
                    "Hybrid_V35",

                "npv":
                    item["final_npv"],

                "runtime":
                    item["runtime"],

                "violations":
                    0

            })



    df = pd.DataFrame(rows)



    # best NPV per dataset

    df["best_npv"] = (

        df.groupby("dataset")["npv"]

        .transform("max")

    )


    df["gap_percent"] = (

        (df["best_npv"] - df["npv"])

        /

        df["best_npv"]

        *

        100

    )


    df["selected"] = (

        df["npv"]

        ==

        df["best_npv"]

    )



    df = df.sort_values(

        [
            "dataset",
            "npv"
        ],

        ascending=[
            True,
            False
        ]

    )



    df.to_csv(

        OUTPUT,

        index=False

    )



    print("="*100)
    print(
        "FINAL HYBRID MODEL COMPARISON"
    )
    print("="*100)


    print(df.to_string(index=False))


    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT
    )



if __name__=="__main__":

    main()