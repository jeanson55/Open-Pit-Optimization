"""
Hybrid V3.4 Comparison Report Generator

Combines:
- Hybrid V3.1
- Hybrid V3.2
- Hybrid V3.3
- Hybrid V3.4
- Spatial
- METIS

and creates a unified comparison table.
"""

import json
from pathlib import Path

import pandas as pd


OUTPUT = Path(
    "results/hybrid_v34_comparison_report.csv"
)


def load_json(path):

    with open(path) as f:
        return json.load(f)



def add_row(
        rows,
        dataset,
        method,
        npv,
        runtime,
        violations
):

    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "npv": float(npv),
            "runtime": float(runtime),
            "violations": int(violations)
        }
    )



def process_partition_analysis(rows):

    path = Path(
        "results/partition_analysis_summary.json"
    )


    if not path.exists():
        return


    data = load_json(path)


    for dataset, info in data.items():

        for item in info["results"]:

            add_row(
                rows,
                dataset,
                item["method"],
                item["npv"],
                item["runtime"],
                item["violations"]
            )



def process_v33(rows):

    path = Path(
        "results/hybrid_v33_final_summary.json"
    )


    if not path.exists():
        return


    data = load_json(path)


    # handle dictionary format
    if isinstance(data, dict):

        items = []


        for key, value in data.items():

            if isinstance(value, dict):

                if "dataset" not in value:
                    value["dataset"] = key

                items.append(value)

        data = items



    for item in data:

        dataset = item.get(
            "dataset"
        )


        npv = item.get(
            "final_npv",
            item.get(
                "npv",
                0
            )
        )


        runtime = item.get(
            "total_runtime_s",
            item.get(
                "runtime",
                0
            )
        )


        violations = item.get(
            "final_violations",
            item.get(
                "violations",
                0
            )
        )


        add_row(
            rows,
            dataset,
            "Hybrid_V33",
            npv,
            runtime,
            violations
        )



def process_v34(rows):

    files = Path(
        "results"
    ).glob(
        "adaptive_*_hybrid_v34_fixed_add.json"
    )


    for path in files:

        data = load_json(path)


        add_row(

            rows,

            data["dataset"],

            "Hybrid_V34",

            data.get(
                "final_npv",
                data.get(
                    "npv",
                    0
                )
            ),

            data.get(
                "total_runtime_s",
                data.get(
                    "runtime",
                    0
                )
            ),

            data.get(
                "final_violations",
                data.get(
                    "violations",
                    0
                )
            )

        )



def main():

    rows = []


    process_partition_analysis(rows)

    process_v33(rows)

    process_v34(rows)



    df = pd.DataFrame(rows)



    if df.empty:

        print(
            "No results found"
        )

        return



    # remove duplicates
    df = df.drop_duplicates(
        subset=[
            "dataset",
            "method"
        ],
        keep="last"
    )



    # calculate best NPV per dataset

    df["best_npv"] = (
        df.groupby(
            "dataset"
        )["npv"]
        .transform("max")
    )


    df["gap_percent"] = (

        (
            df["best_npv"]
            -
            df["npv"]
        )

        /

        df["best_npv"]

        *

        100

    )


    df["selected"] = (
        df["gap_percent"]
        <
        1e-9
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
        "HYBRID V3.4 COMPARISON REPORT"
    )

    print("="*100)


    print(
        df.to_string(
            index=False
        )
    )


    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT
    )



if __name__ == "__main__":

    main()