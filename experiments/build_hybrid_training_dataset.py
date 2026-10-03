import json
import csv
from pathlib import Path


RESULTS = Path("results")

OUTPUT = RESULTS / "hybrid_training_data.csv"


DATASETS = [
    "bauxitemed",
    "cucase",
    "cupipe",
    "mclaughlingeo",
]


METHODS = {
    "Spatial": "_spatial_add.json",
    "METIS": "_metis_add.json",
    "Hybrid_V31": "_hybrid_add.json",
    "Hybrid_V32": "_hybrid_v32_add.json",
}



def load_json(path):

    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)



def main():

    rows = []


    for dataset in DATASETS:

        candidates = []


        # -------------------------
        # Load all methods
        # -------------------------

        for method, suffix in METHODS.items():

            file = RESULTS / (
                f"adaptive_{dataset}{suffix}"
            )

            data = load_json(file)


            if data is None:
                continue


            row = {

                "dataset": dataset,

                "method": method,

                "final_npv":
                    data["final_npv"],

                "total_runtime":
                    data["total_runtime_s"],

                "partition_runtime":
                    data["partition_runtime_s"],

                "solver_runtime":
                    data["local_solve_runtime_s"],

                "initial_violations":
                    data["initial_violations"],

                "final_violations":
                    data["final_violations"],

            }


            candidates.append(row)



        if not candidates:
            continue



        # -------------------------
        # Best NPV label
        # -------------------------

        best_npv = max(
            x["final_npv"]
            for x in candidates
        )

        for row in candidates:
            row["best_choice"] = (
                1
                if row["final_npv"] == best_npv
                else 0
            )

            rows.append(row)



    # -------------------------
    # Write CSV
    # -------------------------

    OUTPUT.parent.mkdir(
        exist_ok=True
    )


    fields = [
        "dataset",
        "method",
        "final_npv",
        "total_runtime",
        "partition_runtime",
        "solver_runtime",
        "initial_violations",
        "final_violations",
        "best_choice",
    ]


    with open(
        OUTPUT,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)



    print("="*80)
    print("HYBRID V3.3 TRAINING DATASET CREATED")
    print("="*80)

    print(
        f"Rows: {len(rows)}"
    )

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()