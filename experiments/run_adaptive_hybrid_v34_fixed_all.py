from pathlib import Path
import subprocess


DATASETS = [
    "bauxitemed",
    "cucase",
    "cupipe",
    "mclaughlingeo"
]


def main():

    for dataset in DATASETS:

        print("\n")
        print("=" * 90)
        print(
            f"RUNNING HYBRID V3.4 FIXED : {dataset.upper()}"
        )
        print("=" * 90)


        cmd = [

            "../lg/.venv-linux/bin/python",

            "experiments/experiment_adaptive_solver.py",

            "--dataset",
            dataset,

            "--data",
            f"../lg/{dataset}.dat",

            "--method",
            "hybrid_v34",

            "--partitions",
            "8",

            "--policy",
            "heuristic",

            "--refinement",
            "add",

            "--output",
            f"results/adaptive_{dataset}_hybrid_v34_fixed_add.json"

        ]


        subprocess.run(cmd)


    print("\n")
    print("=" * 90)
    print(
        "HYBRID V3.4 FIXED FOUR DATASET RUN COMPLETE"
    )
    print("=" * 90)



if __name__ == "__main__":
    main()