import subprocess
from pathlib import Path


BASE = Path("../lg")

DATASETS = {
    "bauxitemed": "bauxitemed.dat",
    "cucase": "cucase.dat",
    "cupipe": "cupipe.dat",
    "mclaughlingeo": "mclaughlingeo.dat",
}


PYTHON = "../lg/.venv-linux/bin/python"


def run_experiment(dataset, filename):

    output = (
        f"results/"
        f"adaptive_{dataset}_hybrid_v32_add.json"
    )

    cmd = [
        PYTHON,
        "experiments/experiment_adaptive_solver.py",

        "--dataset",
        dataset,

        "--data",
        str(BASE / filename),

        "--method",
        "hybrid",

        "--partitions",
        "8",

        "--policy",
        "heuristic",

        "--refinement",
        "add",

        "--output",
        output,
    ]


    print("\n")
    print("=" * 90)
    print(f"RUNNING HYBRID V3.2 : {dataset.upper()}")
    print("=" * 90)


    subprocess.run(
        cmd,
        check=True
    )


def main():

    for dataset, filename in DATASETS.items():

        run_experiment(
            dataset,
            filename
        )


    print("\n")
    print("=" * 90)
    print("ALL HYBRID V3.2 ADAPTIVE EXPERIMENTS COMPLETE")
    print("=" * 90)


    print(
        """
Generated files:

results/
    adaptive_bauxitemed_hybrid_v32_add.json
    adaptive_cucase_hybrid_v32_add.json
    adaptive_cupipe_hybrid_v32_add.json
    adaptive_mclaughlingeo_hybrid_v32_add.json
"""
    )


if __name__ == "__main__":
    main()