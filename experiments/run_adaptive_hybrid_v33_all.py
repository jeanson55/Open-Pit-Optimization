import json
import subprocess
from pathlib import Path
import time


DATASETS = {
    "bauxitemed": "../lg/bauxitemed.dat",
    "cucase": "../lg/cucase.dat",
    "cupipe": "../lg/cupipe.dat",
    "mclaughlingeo": "../lg/mclaughlingeo.dat",
}


OUTPUT_DIR = Path("results")


def run_experiment(dataset, datafile):

    print("\n")
    print("=" * 90)
    print(f"RUNNING HYBRID V3.3 : {dataset.upper()}")
    print("=" * 90)


    output = (
        OUTPUT_DIR /
        f"adaptive_{dataset}_hybrid_v33_add.json"
    )


    cmd = [
        "../lg/.venv-linux/bin/python",
        "experiments/experiment_adaptive_solver.py",

        "--dataset",
        dataset,

        "--data",
        datafile,

        "--method",
        "hybrid_v33",

        "--partitions",
        "8",

        "--policy",
        "heuristic",

        "--refinement",
        "add",

        "--output",
        str(output),
    ]


    start = time.perf_counter()


    subprocess.run(
        cmd,
        check=True
    )


    elapsed = time.perf_counter() - start


    print(
        f"\nCompleted {dataset}"
    )

    print(
        f"Wall time: {elapsed:.2f}s"
    )



def create_summary():

    print("\n")
    print("=" * 100)
    print("HYBRID V3.3 FINAL SUMMARY")
    print("=" * 100)


    rows=[]


    for dataset in DATASETS:

        file = (
            OUTPUT_DIR /
            f"adaptive_{dataset}_hybrid_v33_add.json"
        )


        if not file.exists():
            continue


        with open(file) as f:
            data=json.load(f)


        rows.append(
            {
                "dataset": dataset,
                "NPV": data["final_npv"],
                "runtime_s": data["total_runtime_s"],
                "partition_s": data["partition_runtime_s"],
                "solve_s": data["local_solve_runtime_s"],
                "violations": data["final_violations"],
                "initial_violations":
                    data["initial_violations"],
            }
        )


    print()

    print(
        f"{'Dataset':20}"
        f"{'NPV':15}"
        f"{'Runtime':12}"
        f"{'Violations':12}"
    )

    print("-"*60)


    for r in rows:

        print(
            f"{r['dataset']:20}"
            f"{r['NPV']:15,.0f}"
            f"{r['runtime_s']:12.2f}"
            f"{r['violations']:12}"
        )


    with open(
        OUTPUT_DIR /
        "hybrid_v33_final_summary.json",
        "w"
    ) as f:

        json.dump(
            rows,
            f,
            indent=4
        )


    print("\nSaved:")
    print(
        "results/hybrid_v33_final_summary.json"
    )



if __name__ == "__main__":


    OUTPUT_DIR.mkdir(
        exist_ok=True
    )


    for dataset,datafile in DATASETS.items():

        run_experiment(
            dataset,
            datafile
        )


    create_summary()