"""
Hybrid V3.5 Complete Four Dataset Runner
"""

import subprocess
import json
from pathlib import Path


DATASETS = [

    (
        "bauxitemed",
        "../lg/bauxitemed.dat"
    ),

    (
        "cucase",
        "../lg/cucase.dat"
    ),

    (
        "cupipe",
        "../lg/cupipe.dat"
    ),

    (
        "mclaughlingeo",
        "../lg/mclaughlingeo.dat"
    )

]


OUTPUT_DIR = Path(
    "results"
)



def main():


    summary=[]


    for name, data in DATASETS:


        print("\n")
        print("="*90)
        print(
            f"RUNNING HYBRID V3.5 : {name.upper()}"
        )
        print("="*90)



        output = OUTPUT_DIR / (
            f"adaptive_{name}_hybrid_v35_add.json"
        )



        cmd = [

            "../lg/.venv-linux/bin/python",

            "experiments/experiment_adaptive_solver.py",

            "--dataset",
            name,

            "--data",
            data,

            "--method",
            "hybrid_v35",

            "--partitions",
            "8",

            "--policy",
            "heuristic",

            "--refinement",
            "add",

            "--output",
            str(output)

        ]



        subprocess.run(
            cmd,
            check=True
        )



        with open(output) as f:

            result=json.load(f)



        summary.append(

            {

                "dataset":
                    name,

                "final_npv":
                    result.get(
                        "final_npv",
                        0
                    ),

                "runtime":
                    result.get(
                        "total_runtime_s",
                        0
                    )

            }

        )



    summary_file = OUTPUT_DIR / (
        "hybrid_v35_final_summary.json"
    )



    with open(summary_file,"w") as f:

        json.dump(
            summary,
            f,
            indent=2
        )



    print("\n")
    print("="*90)
    print(
        "HYBRID V3.5 FOUR DATASET RUN COMPLETE"
    )
    print("="*90)



    for item in summary:

        print(
            item
        )


    print()

    print(
        "Saved:"
    )

    print(
        summary_file
    )



if __name__=="__main__":

    main()