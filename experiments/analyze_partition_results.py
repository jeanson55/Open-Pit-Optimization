import json
from pathlib import Path


RESULTS = Path("results")


DATASETS = [
    "bauxitemed",
    "cucase",
    "cupipe",
    "mclaughlingeo",
]


METHODS = {
    "Spatial": "_spatial_add.json",
    "METIS": "_metis_add.json",
    "Hybrid_V32": "_hybrid_v32_add.json",
    "Hybrid_V31": "_hybrid_add.json",
}



def load_result(dataset, suffix):

    path = RESULTS / (
        f"adaptive_{dataset}{suffix}"
    )

    if not path.exists():
        return None

    with open(path) as f:
        return json.load(f)



def main():

    print("="*110)
    print("ADAPTIVE PARTITION RESULT ANALYSIS")
    print("="*110)


    summary = {}


    for dataset in DATASETS:

        print("\n")
        print("-"*110)
        print(dataset.upper())
        print("-"*110)


        rows = []


        for method, suffix in METHODS.items():

            data = load_result(
                dataset,
                suffix
            )


            if data is None:
                continue


            rows.append(
                {
                    "method": method,
                    "npv": data["final_npv"],
                    "runtime": data["total_runtime_s"],
                    "violations": data["initial_violations"],
                }
            )


        if not rows:
            continue


        best_npv = max(
            r["npv"]
            for r in rows
        )


        best_method = [
            r["method"]
            for r in rows
            if r["npv"] == best_npv
        ][0]


        print(
            f"{'Method':15}"
            f"{'NPV':15}"
            f"{'Gap %':12}"
            f"{'Runtime(s)':15}"
            f"{'Violations':12}"
        )

        print("-"*70)


        for r in rows:

            gap = (
                100 *
                (best_npv-r["npv"])
                /
                best_npv
            )


            print(
                f"{r['method']:15}"
                f"{r['npv']:15,.0f}"
                f"{gap:12.3f}"
                f"{r['runtime']:15.2f}"
                f"{r['violations']:12}"
            )


        print()

        print(
            "Best:",
            best_method
        )


        summary[dataset] = {
            "best_method": best_method,
            "best_npv": best_npv,
            "results": rows,
        }



    out = RESULTS / "partition_analysis_summary.json"


    with open(out,"w") as f:
        json.dump(
            summary,
            f,
            indent=2
        )


    print("\n")
    print("="*110)
    print("Saved:")
    print(out)



if __name__ == "__main__":
    main()