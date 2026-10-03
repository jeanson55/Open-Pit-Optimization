"""Run genuine spectral partitions through the shared adaptive HPF pipeline.

This extends the Spatial/METIS paired grid to spectral partitions at matched
counts. It fails closed if SciPy or scikit-learn is unavailable, and each
completed case is saved separately so the sweep can resume.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bauxitemed", "cucase", "cupipe", "mclaughlingeo")
DEFAULT_PARTITIONS = (4,)


def _case_path(dataset: str, k: int) -> Path:
    return ROOT / "results" / f"partition_grid_{dataset}_spectral_k{k}_add.json"


def _record(dataset: str, k: int, path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics["final_violations"] != 0:
        raise RuntimeError(f"spectral result is infeasible: {path}")
    if metrics["partition_method"] != "spectral":
        raise RuntimeError(f"wrong partition method recorded in {path}")
    quality = metrics["partition_quality"]
    return {
        "dataset": dataset,
        "partitions": k,
        "method": "spectral",
        "final_npv": metrics["final_npv"],
        "partition_runtime_s": metrics["partition_runtime_s"],
        "local_solve_runtime_s": metrics["local_solve_runtime_s"],
        "refinement_runtime_s": metrics["refinement_runtime_s"],
        "total_runtime_s": metrics["total_runtime_s"],
        "initial_violations": metrics["initial_violations"],
        "violation_cluster_count": metrics["violation_cluster_count"],
        "cascade_added_blocks": metrics["cascade_added_blocks"],
        "cascade_depth": metrics["cascade_depth"],
        "final_violations": metrics["final_violations"],
        "edge_cut": quality["edge_cut"],
        "cut_edges": quality["cut_edges"],
        "balance": quality["balance"],
        "min_part": quality["min_part"],
        "max_part": quality["max_part"],
        "result_file": str(path.relative_to(ROOT)),
    }


def run(datasets: tuple[str, ...] = DATASETS,
        partitions: tuple[int, ...] = DEFAULT_PARTITIONS,
        resume: bool = True) -> list[dict]:
    missing = [name for name in ("scipy", "sklearn")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"Genuine spectral run unavailable; missing {missing}")

    for dataset in datasets:
        for k in partitions:
            output = _case_path(dataset, k)
            if resume and output.exists():
                try:
                    _record(dataset, k, output)
                    print(f"Reuse completed: {output.name}", flush=True)
                    continue
                except (OSError, json.JSONDecodeError, KeyError, ValueError):
                    pass
            command = [
                sys.executable,
                str(ROOT / "experiments" / "experiment_adaptive_solver.py"),
                "--dataset", dataset,
                "--data", str(ROOT.parent / "lg" / f"{dataset}.dat"),
                "--method", "spectral",
                "--partitions", str(k),
                "--policy", "heuristic",
                "--refinement", "add",
                "--output", str(output),
            ]
            print(f"\n=== spectral adaptive run: {dataset}, k={k} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            _record(dataset, k, output)

    rows = [_record(dataset, k, _case_path(dataset, k))
            for dataset in datasets for k in partitions]
    rows.sort(key=lambda row: (row["dataset"], row["partitions"]))
    json_path = ROOT / "results" / "spectral_partition_extension_summary.json"
    csv_path = ROOT / "results" / "spectral_partition_extension_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {json_path.relative_to(ROOT)} and {csv_path.relative_to(ROOT)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--partitions", type=int, nargs="+", default=DEFAULT_PARTITIONS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run(tuple(args.datasets), tuple(args.partitions), resume=not args.no_resume)


if __name__ == "__main__":
    main()
