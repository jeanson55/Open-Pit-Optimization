"""Run paired Spatial/METIS partition sweeps and assemble V3.6 outcomes.

The historical eight-partition Spatial and METIS runs are reused. New
partition-count runs are stored separately so previous comparison artifacts
are preserved and incomplete sweeps can be resumed safely.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bauxitemed", "cucase", "cupipe", "mclaughlingeo")
METHODS = ("spatial", "metis")
DEFAULT_PARTITIONS = (4, 12)


def _output_path(dataset: str, method: str, k: int) -> Path:
    return ROOT / "results" / f"partition_grid_{dataset}_{method}_k{k}_add.json"


def _read_metrics(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _record(dataset: str, method: str, k: int, path: Path) -> dict:
    metrics = _read_metrics(path)
    quality = metrics.get("partition_quality", {})
    return {
        "dataset": dataset,
        "partitions": k,
        "method": method,
        "final_npv": metrics["final_npv"],
        "initial_npv": metrics["initial_npv"],
        "partition_runtime_s": metrics["partition_runtime_s"],
        "local_solve_runtime_s": metrics["local_solve_runtime_s"],
        "refinement_runtime_s": metrics["refinement_runtime_s"],
        "total_runtime_s": metrics["total_runtime_s"],
        "initial_violations": metrics["initial_violations"],
        "violation_cluster_count": metrics["violation_cluster_count"],
        "cascade_added_blocks": metrics["cascade_added_blocks"],
        "cascade_depth": metrics["cascade_depth"],
        "final_violations": metrics["final_violations"],
        "edge_cut": quality.get("edge_cut"),
        "cut_edges": quality.get("cut_edges"),
        "balance": quality.get("balance"),
        "min_part": quality.get("min_part"),
        "max_part": quality.get("max_part"),
        "result_file": str(path.relative_to(ROOT)),
    }


def _historical_k8_rows() -> list[dict]:
    # Use completed V3.6 Spatial and V3.5 METIS runs at eight partitions.
    rows = []
    for dataset, method, version in (
        (dataset, method, version)
        for dataset in DATASETS
        for method, version in (("spatial", "v36"), ("metis", "v35"))
    ):
        path = ROOT / "results" / f"adaptive_{dataset}_hybrid_{version}_add.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing historical 8-partition run: {path}")
        rows.append(_record(dataset, method, 8, path))
    return rows


def run(partitions: tuple[int, ...] = DEFAULT_PARTITIONS,
        datasets: tuple[str, ...] = DATASETS,
        resume: bool = True) -> list[dict]:
    interpreter = sys.executable
    for dataset in datasets:
        for k in partitions:
            for method in METHODS:
                output = _output_path(dataset, method, k)
                if resume and output.exists():
                    try:
                        _read_metrics(output)
                        print(f"Reuse completed: {output.name}", flush=True)
                        continue
                    except (OSError, json.JSONDecodeError):
                        pass
                command = [
                    interpreter,
                    str(ROOT / "experiments" / "experiment_adaptive_solver.py"),
                    "--dataset", dataset,
                    "--data", str(ROOT.parent / "lg" / f"{dataset}.dat"),
                    "--method", method,
                    "--partitions", str(k),
                    "--policy", "heuristic",
                    "--refinement", "add",
                    "--output", str(output),
                ]
                print(f"\n=== {dataset}: {method}, k={k} ===", flush=True)
                subprocess.run(command, cwd=ROOT, check=True)

    rows = _historical_k8_rows()
    for dataset in datasets:
        for k in partitions:
            for method in METHODS:
                path = _output_path(dataset, method, k)
                rows.append(_record(dataset, method, k, path))

    rows.sort(key=lambda row: (row["dataset"], row["partitions"], row["method"]))
    output_csv = ROOT / "results" / "hybrid_v36_partition_grid_outcomes.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_json = ROOT / "results" / "hybrid_v36_partition_grid_outcomes.json"
    output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved {output_csv.relative_to(ROOT)} ({len(rows)} paired outcomes)", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=int, nargs="+", default=DEFAULT_PARTITIONS)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--no-resume", action="store_true",
                        help="rerun even when a complete per-case JSON already exists")
    args = parser.parse_args()
    run(tuple(args.partitions), tuple(args.datasets), resume=not args.no_resume)


if __name__ == "__main__":
    main()
