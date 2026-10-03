"""Boundary-refine the best measured partition/count for each deposit.

Selection is by observed add-only NPV within the completed grid. This is an
analysis of refinement capability, not a prospective partition selector.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bauxitemed", "cucase", "cupipe", "mclaughlingeo")


def run() -> list[dict]:
    outcomes_path = ROOT / "results" / "hybrid_v36_partition_grid_outcomes.csv"
    with outcomes_path.open(newline="", encoding="utf-8") as stream:
        outcomes = list(csv.DictReader(stream))
    best = {}
    for row in outcomes:
        dataset = row["dataset"]
        if dataset not in DATASETS:
            continue
        if dataset not in best or float(row["final_npv"]) > float(best[dataset]["final_npv"]):
            best[dataset] = row

    summary = []
    for dataset in DATASETS:
        row = best[dataset]
        method = row["method"]
        k = int(row["partitions"])
        output = ROOT / "results" / f"partition_grid_{dataset}_{method}_k{k}_boundary.json"
        command = [
            sys.executable,
            str(ROOT / "experiments" / "experiment_adaptive_solver.py"),
            "--dataset", dataset,
            "--data", str(ROOT.parent / "lg" / f"{dataset}.dat"),
            "--method", method,
            "--partitions", str(k),
            "--policy", "heuristic",
            "--refinement", "boundary",
            "--output", str(output),
        ]
        print(f"\n=== boundary refinement: {dataset}, {method}, k={k} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        with output.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        add_npv = float(row["final_npv"])
        summary.append({
            "dataset": dataset,
            "selected_method": method,
            "partitions": k,
            "add_only_npv": add_npv,
            "boundary_npv": float(metrics["final_npv"]),
            "boundary_npv_gain": float(metrics["final_npv"]) - add_npv,
            "boundary_relative_gain_percent": (
                (float(metrics["final_npv"]) - add_npv) / max(abs(add_npv), 1.0) * 100
            ),
            "initial_violations": metrics["initial_violations"],
            "final_violations": metrics["final_violations"],
            "boundary_nodes": metrics.get("boundary_nodes"),
            "boundary_arcs": metrics.get("boundary_arcs"),
            "boundary_iterations": metrics.get("boundary_refinement_iterations"),
            "boundary_solver_runtime_s": metrics.get("boundary_solver_runtime_s"),
            "refinement_runtime_s": metrics["refinement_runtime_s"],
            "total_runtime_s": metrics["total_runtime_s"],
            "partition_runtime_s": metrics["partition_runtime_s"],
            "result_file": str(output.relative_to(ROOT)),
        })
    summary_path = ROOT / "results" / "hybrid_v36_best_boundary_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = ROOT / "results" / "hybrid_v36_best_boundary_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved {summary_path.relative_to(ROOT)}", flush=True)
    return summary


if __name__ == "__main__":
    run()
