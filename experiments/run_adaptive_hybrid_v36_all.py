"""Run V3.6 selection through the existing adaptive solver on four 3D models."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bauxitemed", "cucase", "cupipe", "mclaughlingeo")
OUTPUT_DIR = ROOT / "results"


def main() -> None:
    interpreter = sys.executable
    summary = []
    for dataset in DATASETS:
        output = OUTPUT_DIR / f"adaptive_{dataset}_hybrid_v36_add.json"
        command = [
            interpreter,
            str(ROOT / "experiments" / "experiment_adaptive_solver.py"),
            "--dataset", dataset,
            "--data", str(ROOT.parent / "lg" / f"{dataset}.dat"),
            "--method", "hybrid_v36",
            "--partitions", "8",
            "--policy", "heuristic",
            "--refinement", "add",
            "--output", str(output),
        ]
        print(f"\n=== Hybrid V3.6: {dataset} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        with output.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        selection = metrics.get("hybrid_v36_selection", {})
        summary.append({
            "dataset": dataset,
            "selected_method": selection.get("selected_method"),
            "final_npv": metrics["final_npv"],
            "final_violations": metrics["final_violations"],
            "total_runtime_s": metrics["total_runtime_s"],
            "partition_runtime_s": metrics["partition_runtime_s"],
            "ranked_candidates": selection.get("ranked_candidates", []),
            "result_file": str(output.relative_to(ROOT)),
        })
    summary_path = OUTPUT_DIR / "hybrid_v36_final_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
