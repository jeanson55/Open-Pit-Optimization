"""Repeat Experiment E's four real partition methods at selected counts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "lg" / "bauxitemed.dat"
PIT = ROOT / "results" / "bauxitemed_baseline_pit.npy"
COUNTS = (4, 12)


def run(counts: tuple[int, ...] = COUNTS) -> list[dict]:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if not PIT.exists():
        raise FileNotFoundError(PIT)
    results = []
    for k in counts:
        output = ROOT / "results" / f"experiment_e_partitioning_k{k}.json"
        command = [
            sys.executable,
            str(ROOT / "experiments" / "experiment_e_partitioning.py"),
            "--data", str(DATA),
            "--pit", str(PIT),
            "--output", str(output),
            "--partitions", str(k),
        ]
        print(f"\n=== Experiment E, bauxitemed, k={k} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        with output.open(encoding="utf-8") as stream:
            report = json.load(stream)
        for method, metrics in report["methods"].items():
            if metrics.get("implementation", "").find("fallback") >= 0:
                raise RuntimeError(f"{method} used a fallback at k={k}: {metrics['implementation']}")
            results.append({
                "dataset": "bauxitemed",
                "partitions": k,
                "method": method,
                "balance": metrics["balance"],
                "cut_arcs": metrics["cut_edges"],
                "cut_fraction": metrics["edge_cut"],
                "initial_violations": metrics["initial_violations"],
                "violation_clusters": metrics["violation_cluster_count"],
                "cascade_added_blocks": metrics["cascade_added_blocks"],
                "cascade_depth": metrics["cascade_depth"],
                "final_violations": metrics["final_violations"],
                "final_npv": metrics["final_npv"],
                "partition_runtime_s": metrics["partition_runtime_s"],
                "local_solve_runtime_s": metrics["local_solve_runtime_s"],
                "refinement_runtime_s": metrics["refinement_runtime_s"],
                "implementation": metrics.get("implementation", "native"),
                "source_json": str(output.relative_to(ROOT)),
            })
    summary = ROOT / "results" / "experiment_e_partition_counts_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {summary.relative_to(ROOT)}", flush=True)
    return results


if __name__ == "__main__":
    run()
