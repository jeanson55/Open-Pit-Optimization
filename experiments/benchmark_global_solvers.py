"""Benchmark standalone LiteralLG and Pseudoflow on one block model."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpit.hybrid_optimizer import HybridOptimizer
from gpit.lg_literal_loader import build_from_gpit


def run(data: Path, output: Path, nx_dim: int, ny_dim: int, nz_dim: int,
        precedence: str = "1x5") -> dict:
    values_array = np.loadtxt(data, dtype=np.float64)
    expected = nx_dim * ny_dim * nz_dim
    if values_array.size != expected:
        raise ValueError(
            f"{data.name} has {values_array.size:,} values; dimensions "
            f"{nx_dim}x{ny_dim}x{nz_dim} require {expected:,}"
        )
    model = pd.DataFrame({"EPV": values_array})
    bm = {
        "nx": nx_dim, "ny": ny_dim, "nz": nz_dim,
        "xorg": 0, "yorg": 0, "zorg": 0,
        "xs": 1, "ys": 1, "zs": 1, "volume": 1,
    }

    graph_start = time.perf_counter()
    values, arcs, diagnostics = build_from_gpit(model, bm, "EPV", precedence)
    graph_seconds = time.perf_counter() - graph_start
    solver = HybridOptimizer(values, arcs, model, bm)
    lg_nodes = solver.solve_lg()
    lg_violations = sum(
        1 for u, successors in arcs.items() if u in lg_nodes
        for v in successors if v not in lg_nodes
    )
    hpf_vector = solver.solve_hpf(precedence=precedence)
    hpf_nodes = {i + 1 for i, picked in enumerate(hpf_vector) if picked}
    hpf_violations = sum(
        1 for u, successors in arcs.items() if u in hpf_nodes
        for v in successors if v not in hpf_nodes
    )
    comparison = solver.compare()

    result = {
        "dataset": data.stem,
        "data_path": str(data),
        "dimensions": [nx_dim, ny_dim, nz_dim],
        "precedence_template": precedence,
        "graph_build_runtime_s": graph_seconds,
        **diagnostics,
        "lg_runtime_s": solver.lg_time_s,
        "lg_blocks": len(lg_nodes),
        "lg_npv": solver.lg_npv,
        "lg_violations": lg_violations,
        "hpf_runtime_s": solver.hpf_time_s,
        "hpf_blocks": len(hpf_nodes),
        "hpf_npv": solver.hpf_npv,
        "hpf_violations": hpf_violations,
        "npv_delta_lg_minus_hpf": solver.lg_npv - solver.hpf_npv,
        "comparison": comparison,
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_memory_scope": "whole benchmark process, not separated by solver",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "dataset", "graph_nodes", "precedence_arcs", "lg_runtime_s", "lg_npv",
        "lg_violations", "hpf_runtime_s", "hpf_npv", "hpf_violations",
        "npv_delta_lg_minus_hpf", "peak_process_rss_mb")}, indent=2))
    print(f"Wrote {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/BauxiteMed/bauxitemed.dat"))
    parser.add_argument("--output", type=Path, default=Path("results/global_solver_benchmark_bauxitemed.json"))
    parser.add_argument("--nx", type=int, default=120)
    parser.add_argument("--ny", type=int, default=120)
    parser.add_argument("--nz", type=int, default=26)
    parser.add_argument("--precedence", default="1x5")
    args = parser.parse_args()
    run(args.data, args.output, args.nx, args.ny, args.nz, args.precedence)


if __name__ == "__main__":
    main()
