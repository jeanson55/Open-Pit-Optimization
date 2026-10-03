"""Experiment E: compare graph partitions and repaired local solutions on bauxitemed."""
from __future__ import annotations

import argparse
import json
import time
import sys
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from partition.hybrid import hybrid_partition
from partition.hybrid_spatial_graph import hybrid_spatial_graph_partition
from partition.metis import metis_partition
from partition.metrics import evaluate_partition
from partition.spatial import spatial_partition
from partition.spectral import spectral_partition
from optimization.partitioned import solve_partitioned


def cut_arc_cluster_metrics(graph, labels):
    """Summarize connected groups of cross-partition precedence arcs."""
    import networkx as nx

    cut_arcs = [(u, v) for u, v in graph.edges if labels[u] != labels[v]]
    cut_graph = nx.Graph()
    cut_graph.add_edges_from(cut_arcs)
    sizes = sorted((len(c) for c in nx.connected_components(cut_graph)), reverse=True)
    return {
        "cut_arc_cluster_count": len(sizes),
        "largest_cut_arc_cluster_nodes": sizes[0] if sizes else 0,
        "cut_arc_cluster_sizes_top10": sizes[:10],
    }


def run(data_path: Path, pit_path: Path, output_path: Path, k: int = 9,
        selected_methods: list[str] | None = None):
    from graph.load_graph import load_graph

    graph, frame = load_graph(str(data_path))
    pit = np.load(pit_path).astype(bool)
    if len(pit) != len(frame):
        raise ValueError(f"Pit length {len(pit)} does not match block count {len(frame)}")
    values = frame["EPV"].to_numpy(dtype=np.float64)
    baseline = {i + 1 for i, selected in enumerate(pit) if selected}
    nx_dim, ny_dim = 120, 120
    graph.remove_nodes_from([n for n in (0, len(frame) + 1) if n in graph])
    for node in graph.nodes:
        i = int(node) - 1
        graph.nodes[node].update({
            "x": float(i % nx_dim),
            "y": float((i // nx_dim) % ny_dim),
            "z": float(i // (nx_dim * ny_dim)),
            "value": float(values[i]),
        })
    baseline_npv = float(values[pit].sum())
    methods = {
        "bench_spatial": lambda: spatial_partition(graph, k),
        "spectral": lambda: spectral_partition(graph, k),
        "metis": lambda: metis_partition(graph, k),
        "hybrid_spatial_graph": lambda: hybrid_partition(graph, k),
        "hybrid_spatial_graph_v4_1": lambda: hybrid_spatial_graph_partition(graph, k),
    }
    if selected_methods:
        unknown = set(selected_methods) - set(methods)
        if unknown:
            raise ValueError(f"unknown methods requested: {sorted(unknown)}")
        methods = {name: methods[name] for name in selected_methods}
    results = {"dataset": "bauxitemed", "partitions_requested": k, "methods": {}}
    metis_available = importlib.util.find_spec("pymetis") is not None
    spectral_available = (importlib.util.find_spec("scipy") is not None
                         and importlib.util.find_spec("sklearn") is not None)
    cached_labels = None
    cached_solution = None
    cached_solve_metrics = None
    cached_method = None
    for name, method in methods.items():
        start = time.perf_counter()
        labels = method()
        elapsed = time.perf_counter() - start
        metrics = evaluate_partition(graph, labels)
        # Cut arcs are partition boundaries, not closure violations in a pit.
        metrics.pop("violations", None)
        metrics.update(cut_arc_cluster_metrics(graph, labels))
        metrics["partition_runtime_s"] = elapsed
        if name == "hybrid_spatial_graph_v4_1":
            metrics["hybrid_refinement"] = graph.graph.get(
                "hybrid_spatial_graph_diagnostics", {}
            )
        if cached_labels is not None and labels == cached_labels:
            final_solution = cached_solution
            solve_metrics = dict(cached_solve_metrics)
            metrics["solver_reused_from"] = cached_method
        else:
            final_solution, solve_metrics = solve_partitioned(graph, labels, repair_mode="add")
            cached_labels = labels.copy()
            cached_solution = final_solution
            cached_solve_metrics = solve_metrics.copy()
            cached_method = name
        metrics.update(solve_metrics)
        metrics["baseline_npv"] = baseline_npv
        metrics["baseline_violations"] = sum(
            1 for u, v in graph.edges if u in baseline and v not in baseline
        )
        metrics["status"] = "completed: local closure solves merged and repaired"
        if name == "metis" and not metis_available:
            metrics["implementation"] = "spatial fallback; pymetis unavailable"
        if name == "spectral" and not spectral_available:
            metrics["implementation"] = "spatial fallback; scipy/scikit-learn unavailable"
        results["methods"][name] = metrics
        print(f"{name}: balance={metrics['balance']:.3f}, "
              f"cut={metrics['cut_edges']:,}/{metrics['total_edges']:,} "
              f"({metrics['edge_cut']:.3%}), "
              f"initial violations={metrics['initial_violations']:,}, "
              f"repair additions={metrics['cascade_added_blocks']:,}, "
              f"final NPV={metrics['final_npv']:,.2f}, "
              f"local solve={metrics['local_solve_runtime_s']:.1f}s, "
              f"repair={metrics['refinement_runtime_s']:.1f}s")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/BauxiteMed/bauxitemed.dat"))
    parser.add_argument("--pit", type=Path, default=Path("results/bauxitemed_baseline_pit.npy"))
    parser.add_argument("--output", type=Path, default=Path("results/experiment_e_partitioning.json"))
    parser.add_argument("--partitions", type=int, default=9)
    parser.add_argument(
        "--methods", nargs="+",
        choices=("bench_spatial", "spectral", "metis",
                 "hybrid_spatial_graph", "hybrid_spatial_graph_v4_1"),
        help="Optional subset; use this to run a paired baseline/hybrid comparison.",
    )
    args = parser.parse_args()
    run(args.data, args.pit, args.output, args.partitions, args.methods)


if __name__ == "__main__":
    main()






