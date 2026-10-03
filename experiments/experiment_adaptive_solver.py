"""Adaptive partitioned LG / Pseudoflow experiment runner.

The optimization path is unchanged by default. An optional
``--export-visualization`` flag records the actual block-level states produced
by the solver and writes a compact NPZ trace for ParaView conversion.

Supported partition methods:
    spatial, metis, spectral, hybrid_v33, hybrid_v34, hybrid_v35, hybrid_v36

Supported refinement:
    add, boundary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_DIMS = {
    "bauxitemed": (120, 120, 26),
    "cucase": (170, 215, 50),
    "cupipe": (180, 180, 85),
    "mclaughlingeo": (140, 296, 68),
    "sim2d76": (75, 1, 40),
}

DATASET_IDS = {
    "bauxitemed": 0,
    "cucase": 1,
    "cupipe": 2,
    "mclaughlingeo": 3,
    "sim2d76": 4,
}

from graph.load_graph import load_graph
from optimization.adaptive_solver import solve_adaptive
from partition.metis import metis_partition
from partition.metrics import evaluate_partition
from partition.spatial import spatial_partition
from partition.spectral import spectral_partition


def _partition_graph(graph, method: str, k: int, dataset: str):
    """Return partition labels using the existing project implementations."""
    if method == "spatial":
        return spatial_partition(graph, k)

    if method == "metis":
        return metis_partition(graph, k)

    if method == "spectral":
        try:
            import scipy  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "spectral method requires SciPy and scikit-learn; "
                "refusing to record a spatial fallback as spectral"
            ) from exc
        return spectral_partition(graph, k)

    if method == "hybrid_v33":
        from partition.hybrid_v33 import hybrid_v33_partition
        return hybrid_v33_partition(graph, k)

    if method == "hybrid_v34":
        from partition.hybrid_v34 import hybrid_v34_partition
        return hybrid_v34_partition(graph, k)

    if method == "hybrid_v35":
        from partition.hybrid_v35 import hybrid_v35_partition
        return hybrid_v35_partition(
            graph, k, DATASET_IDS.get(dataset, 0)
        )

    if method == "hybrid_v36":
        from partition.hybrid_v36 import hybrid_v36_partition
        return hybrid_v36_partition(
            graph, k, DATASET_IDS.get(dataset, 0)
        )

    raise ValueError(f"Unknown partition method: {method}")


def _node_arrays(graph, frame, nx_dim, ny_dim, nz_dim):
    """Create stable grid arrays aligned with graph node IDs."""
    count = len(frame)
    values = frame["EPV"].to_numpy(dtype=np.float64)

    node_ids = np.asarray(sorted(graph.nodes), dtype=np.int64)
    zero_based = node_ids - 1

    x = zero_based % nx_dim
    y = (zero_based // nx_dim) % ny_dim
    z = zero_based // (nx_dim * ny_dim)

    epv = values[zero_based]

    return node_ids, x.astype(np.int32), y.astype(np.int32), z.astype(np.int32), epv


def _save_visualization_trace(
    path: Path,
    graph,
    labels,
    trace,
    metrics,
    nx_dim,
    ny_dim,
    nz_dim,
):
    """Write actual solver states to a compact NPZ archive."""
    node_ids, x, y, z, epv = _node_arrays(
        graph, trace["frame"], nx_dim, ny_dim, nz_dim
    )

    node_index = {
        int(node): i
        for i, node in enumerate(node_ids)
    }

    n = len(node_ids)

    partition_id = np.full(n, -1, dtype=np.int32)
    for node, part in labels.items():
        partition_id[node_index[int(node)]] = int(part)

    def mask(nodes):
        out = np.zeros(n, dtype=np.uint8)
        for node in nodes:
            idx = node_index.get(int(node))
            if idx is not None:
                out[idx] = 1
        return out

    local_selected = mask(trace["merged"])
    repaired_selected = mask(trace["repaired"])
    final_selected = mask(trace["final_solution"])
    boundary_member = mask(
        trace.get("boundary_final", set())
    )
    boundary_added = mask(
        trace.get("boundary_reoptimized_added", set())
    )
    boundary_removed = mask(
        trace.get("boundary_reoptimized_removed", set())
    )
    repair_added = mask(
        trace.get("repair_added", set())
    )

    repair_level = np.zeros(n, dtype=np.int16)
    for node, level in trace.get(
        "repair_level_by_node", {}
    ).items():
        idx = node_index.get(int(node))
        if idx is not None:
            repair_level[idx] = int(level)

    initial_violation_nodes = set()
    for edge in trace.get("initial_violations", []):
        initial_violation_nodes.update(edge)

    initial_violation = mask(initial_violation_nodes)

    np.savez_compressed(
        path,
        node_id=node_ids,
        x=x,
        y=y,
        z=z,
        epv=epv,
        partition_id=partition_id,
        local_selected=local_selected,
        repaired_selected=repaired_selected,
        final_selected=final_selected,
        initial_violation=initial_violation,
        repair_added=repair_added,
        repair_level=repair_level,
        boundary_member=boundary_member,
        boundary_added=boundary_added,
        boundary_removed=boundary_removed,
        dimensions=np.asarray(
            [nx_dim, ny_dim, nz_dim],
            dtype=np.int32,
        ),
        initial_npv=np.asarray(
            [metrics["initial_npv"]],
            dtype=np.float64,
        ),
        final_npv=np.asarray(
            [metrics["final_npv"]],
            dtype=np.float64,
        ),
    )

    # Save JSON-safe metadata separately. Sets and graph objects are excluded.
    metadata = {
        "dataset": metrics.get("dataset"),
        "partition_method": metrics.get("partition_method"),
        "partitions_requested": metrics.get("partitions_requested"),
        "dimensions": [
            nx_dim, ny_dim, nz_dim
        ],
        "metrics": metrics,
        "visualization_arrays": [
            "node_id",
            "x",
            "y",
            "z",
            "epv",
            "partition_id",
            "local_selected",
            "repaired_selected",
            "final_selected",
            "initial_violation",
            "repair_added",
            "repair_level",
            "boundary_member",
            "boundary_added",
            "boundary_removed",
        ],
    }

    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )


def run(
    data: Path,
    output: Path,
    method: str,
    k: int,
    policy: str,
    lg_node_limit: int,
    model_path,
    dims,
    refinement: str,
    visualization_output: Path | None = None,
):
    """Run one experiment and optionally export its actual block trace."""
    nx_dim, ny_dim, nz_dim = dims

    print("Loading graph...")

    graph, frame = load_graph(
        str(data),
        nx_dim=nx_dim,
        ny_dim=ny_dim,
        nz_dim=nz_dim,
    )

    count = len(frame)

    # Remove the artificial HPF source and sink.
    graph.remove_nodes_from(
        [
            node
            for node in (0, count + 1)
            if node in graph
        ]
    )

    print(
        "Graph loaded:",
        graph.number_of_nodes(),
        "nodes,",
        graph.number_of_edges(),
        "edges",
    )

    values = frame["EPV"].to_numpy(dtype=np.float64)

    for node in graph.nodes:
        idx = int(node) - 1
        graph.nodes[node].update(
            {
                "x": float(idx % nx_dim),
                "y": float((idx // nx_dim) % ny_dim),
                "z": float(idx // (nx_dim * ny_dim)),
                "value": float(values[idx]),
            }
        )

    start = time.perf_counter()
    labels = _partition_graph(
        graph, method, k, data.stem
    )
    partition_seconds = time.perf_counter() - start

    partition_quality = evaluate_partition(
        graph, labels
    )
    partition_quality.pop("violations", None)

    model = None
    if policy == "ml":
        if model_path is None:
            raise ValueError(
                "--model required for ML policy"
            )
        import joblib
        model = joblib.load(model_path)

    need_trace = visualization_output is not None

    if need_trace:
        solution, metrics, trace = solve_adaptive(
            graph,
            labels,
            policy=policy,
            model=model,
            lg_node_limit=lg_node_limit,
            refinement=refinement,
            return_trace=True,
        )
        # The trace needs the frame for the grid export.
        trace["frame"] = frame
    else:
        solution, metrics = solve_adaptive(
            graph,
            labels,
            policy=policy,
            model=model,
            lg_node_limit=lg_node_limit,
            refinement=refinement,
        )
        trace = None

    metrics.update(
        {
            "partition_method": method,
            "partition_runtime_s": partition_seconds,
            "partitions_requested": k,
            "dataset": data.stem,
            "dimensions": [
                nx_dim, ny_dim, nz_dim
            ],
            "precedence_template": "1x5",
            "boundary_refinement_strategy": refinement,
        }
    )

    metrics["partition_quality"] = partition_quality

    if "hybrid_v36_selection" in graph.graph:
        metrics["hybrid_v36_selection"] = (
            graph.graph["hybrid_v36_selection"]
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(metrics, indent=2)
    )

    if need_trace:
        visualization_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        _save_visualization_trace(
            visualization_output,
            graph,
            labels,
            trace,
            metrics,
            nx_dim,
            ny_dim,
            nz_dim,
        )
        print(
            "Saved visualization trace:",
            visualization_output,
        )

    print(
        json.dumps(
            {
                "dataset": metrics["dataset"],
                "partition_method": method,
                "solver_counts": metrics["solver_counts"],
                "final_npv": metrics["final_npv"],
                "total_runtime_s": metrics["total_runtime_s"],
            },
            indent=2,
        )
    )

    print("Saved:", output)
    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_DIMS),
        default="bauxitemed",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--method",
        choices=(
            "spatial",
            "metis",
            "spectral",
            "hybrid_v33",
            "hybrid_v34",
            "hybrid_v35",
            "hybrid_v36",
        ),
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--policy",
        choices=("heuristic", "ml"),
        default="heuristic",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--lg-node-limit",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--refinement",
        choices=("boundary", "add"),
        default="boundary",
    )
    parser.add_argument(
        "--export-visualization",
        type=Path,
        help=(
            "Write actual block-level solver states to an NPZ "
            "archive for ParaView."
        ),
    )

    args = parser.parse_args()

    data = (
        args.data
        or ROOT / "data" / f"{args.dataset}.dat"
    )

    output = (
        args.output
        or ROOT / "results" /
        f"adaptive_{args.dataset}.json"
    )

    dims = DATASET_DIMS[args.dataset]
    method = args.method or "metis"

    run(
        data,
        output,
        method,
        args.partitions,
        args.policy,
        args.lg_node_limit,
        args.model,
        dims,
        args.refinement,
        args.export_visualization,
    )


if __name__ == "__main__":
    main()
