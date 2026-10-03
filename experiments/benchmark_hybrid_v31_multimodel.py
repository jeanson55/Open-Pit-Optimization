"""
Four-model partition benchmark for Hybrid V3.1.

Compares:
    1. Spatial partitioning
    2. METIS partitioning
    3. Hybrid V3.1 partition selection

Models:
    - bauxitemed
    - cucase
    - cupipe
    - mclaughlingeo

The graph construction follows the established gPit benchmark construction
using Pseudoflow + IA1x5 precedence arcs.

Run from WSL with:

PYTHONPATH=/home/jeanson/pit-exp-deps:../lg:. \
../lg/.venv-linux/bin/python \
experiments/benchmark_hybrid_v31_multimodel.py
"""

import time

import networkx as nx
import numpy as np
import pandas as pd

from gpit import precedence as prec
from gpit.pseudoflow import Pseudoflow

from partition.hybrid import hybrid_partition
from partition.metis import metis_partition
from partition.spatial import spatial_partition


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS = {
    "bauxitemed": (120, 120, 26),
    "cucase": (70, 70, 30),
    "cupipe": (100, 100, 50),
    "mclaughlingeo": (90, 90, 50),
}

BASE = "../lg"
N_PARTITIONS = 8


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(path, nx_, ny_, nz_):
    """
    Build the authoritative gPit precedence graph for a real dataset.

    The construction matches the existing standalone gPit benchmark:
        - Pseudoflow external arcs
        - IA1x5 precedence arcs
        - removal of source/sink/non-real nodes
        - original .dat flat-index value mapping
    """

    print(f"  Loading data: {path}")

    flat = np.loadtxt(path, dtype=np.float64)

    print(f"  Grid cells: {len(flat):,}")

    df = pd.DataFrame(
        {
            "EPV": flat
        }
    )

    bmparms = {
        "nx": nx_,
        "ny": ny_,
        "nz": nz_,
        "xorg": 0,
        "yorg": 0,
        "zorg": 0,
        "xs": 1,
        "ys": 1,
        "zs": 1,
        "volume": 1,
    }

    g_full = nx.DiGraph()

    # -----------------------------------------------------------------------
    # External arcs
    # -----------------------------------------------------------------------

    ext = Pseudoflow(df, bmparms)

    ext.EPV = np.asarray(df["EPV"])

    g_full = ext.CreateExtArcs(g_full)

    # -----------------------------------------------------------------------
    # IA1x5 precedence arcs
    # -----------------------------------------------------------------------

    for ind_z in range(nz_ - 1):

        pos_z = nz_ - ind_z - 2

        for pos_y in range(
            ind_z + 1,
            ny_ - ind_z - 1,
        ):

            for pos_x in range(
                ind_z + 1,
                nx_ - ind_z - 1,
            ):

                g_full = prec.IA1x5(
                    pos_x,
                    pos_y,
                    pos_z,
                    nx_,
                    ny_,
                    g_full,
                )

    # -----------------------------------------------------------------------
    # Identify source/sink
    # -----------------------------------------------------------------------

    source = 0
    sink = nx_ * ny_ * nz_ + 1

    # -----------------------------------------------------------------------
    # Keep only real block nodes
    # -----------------------------------------------------------------------

    real_nodes = [
        node
        for node in g_full.nodes
        if node != source
        and node != sink
        and g_full.in_degree(node) > 0
        and g_full.out_degree(node) > 0
    ]

    g = g_full.subgraph(real_nodes).copy()

    # -----------------------------------------------------------------------
    # Add block attributes
    #
    # The node ID is the original flat-array index + 1.
    # -----------------------------------------------------------------------

    for node in g.nodes:

        idx = int(node) - 1

        g.nodes[node]["value"] = float(flat[idx])

        g.nodes[node]["x"] = idx % nx_

        g.nodes[node]["y"] = (
            (idx // nx_) % ny_
        )

        g.nodes[node]["z"] = (
            idx // (nx_ * ny_)
        )

    return g


# ---------------------------------------------------------------------------
# Partition statistics
# ---------------------------------------------------------------------------

def partition_stats(graph, labels):
    """
    Calculate basic partition-quality metrics.
    """

    partitions = sorted(
        set(labels.values())
    )

    sizes = {
        partition: 0
        for partition in partitions
    }

    values = {
        partition: 0.0
        for partition in partitions
    }

    # -----------------------------------------------------------------------
    # Count nodes and economic value
    # -----------------------------------------------------------------------

    for node, partition in labels.items():

        sizes[partition] += 1

        values[partition] += float(
            graph.nodes[node]["value"]
        )

    # -----------------------------------------------------------------------
    # Node-count balance
    # -----------------------------------------------------------------------

    min_size = min(sizes.values())

    max_size = max(sizes.values())

    balance = (
        max_size / min_size
        if min_size > 0
        else float("inf")
    )

    # -----------------------------------------------------------------------
    # Precedence edge cut
    # -----------------------------------------------------------------------

    cut = 0

    for u, v in graph.edges:

        if labels[u] != labels[v]:

            cut += 1

    return {
        "partitions": len(partitions),
        "min_size": min_size,
        "max_size": max_size,
        "balance": balance,
        "cut": cut,
        "cut_percent": (
            100.0 * cut / graph.number_of_edges()
            if graph.number_of_edges() > 0
            else 0.0
        ),
        "min_value": min(values.values()),
        "max_value": max(values.values()),
    }


# ---------------------------------------------------------------------------
# Print partition statistics
# ---------------------------------------------------------------------------

def print_partition_result(
    method_name,
    runtime,
    graph,
    labels,
):
    """
    Print one partition result.
    """

    stats = partition_stats(
        graph,
        labels,
    )

    print(
        f"{method_name:14s}"
        f" runtime={runtime:8.3f}s"
        f"  partitions={stats['partitions']:2d}"
        f"  size={stats['min_size']:,}"
        f"-{stats['max_size']:,}"
        f"  balance={stats['balance']:.6f}"
        f"  cut={stats['cut']:,}"
        f"  cut%={stats['cut_percent']:.4f}%"
    )

    return stats


# ---------------------------------------------------------------------------
# Determine which seed Hybrid selected
# ---------------------------------------------------------------------------

def identify_hybrid_seed(
    hybrid_labels,
    spatial_labels,
    metis_labels,
):
    """
    Determine whether Hybrid V3.1 selected the Spatial
    or METIS labeling.

    Hybrid V3.1 currently returns one of the candidate
    seed labelings.
    """

    if hybrid_labels == spatial_labels:

        return "SPATIAL"

    if hybrid_labels == metis_labels:

        return "METIS"

    return "OTHER"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():

    print()
    print("=" * 100)
    print("HYBRID V3.1 — FOUR-MODEL PARTITION BENCHMARK")
    print("=" * 100)

    all_results = {}

    # =======================================================================
    # Process each real model
    # =======================================================================

    for model_name, dimensions in MODELS.items():

        nx_, ny_, nz_ = dimensions

        print()
        print("=" * 100)
        print(
            f"{model_name.upper()} "
            f"({nx_} x {ny_} x {nz_})"
        )
        print("=" * 100)

        path = f"{BASE}/{model_name}.dat"

        # -------------------------------------------------------------------
        # Build graph
        # -------------------------------------------------------------------

        print()
        print("GRAPH CONSTRUCTION")

        start = time.perf_counter()

        graph = build_graph(
            path,
            nx_,
            ny_,
            nz_,
        )

        build_runtime = (
            time.perf_counter() - start
        )

        print(
            f"  Graph build time : "
            f"{build_runtime:.3f} s"
        )

        print(
            f"  Real nodes       : "
            f"{graph.number_of_nodes():,}"
        )

        print(
            f"  Precedence edges : "
            f"{graph.number_of_edges():,}"
        )

        # -------------------------------------------------------------------
        # Spatial
        # -------------------------------------------------------------------

        print()
        print("PARTITIONING")

        start = time.perf_counter()

        spatial_labels = spatial_partition(
            graph,
            N_PARTITIONS,
        )

        spatial_runtime = (
            time.perf_counter() - start
        )

        spatial_stats = print_partition_result(
            "Spatial",
            spatial_runtime,
            graph,
            spatial_labels,
        )

        # -------------------------------------------------------------------
        # METIS
        # -------------------------------------------------------------------

        start = time.perf_counter()

        metis_labels = metis_partition(
            graph,
            N_PARTITIONS,
        )

        metis_runtime = (
            time.perf_counter() - start
        )

        metis_stats = print_partition_result(
            "METIS",
            metis_runtime,
            graph,
            metis_labels,
        )

        # -------------------------------------------------------------------
        # Hybrid V3.1
        # -------------------------------------------------------------------

        start = time.perf_counter()

        hybrid_labels = hybrid_partition(
            graph,
            N_PARTITIONS,
        )

        hybrid_runtime = (
            time.perf_counter() - start
        )

        hybrid_stats = print_partition_result(
            "Hybrid V3.1",
            hybrid_runtime,
            graph,
            hybrid_labels,
        )

        # -------------------------------------------------------------------
        # Identify Hybrid selection
        # -------------------------------------------------------------------

        selected_seed = identify_hybrid_seed(
            hybrid_labels,
            spatial_labels,
            metis_labels,
        )

        # -------------------------------------------------------------------
        # Number of changed assignments
        # -------------------------------------------------------------------

        spatial_changes = sum(
            1
            for node in graph.nodes
            if hybrid_labels[node]
            != spatial_labels[node]
        )

        metis_changes = sum(
            1
            for node in graph.nodes
            if hybrid_labels[node]
            != metis_labels[node]
        )

        print()
        print("HYBRID SELECTION")
        print(
            f"  Selected seed       : "
            f"{selected_seed}"
        )

        print(
            f"  Differences vs Spatial : "
            f"{spatial_changes:,}"
        )

        print(
            f"  Differences vs METIS   : "
            f"{metis_changes:,}"
        )

        # -------------------------------------------------------------------
        # Store results
        # -------------------------------------------------------------------

        all_results[model_name] = {
            "dimensions": dimensions,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "build_runtime": build_runtime,

            "spatial": {
                "runtime": spatial_runtime,
                **spatial_stats,
            },

            "metis": {
                "runtime": metis_runtime,
                **metis_stats,
            },

            "hybrid": {
                "runtime": hybrid_runtime,
                **hybrid_stats,
            },

            "selected_seed": selected_seed,

            "hybrid_vs_spatial_changes": (
                spatial_changes
            ),

            "hybrid_vs_metis_changes": (
                metis_changes
            ),
        }

    # =======================================================================
    # Final summary
    # =======================================================================

    print()
    print()
    print("=" * 110)
    print("FINAL FOUR-MODEL SUMMARY")
    print("=" * 110)

    print()

    header = (
        f"{'Model':18s}"
        f"{'Hybrid Seed':14s}"
        f"{'Spatial Bal.':14s}"
        f"{'METIS Bal.':14s}"
        f"{'Hybrid Bal.':14s}"
        f"{'Spatial Cut':14s}"
        f"{'METIS Cut':14s}"
        f"{'Hybrid Cut':14s}"
    )

    print(header)
    print("-" * len(header))

    for model_name, result in all_results.items():

        print(
            f"{model_name:18s}"
            f"{result['selected_seed']:14s}"
            f"{result['spatial']['balance']:14.4f}"
            f"{result['metis']['balance']:14.4f}"
            f"{result['hybrid']['balance']:14.4f}"
            f"{result['spatial']['cut']:14,}"
            f"{result['metis']['cut']:14,}"
            f"{result['hybrid']['cut']:14,}"
        )

    # =======================================================================
    # Runtime summary
    # =======================================================================

    print()
    print("=" * 110)
    print("PARTITION RUNTIME SUMMARY")
    print("=" * 110)

    header = (
        f"{'Model':18s}"
        f"{'Spatial (s)':16s}"
        f"{'METIS (s)':16s}"
        f"{'Hybrid V3.1 (s)':18s}"
    )

    print(header)
    print("-" * len(header))

    for model_name, result in all_results.items():

        print(
            f"{model_name:18s}"
            f"{result['spatial']['runtime']:16.3f}"
            f"{result['metis']['runtime']:16.3f}"
            f"{result['hybrid']['runtime']:18.3f}"
        )

    # =======================================================================
    # Selection summary
    # =======================================================================

    print()
    print("=" * 110)
    print("HYBRID V3.1 SELECTION SUMMARY")
    print("=" * 110)

    spatial_count = 0
    metis_count = 0
    other_count = 0

    for model_name, result in all_results.items():

        seed = result["selected_seed"]

        print(
            f"  {model_name:18s}: "
            f"{seed}"
        )

        if seed == "SPATIAL":
            spatial_count += 1

        elif seed == "METIS":
            metis_count += 1

        else:
            other_count += 1

    print()
    print(
        f"  Spatial selections : "
        f"{spatial_count}"
    )

    print(
        f"  METIS selections  : "
        f"{metis_count}"
    )

    print(
        f"  Other selections  : "
        f"{other_count}"
    )

    # =======================================================================
    # Final
    # =======================================================================

    print()
    print("=" * 110)
    print("BENCHMARK COMPLETE")
    print("=" * 110)
    print()


if __name__ == "__main__":
    main()