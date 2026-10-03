"""Build thesis reporting artifacts from recorded runs; starts no optimizer."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import platform
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATASETS = {
    "bauxitemed": "data/BauxiteMed/bauxitemed.dat",
    "cucase": "data/BauxiteMed/cucase.dat",
}


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> tuple[list[dict], dict]:
    e_bauxite = read_json("experiment_e_hybrid_v4_1_bauxitemed.json")
    spatial = e_bauxite["methods"]["bench_spatial"]
    records = []
    for dataset, partition_file in (
        ("bauxitemed", "experiment_boundary_refinement_bauxitemed_spatial.json"),
        ("cucase", "experiment_boundary_refinement_cucase.json"),
    ):
        part = read_json(partition_file)
        global_run = read_json(f"global_solver_benchmark_{dataset}.json")
        source = ROOT / DATASETS[dataset]
        if not source.is_file():
            raise FileNotFoundError(source)
        exact_npv = float(global_run["hpf_npv"])
        part_npv = float(part["final_npv"])
        quality = spatial if dataset == "bauxitemed" else part.get("partition_quality", {})
        records.append({
            "dataset": dataset,
            "nodes": int(global_run["graph_nodes"]),
            "precedence_arcs": int(global_run["precedence_arcs"]),
            "exact_solver": "global HPF (LiteralLG agrees exactly)",
            "exact_npv": exact_npv,
            "exact_violations": int(global_run["hpf_violations"]),
            "lg_npv": float(global_run["lg_npv"]),
            "lg_runtime_s": float(global_run["lg_runtime_s"]),
            "hpf_runtime_s": float(global_run["hpf_runtime_s"]),
            "partition_method": part["partition_method"],
            "partitions": int(part["partitions_requested"]),
            "balance_max_over_mean": quality.get("balance", ""),
            "cut_fraction": quality.get("edge_cut", ""),
            "cut_arcs": quality.get("cut_edges", ""),
            "initial_violations": int(part["initial_violations"]),
            "violation_clusters": int(part["violation_cluster_count"]),
            "cascade_added_blocks": int(part["cascade_added_blocks"]),
            "cascade_depth": int(part["cascade_depth"]),
            "add_only_npv": float(part["boundary_seed_npv"]),
            "boundary_npv_gain": float(part["boundary_npv_gain"]),
            "refinement_runtime_s": float(part["refinement_runtime_s"]),
            "partition_runtime_s": float(part["partition_runtime_s"]),
            "local_solve_runtime_s": float(part["local_solve_runtime_s"]),
            "partition_total_runtime_s": float(part["total_runtime_s"]),
            "final_violations": int(part["final_violations"]),
            "partition_final_npv": part_npv,
            "npv_gap_percent": (part_npv / exact_npv - 1.0) * 100.0,
            "source_file": DATASETS[dataset],
            "source_sha256_at_report_generation": sha256(source),
            "partition_result_file": f"results/{partition_file}",
            "global_result_file": f"results/global_solver_benchmark_{dataset}.json",
        })

    pit_path = RESULTS / "bauxitemed_baseline_pit.npy"
    pit_bytes = pit_path.read_bytes()
    if pit_bytes[:6] != b"\x93NUMPY" or pit_bytes[6] != 1:
        raise ValueError("Expected a NumPy v1 pit-membership file")
    header_size = struct.unpack_from("<H", pit_bytes, 8)[0]
    header = ast_literal(pit_bytes[10:10 + header_size].decode("latin1").strip())
    if header.get("descr") != "<f8" or header.get("fortran_order"):
        raise ValueError(f"Unsupported pit array encoding: {header}")
    pit_values = [item[0] for item in struct.iter_unpack("<d", pit_bytes[10 + header_size:])]
    econ = [float(x) for x in (ROOT / DATASETS["bauxitemed"]).read_text(encoding="ascii").split()]
    if len(pit_values) != 120 * 120 * 26 or len(econ) != len(pit_values):
        raise ValueError("Bauxitemed pit/data length does not match its declared dimensions")
    pit_npv = sum(value for selected, value in zip(pit_values, econ) if selected != 0)
    selected_count = sum(selected != 0 for selected in pit_values)
    exact = read_json("global_solver_benchmark_bauxitemed.json")
    if selected_count != exact["hpf_blocks"] or pit_npv != exact["hpf_npv"]:
        raise ValueError("Saved pit overlay does not match the exact bauxitemed reference")

    manifest = {
        "purpose": "reproducibility and provenance for a report generated from recorded runs",
        "historical_experiment_environment": {
            "platform": "Ubuntu (recorded in experiments/EXPERIMENT_E.md)",
            "python_version": "not recorded in historical result artifacts",
            "optional_packages_recorded_for_full_experiment_e": {
                "scipy": "1.18.1", "scikit_learn": "1.9.1", "pymetis": "2025.2.2"
            },
            "cpu_and_physical_memory": "not recorded; peak process RSS is available in global benchmark JSON",
            "hashes_captured_at_original_run": False,
            "hash_note": "SHA-256 values below were computed from current source files during report generation.",
        },
        "report_builder_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "inputs": [
            {"path": r["source_file"], "sha256_current": r["source_sha256_at_report_generation"]}
            for r in records
        ],
        "visualization_pit": {
            "path": "results/bauxitemed_baseline_pit.npy",
            "sha256_current": sha256(pit_path),
            "selected_blocks": selected_count,
            "npv": pit_npv,
            "matches_exact_global_result": True,
        },
        "source_run_artifacts": sorted({
            item for r in records
            for item in (r["partition_result_file"], r["global_result_file"])
        } | {"results/experiment_e_hybrid_v4_1_bauxitemed.json"}),
    }
    return records, manifest


def ast_literal(value: str) -> dict:
    """Parse the small dict literal in a NumPy array header without NumPy."""
    import ast
    result = ast.literal_eval(value)
    if not isinstance(result, dict):
        raise ValueError("NumPy header is not a dictionary")
    return result


def write_csv(records: list[dict]) -> None:
    with (RESULTS / "thesis_additions_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def write_figure(records: list[dict]) -> None:
    """Make a dependency-free NPV/reference and feasibility SVG."""
    left, bar_width = 300, 600
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1060" height="500" viewBox="0 0 1060 500">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:23px;font-weight:700}.sub{font-size:13px;fill:#4b5563}.label{font-size:15px;font-weight:600}.small{font-size:13px}.axis{stroke:#9ca3af;stroke-width:1}.exact{fill:#2563eb}.part{fill:#f59e0b}</style>',
        '<text x="34" y="40" class="title">Partitioned solutions against exact global references</text>',
        '<text x="34" y="64" class="sub">Boundary-refined feasible NPV; bars are normalized within each deposit</text>',
        '<rect x="710" y="78" width="13" height="13" class="exact"/><text x="731" y="89" class="small">Global HPF exact reference</text>',
        '<rect x="710" y="100" width="13" height="13" class="part"/><text x="731" y="111" class="small">Partition + boundary refinement</text>',
    ]
    for idx, row in enumerate(records):
        y = 158 + idx * 155
        ratio = max(0.0, min(1.0, row["partition_final_npv"] / row["exact_npv"]))
        pieces.extend([
            f'<text x="34" y="{y+8}" class="label">{html.escape(row["dataset"])}</text>',
            f'<text x="34" y="{y+34}" class="small">{row["nodes"]:,} nodes · {row["partition_method"]}, {row["partitions"]} parts</text>',
            f'<rect x="{left}" y="{y-6}" width="{bar_width}" height="30" rx="4" class="exact"/>',
            f'<text x="{left+bar_width+12}" y="{y+14}" class="small">${row["exact_npv"]/1e6:.3f}M · 0 violations</text>',
            f'<rect x="{left}" y="{y+34}" width="{bar_width*ratio:.2f}" height="30" rx="4" class="part"/>',
            f'<text x="{left+bar_width*ratio+12:.2f}" y="{y+55}" class="small">${row["partition_final_npv"]/1e6:.3f}M · {row["final_violations"]} violations</text>',
            f'<text x="{left}" y="{y+89}" class="small">Initial violations {row["initial_violations"]:,} → {row["final_violations"]}; '
            f'{row["violation_clusters"]:,} clusters; cascade {row["cascade_depth"]} levels; '
            f'boundary gain ${row["boundary_npv_gain"]/1e3:,.1f}k in {row["refinement_runtime_s"]:.1f}s; '
            f'gap {row["npv_gap_percent"]:.2f}%</text>',
        ])
    pieces.extend([
        f'<line x1="{left}" y1="450" x2="{left+bar_width}" y2="450" class="axis"/>',
        f'<text x="{left}" y="474" class="small">0</text>',
        f'<text x="{left+bar_width-100}" y="474" class="small">100% of exact NPV</text>',
        '<text x="34" y="492" class="sub">Recorded project JSON results; runtimes should not be compared across machines.</text>',
        '</svg>',
    ])
    (RESULTS / "thesis_additions_npv_comparison.svg").write_text("\n".join(pieces), encoding="utf-8")


def write_partition_map() -> None:
    """Plot the actual 9-way spatial grid and exact-pit membership at one bench."""
    path = RESULTS / "bauxitemed_baseline_pit.npy"
    data = path.read_bytes()
    header_size = struct.unpack_from("<H", data, 8)[0]
    header = ast_literal(data[10:10 + header_size].decode("latin1").strip())
    if header.get("descr") != "<f8" or header.get("shape") != (120 * 120 * 26,):
        raise ValueError("Unexpected bauxitemed pit array layout")
    flat = [item[0] for item in struct.iter_unpack("<d", data[10 + header_size:])]
    per_bench = 120 * 120
    counts = [sum(flat[z * per_bench:(z + 1) * per_bench]) for z in range(26)]
    bench = max(range(26), key=counts.__getitem__)
    colors = ["#bfdbfe", "#a7f3d0", "#fde68a", "#fbcfe8", "#c4b5fd", "#fed7aa", "#a5f3fc", "#d9f99d", "#fecaca"]
    cell, x0, y0 = 4, 90, 116
    size = 120 * cell
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="700" viewBox="0 0 900 700">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#4b5563}.small{font-size:12px}.part{font-size:14px;font-weight:600}.grid{stroke:#475569;stroke-width:.35;opacity:.32}.boundary{stroke:#111827;stroke-width:2}</style>',
        '<text x="34" y="38" class="title">Bauxitemed: spatial partitions and exact pit membership</text>',
        f'<text x="34" y="63" class="sub">9-way spatial partition · bench z={bench} (zero-based) has the most selected blocks</text>',
        '<text x="34" y="85" class="sub">Pastel cells show partition; dark cells are blocks selected by the exact global pit</text>',
    ]
    # Spatial partition code uses a 3-by-3 grid over x,y in [0,120).
    for py in range(3):
        for px in range(3):
            part_id = py * 3 + px
            svg.append(f'<rect x="{x0+px*40*cell}" y="{y0+py*40*cell}" width="{40*cell}" height="{40*cell}" fill="{colors[part_id]}"/>')
            svg.append(f'<text x="{x0+(px*40+20)*cell-12}" y="{y0+(py*40+20)*cell+5}" class="part">P{part_id}</text>')
    selected = flat[bench * per_bench:(bench + 1) * per_bench]
    for iy, value in enumerate(selected):
        if value != 0:
            x, y = iy % 120, iy // 120
            svg.append(f'<rect x="{x0+x*cell}" y="{y0+y*cell}" width="{cell}" height="{cell}" fill="#111827"/>')
    for tick in range(0, 121, 10):
        coord = x0 + tick * cell
        svg.append(f'<line x1="{coord}" y1="{y0}" x2="{coord}" y2="{y0+size}" class="grid"/>')
        svg.append(f'<line x1="{x0}" y1="{y0+tick*cell}" x2="{x0+size}" y2="{y0+tick*cell}" class="grid"/>')
    for tick in (40, 80):
        coord = x0 + tick * cell
        svg.append(f'<line x1="{coord}" y1="{y0}" x2="{coord}" y2="{y0+size}" class="boundary"/>')
        svg.append(f'<line x1="{x0}" y1="{y0+tick*cell}" x2="{x0+size}" y2="{y0+tick*cell}" class="boundary"/>')
    svg.extend([
        f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="none" class="boundary"/>',
        f'<text x="{x0+size/2-25}" y="{y0+size+24}" class="small">x (block index)</text>',
        f'<text x="{x0-28}" y="{y0+size/2}" class="small" transform="rotate(-90 {x0-28} {y0+size/2})">y (block index)</text>',
        '<rect x="610" y="130" width="15" height="15" fill="#111827"/><text x="635" y="143" class="small">Exact pit block</text>',
        f'<text x="610" y="174" class="small">Selected blocks on bench: {int(counts[bench]):,}</text>',
        '<text x="610" y="202" class="small">Boundaries are the actual spatial grid cuts</text>',
        '<text x="34" y="648" class="sub">Membership overlay verified against the global HPF result: 73,419 blocks; NPV 29,690,715.</text>',
        '</svg>',
    ])
    (RESULTS / "thesis_additions_bauxitemed_partition_map.svg").write_text("\n".join(svg), encoding="utf-8")


def write_report(records: list[dict], manifest: dict) -> None:
    rows = []
    for r in records:
        rows.append(
            f"| {r['dataset']} | {r['nodes']:,} | {r['partition_method']} ({r['partitions']}) | "
            f"{r['initial_violations']:,} / {r['violation_clusters']:,} | "
            f"{r['cascade_added_blocks']:,} / {r['cascade_depth']} | "
            f"{r['add_only_npv']:,.0f} → {r['partition_final_npv']:,.0f} | "
            f"{r['boundary_npv_gain']:+,.0f} / {r['refinement_runtime_s']:.2f} | "
            f"{r['final_violations']} | {r['exact_npv']:,.0f} | {r['npv_gap_percent']:.2f}% |"
        )
    lines = [
        "# Consolidated thesis additions: exact comparison, refinement, provenance, and figure", "",
        "This report consolidates existing experiment outputs; it does not rerun the optimizers. NPV is the input EPV objective used by the closure model, not discounted mine-life cash flow.", "",
        "## Exact global reference and partitioned feasible result", "",
        "Global HPF is the exact maximum-closure reference for each dataset. Direct LiteralLG agrees in selected membership and NPV with zero violations. Partitioned results are feasible after boundary refinement but need not be globally optimal.", "",
        "| Dataset | Nodes | Partition method (k) | Initial violations / clusters | Cascade blocks / depth | Add-only → boundary NPV | Boundary gain / runtime (s) | Final violations | Exact global NPV | Gap to exact |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows, "",
        "![Exact and partitioned NPV comparison](../results/thesis_additions_npv_comparison.svg)", "",
        "![Actual bauxitemed spatial partition with exact pit membership](../results/thesis_additions_bauxitemed_partition_map.svg)", "",
        "The map shows the actual 3×3 spatial partition boundaries and the global exact pit selection on the bench with the most selected blocks. Its membership and NPV are checked against the full-model HPF JSON before generation.", "",
        "![3D bauxitemed block-grid view](../results/bauxitemed_3d_block_model.png)", "",
        "The 3D view uses the exact global pit membership and the Experiment E spatial partition grid. It reconstructs x/y/z block indices from the 120×120×26 input order; z is bench index, not surveyed topography. Regenerate it with `python experiments/build_bauxitemed_3d_block_model.py` (requires Pillow).", "",
        "The NPV gap is `(partition final NPV / exact global NPV - 1) × 100`; negative values mean the feasible partitioned result is below the exact reference. Bars are independently scaled within each deposit.", "",
        "## Partition quality measurements", "",
        "For full bauxitemed Experiment E, the 9-way spatial partition has max/mean size balance 1.557 and 18,800 cut arcs of 1,130,500 (1.663%). Hybrid V4.1 made zero moves and has the same partition metrics. The selected cucase 9-way METIS boundary result does not save balance/cut counts; these are left unavailable rather than inferred from a different partition count or run.", "",
        "## Reproducibility record", "",
        "- The experiment notes identify Ubuntu and record SciPy 1.18.1, scikit-learn 1.9.1, and pymetis 2025.2.2 for the full genuine-method Experiment E run.",
        "- Current input paths and SHA-256 values are in `results/thesis_additions_provenance.json`. The hashes were computed while generating this report; original runs did not record them.",
        "- Historical artifacts do not state the Python version, CPU model, or physical memory. Global benchmark JSON records whole-process peak RSS, not solver-specific memory. A fully specified hardware benchmark requires capturing those fields on a rerun.",
        "- Regenerate the comparison, report, checksums, and 2D SVGs from existing results with `python experiments/build_thesis_additions.py`; regenerate the 3D PNG with `python experiments/build_bauxitemed_3d_block_model.py` (Pillow required). Neither command launches an optimizer.", "",
        "## Source artifacts", "",
    ]
    lines.extend(f"- `{name}`" for name in manifest["source_run_artifacts"])
    lines.extend([
        "", "## Interpretation", "",
        "Boundary refinement improves the add-only seed on both deposits and restores zero final precedence violations. The bauxitemed spatial result remains about 2.98% below the exact global objective; cucase METIS remains about 1.77% below. These results show feasibility restoration and local objective improvement, not global-optimality guarantees or a speed advantage.", "",
    ])
    (ROOT / "experiments" / "THESIS_ADDITIONS.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "thesis_additions_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    records, manifest = collect()
    write_csv(records)
    write_figure(records)
    write_partition_map()
    write_report(records, manifest)
    print("Wrote experiments/THESIS_ADDITIONS.md")
    print("Wrote results/thesis_additions_comparison.csv")
    print("Wrote results/thesis_additions_npv_comparison.svg")
    print("Wrote results/thesis_additions_bauxitemed_partition_map.svg")
    print("Wrote results/thesis_additions_provenance.json")
    for row in records:
        print(f"{row['dataset']}: exact={row['exact_npv']:,.0f}; partition={row['partition_final_npv']:,.0f}; gap={row['npv_gap_percent']:.3f}%; final violations={row['final_violations']}")


if __name__ == "__main__":
    main()
