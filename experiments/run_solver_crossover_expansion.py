"""Add larger paired LG/HPF timing samples across the four 3D deposits.

Existing training data and models are preserved. New samples use a separate
directory, then a separate grouped model is fitted from old plus new rows.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATASETS = ("bauxitemed", "cucase", "cupipe", "mclaughlingeo")
BASE_RESULTS = ROOT / "results"
EXPANSION = BASE_RESULTS / "solver_crossover_expansion_20260929_01"
RAW_DIR = EXPANSION / "raw"
FIT_DIR = EXPANSION / "fit"
INDUSTRIAL_FIT_DIR = EXPANSION / "fit_industrial_only"


def _merge_csvs(paths: list[Path], output: Path) -> int:
    rows = []
    fieldnames = None
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            if reader.fieldnames != fieldnames:
                raise ValueError(f"CSV schema mismatch in {path}")
            rows.extend(reader)
    if not rows or not fieldnames:
        raise ValueError(f"No solver timing rows available for {output.name}")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(samples_per_size: int = 4, sizes: tuple[int, ...] = (8, 9, 10),
        max_lg_nodes: int = 4096, lg_timeout_s: float = 5.0) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        raw_csv = RAW_DIR / f"adaptive_solver_crossover_{dataset}.csv"
        raw_json = RAW_DIR / f"adaptive_solver_training_{dataset}.json"
        if raw_csv.exists() and raw_json.exists():
            print(f"Reuse completed timing samples: {dataset}", flush=True)
            continue
        data = ROOT.parent / "lg" / f"{dataset}.dat"
        command = [
            sys.executable,
            str(ROOT / "experiments" / "benchmark_solver_selector.py"),
            "--dataset", dataset,
            "--data", str(data),
            "--outdir", str(RAW_DIR),
            "--samples-per-size", str(samples_per_size),
            "--sizes", *(str(size) for size in sizes),
            "--max-lg-nodes", str(max_lg_nodes),
            "--lg-timeout-s", str(lg_timeout_s),
        ]
        print(f"\n=== LG/HPF crossover: {dataset} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    for dataset in DATASETS + ("sim2d76",):
        source = BASE_RESULTS / f"adaptive_solver_training_{dataset}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing original training dataset: {source}")
        shutil.copy2(source, FIT_DIR / source.name)
    counts = {}
    for dataset in DATASETS + ("sim2d76",):
        old_path = BASE_RESULTS / f"adaptive_solver_crossover_{dataset}.csv"
        new_path = RAW_DIR / f"adaptive_solver_crossover_{dataset}.csv"
        paths = [old_path, new_path] if dataset in DATASETS else [old_path]
        counts[dataset] = _merge_csvs(
            paths, FIT_DIR / f"adaptive_solver_crossover_{dataset}.csv"
        )

    from experiments.train_multidataset_selector import run as train_selector

    metrics = train_selector(FIT_DIR)
    INDUSTRIAL_FIT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        for prefix in ("adaptive_solver_training_", "adaptive_solver_crossover_"):
            source = FIT_DIR / f"{prefix}{dataset}.csv"
            if source.exists():
                shutil.copy2(source, INDUSTRIAL_FIT_DIR / source.name)
    industrial_metrics = train_selector(INDUSTRIAL_FIT_DIR, datasets=DATASETS)
    result = {
        "datasets_with_new_samples": list(DATASETS),
        "new_patch_sizes": list(sizes),
        "samples_per_size": samples_per_size,
        "max_lg_nodes": max_lg_nodes,
        "lg_timeout_s": lg_timeout_s,
        "combined_crossover_rows_by_dataset": counts,
        "training_metrics": metrics,
        "industrial_only_training_metrics": industrial_metrics,
        "raw_results_directory": str(RAW_DIR),
        "separate_model_and_metrics_directory": str(FIT_DIR),
        "industrial_only_model_and_metrics_directory": str(INDUSTRIAL_FIT_DIR),
    }
    summary = EXPANSION / "expansion_summary.json"
    summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {summary.relative_to(ROOT)}", flush=True)
    return result


if __name__ == "__main__":
    run()
