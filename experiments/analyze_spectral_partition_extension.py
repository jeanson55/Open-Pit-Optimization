"""Compare matched k=4 spectral outcomes and a deposit-held-out baseline."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("bauxitemed", "cucase", "cupipe", "mclaughlingeo")
METHODS = ("spatial", "metis", "spectral")


def _load(dataset: str, method: str) -> dict:
    path = ROOT / "results" / f"partition_grid_{dataset}_{method}_k4_add.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict:
    runs = {d: {m: _load(d, m) for m in METHODS} for d in DATASETS}
    winners = {}
    by_dataset = []
    for dataset, candidates in runs.items():
        npvs = {method: float(result["final_npv"])
                for method, result in candidates.items()}
        best = max(npvs.values())
        winners[dataset] = [method for method, value in npvs.items() if value == best]
        by_dataset.append({
            "dataset": dataset,
            "best_methods": winners[dataset],
            "best_npv": best,
            "methods": {
                method: {
                    "npv": npvs[method],
                    "relative_npv_regret": (best - npvs[method]) / max(abs(best), 1.0),
                    "partition_seconds": float(result["partition_runtime_s"]),
                    "local_solve_seconds": float(result["local_solve_runtime_s"]),
                    "refinement_seconds": float(result["refinement_runtime_s"]),
                    "total_reported_seconds": float(result["partition_runtime_s"])
                    + float(result["total_runtime_s"]),
                    "cut_fraction": float(result["partition_quality"]["edge_cut"]),
                    "balance_max_over_mean": float(result["partition_quality"]["balance"]),
                    "initial_violations": int(result["initial_violations"]),
                    "cascade_added_blocks": int(result["cascade_added_blocks"]),
                    "cascade_depth": int(result["cascade_depth"]),
                    "final_violations": int(result["final_violations"]),
                }
                for method, result in candidates.items()
            },
        })

    # A grouped, transparent selector baseline: choose the most frequent
    # winning method among the three training deposits. No model is fitted to
    # only four independent groups.
    held_out = []
    for test_dataset in DATASETS:
        train_counts = Counter(
            method for dataset in DATASETS if dataset != test_dataset
            for method in winners[dataset]
        )
        max_count = max(train_counts.values())
        tied = sorted(method for method, count in train_counts.items() if count == max_count)
        selected = tied[0]
        best = max(float(runs[test_dataset][m]["final_npv"]) for m in METHODS)
        selected_npv = float(runs[test_dataset][selected]["final_npv"])
        held_out.append({
            "held_out_dataset": test_dataset,
            "training_winner_counts": dict(train_counts),
            "selected_method": selected,
            "observed_best_methods": winners[test_dataset],
            "correct": selected in winners[test_dataset],
            "relative_npv_regret": (best - selected_npv) / max(abs(best), 1.0),
        })

    baselines = {}
    for method in METHODS:
        regrets = [row["methods"][method]["relative_npv_regret"] for row in by_dataset]
        baselines[f"always_{method}"] = {
            "mean_relative_npv_regret": sum(regrets) / len(regrets),
            "wins": sum(method in winners[d] for d in DATASETS),
        }
    result = {
        "partitions": 4,
        "datasets": list(DATASETS),
        "independent_dataset_groups": len(DATASETS),
        "per_deposit_results": by_dataset,
        "leave_one_dataset_out_winner_majority_baseline": {
            "accuracy": sum(row["correct"] for row in held_out) / len(held_out),
            "mean_relative_npv_regret": sum(row["relative_npv_regret"] for row in held_out) / len(held_out),
            "holdouts": held_out,
        },
        "fixed_method_baselines": baselines,
        "limitations": [
            "Only four independent deposits and one matched partition count are included; this is grouped baseline analysis, not ML training.",
            "Spectral uses approximate LOBPCG embeddings; three large-graph runs emitted convergence-tolerance warnings in the console, whose residual vectors are not stored in the JSON metrics.",
            "Reported total is partition runtime plus adaptive-solver total runtime; graph-load time is not included.",
        ],
    }
    path = ROOT / "results" / "spectral_partition_extension_comparison.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {path.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    run()
