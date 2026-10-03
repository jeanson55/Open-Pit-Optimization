"""Fit and leave-one-block-model-out validate an LG/HPF selector."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from optimization.adaptive_solver import FEATURE_NAMES, RuntimeChoiceModel


def run(results_dir: Path, output_path: Path | None = None,
        datasets: tuple[str, ...] | None = None):
    datasets = datasets or ("bauxitemed", "cucase", "cupipe", "mclaughlingeo", "sim2d76")
    base_paths = [results_dir / f"adaptive_solver_training_{name}.csv" for name in datasets]
    missing = [str(path) for path in base_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing training data: " + ", ".join(missing))
    paths = base_paths + [results_dir / f"adaptive_solver_crossover_{name}.csv"
                          for name in datasets
                          if (results_dir / f"adaptive_solver_crossover_{name}.csv").exists()]
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = set(FEATURE_NAMES) | {"dataset", "fastest_solver"}
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"training CSVs are missing columns: {sorted(absent)}")

    x = frame.loc[:, FEATURE_NAMES]
    y = frame["fastest_solver"].astype(str).to_numpy()
    groups = frame["dataset"].astype(str).to_numpy()
    group_count = len(np.unique(groups))
    if group_count < 2:
        raise ValueError("at least two independent block models are required")
    cv = LeaveOneGroupOut()
    template = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                      class_weight="balanced", random_state=417)
    tree_predictions = cross_val_predict(template, x, y, groups=groups, cv=cv)
    lg_log = np.log(np.maximum(frame["lg_runtime_s"].astype(float).to_numpy(), 1e-6))
    hpf_log = np.log(np.maximum(frame["hpf_runtime_s"].astype(float).to_numpy(), 1e-6))
    timeout_flags = frame.get("lg_timed_out", pd.Series(False, index=frame.index)).fillna(False).astype(bool).to_numpy()
    predictions = np.empty(len(frame), dtype=object)
    for train_idx, test_idx in cv.split(x, y, groups):
        lg_reg = RandomForestRegressor(n_estimators=200, max_depth=6,
                                       min_samples_leaf=2, random_state=417, n_jobs=-1)
        hpf_reg = RandomForestRegressor(n_estimators=200, max_depth=6,
                                        min_samples_leaf=2, random_state=418, n_jobs=-1)
        # A timed-out LG runtime is only known to exceed the timeout. Treating
        # that lower bound as an exact duration biases its runtime regressor
        # toward choosing LG too often; keep censored rows for classification
        # labels, but exclude them from LG regression targets.
        lg_train_idx = train_idx[~timeout_flags[train_idx]]
        if len(lg_train_idx) < 2:
            raise ValueError("each training fold needs at least two uncensored LG runtimes")
        lg_reg.fit(x.iloc[lg_train_idx], lg_log[lg_train_idx])
        hpf_reg.fit(x.iloc[train_idx], hpf_log[train_idx])
        predictions[test_idx] = np.where(
            lg_reg.predict(x.iloc[test_idx]) < hpf_reg.predict(x.iloc[test_idx]), "LG", "HPF"
        )
    overall_accuracy = float(np.mean(predictions == y))
    overall_balanced_accuracy = float(balanced_accuracy_score(y, predictions))
    macro_f1 = float(f1_score(y, predictions, labels=["LG", "HPF"], average="macro", zero_division=0))
    lg_seconds = frame["lg_runtime_s"].astype(float).to_numpy()
    hpf_seconds = frame["hpf_runtime_s"].astype(float).to_numpy()
    selected_seconds = np.where(predictions == "LG", lg_seconds, hpf_seconds)
    tree_selected_seconds = np.where(tree_predictions == "LG", lg_seconds, hpf_seconds)
    observed_best_seconds = np.minimum(lg_seconds, hpf_seconds)
    always_lg_seconds = float(lg_seconds.sum())
    always_hpf_seconds = float(hpf_seconds.sum())
    selected_total_seconds = float(selected_seconds.sum())
    per_dataset = {}
    for dataset in sorted(set(groups)):
        mask = groups == dataset
        per_dataset[dataset] = {
            "samples": int(mask.sum()),
            "accuracy": float(np.mean(predictions[mask] == y[mask])),
            "true_labels": frame.loc[mask, "fastest_solver"].value_counts().to_dict(),
            "predicted_labels": pd.Series(predictions[mask]).value_counts().to_dict(),
        }
    lg_reg = RandomForestRegressor(n_estimators=200, max_depth=6,
                                   min_samples_leaf=2, random_state=417, n_jobs=-1)
    hpf_reg = RandomForestRegressor(n_estimators=200, max_depth=6,
                                    min_samples_leaf=2, random_state=418, n_jobs=-1)
    lg_train_mask = ~timeout_flags
    if int(lg_train_mask.sum()) < 2:
        raise ValueError("at least two uncensored LG runtimes are required to fit the selector")
    lg_reg.fit(x.loc[lg_train_mask], lg_log[lg_train_mask])
    hpf_reg.fit(x, hpf_log)
    model = RuntimeChoiceModel(lg_reg, hpf_reg)
    model_classes = ["HPF", "LG"]
    model_path = results_dir / "solver_selector_multidataset.joblib"
    joblib.dump(model, model_path)
    always_lg = np.full(len(y), "LG", dtype=object)
    result = {
        "datasets": sorted(set(groups)), "dataset_count": group_count,
        "sample_count": int(len(frame)), "label_counts": frame["fastest_solver"].value_counts().to_dict(),
        "leave_one_dataset_out_accuracy": overall_accuracy,
        "per_dataset_holdout": per_dataset,
        "decision_tree_leave_one_dataset_out_accuracy": float(np.mean(tree_predictions == y)),
        "decision_tree_balanced_accuracy": float(balanced_accuracy_score(y, tree_predictions)),
        "always_lg_baseline_accuracy": float(np.mean(y == "LG")),
        "always_lg_baseline_balanced_accuracy": float(balanced_accuracy_score(y, always_lg)),
        "leave_one_dataset_out_balanced_accuracy": overall_balanced_accuracy,
        "leave_one_dataset_out_macro_f1": macro_f1,
        "leave_one_dataset_out_selected_runtime_sum_s": selected_total_seconds,
        "decision_tree_selected_runtime_sum_s": float(tree_selected_seconds.sum()),
        "always_lg_observed_runtime_sum_s": always_lg_seconds,
        "always_hpf_observed_runtime_sum_s": always_hpf_seconds,
        "oracle_observed_runtime_sum_s": float(observed_best_seconds.sum()),
        "runtime_speedup_vs_always_lg": always_lg_seconds / max(selected_total_seconds, 1e-12),
        "runtime_speedup_vs_always_hpf": always_hpf_seconds / max(selected_total_seconds, 1e-12),
        "decision_tree_runtime_speedup_vs_always_lg": always_lg_seconds / max(float(tree_selected_seconds.sum()), 1e-12),
        "decision_tree_runtime_speedup_vs_always_hpf": always_hpf_seconds / max(float(tree_selected_seconds.sum()), 1e-12),
        "mean_relative_runtime_regret_vs_observed_oracle": float(np.mean(
            (selected_seconds - observed_best_seconds) / np.maximum(observed_best_seconds, 1e-9)
        )),
        "timed_out_lg_selected_count": int(np.sum((predictions == "LG") & timeout_flags)),
        "runtime_comparison_caveat": "LG timeout durations are lower bounds; runtime totals and regret are optimistic if a held-out prediction selects a timed-out LG sample.",
        "features": list(FEATURE_NAMES), "classes": model_classes,
        "model": str(model_path),
        "model_type": "two random-forest log-runtime regressors; choose the lower predicted time",
        "timed_out_lg_samples": int(timeout_flags.sum()),
        "lg_regression_timeout_handling": "censored timeout rows are excluded from LG runtime regression and retained in faster-solver labels",
        "timed_out_lg_interpretation": "timeouts are lower bounds on LG runtime; exact runtime is not observed",
        "limitation": "patches are spatial samples from five benchmark models; no industrial-site dataset is included",
    }
    output_path = output_path or results_dir / "solver_selector_multidataset.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run(ROOT / "results")
