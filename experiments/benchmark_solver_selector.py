"""Collect LG/HPF timing labels from spatial subgraphs for selector research.

This is a pilot on one block model.  It must not be treated as a generalization
claim; additional independent block models are required for that.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import signal
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpit.lg_literal import LiteralLG
from graph.load_graph import load_graph
from optimization.adaptive_solver import FEATURE_NAMES, _solve_hpf, partition_features

DATASET_DIMS = {
    "bauxitemed": (120, 120, 26), "cucase": (170, 215, 50),
    "cupipe": (180, 180, 85), "mclaughlingeo": (140, 296, 68),
    "sim2d76": (75, 1, 40),
}


class _SolverTimeout(TimeoutError):
    pass


def _timed_lg(values, precedence, timeout_s: float):
    """Run LiteralLG with a hard per-sample alarm; intended for Ubuntu."""
    engine = LiteralLG(values, precedence)
    started = time.perf_counter()
    previous = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum, _frame):
        raise _SolverTimeout

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        engine.solve()
    except _SolverTimeout:
        return None, time.perf_counter() - started, True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return engine, time.perf_counter() - started, False

def run(data: Path, outdir: Path, samples_per_size: int = 8,
        sizes=(2, 3, 4), seed: int = 417,
        dims: tuple[int, int, int] = DATASET_DIMS["bauxitemed"],
        max_lg_nodes: int = 512, lg_timeout_s: float = 10.0) -> dict:
    nx_dim, ny_dim, nz_dim = dims
    graph, frame = load_graph(str(data), nx_dim=nx_dim, ny_dim=ny_dim, nz_dim=nz_dim)
    count = len(frame)
    graph.remove_nodes_from([node for node in (0, count + 1) if node in graph])
    values_array = frame["EPV"].to_numpy(dtype=np.float64)
    coords = {}
    for node in graph.nodes:
        i = int(node) - 1
        xyz = (i % nx_dim, (i // nx_dim) % ny_dim, i // (nx_dim * ny_dim))
        coords[node] = xyz
        graph.nodes[node]["value"] = float(values_array[i])

    rng = np.random.default_rng(seed)
    rows = []
    skipped_large = 0
    for side in sizes:
        width_x, width_y = min(side, nx_dim), min(side, ny_dim)
        anchors = [(x, y) for x in range(0, nx_dim - width_x + 1, width_x + 9)
                   for y in range(0, ny_dim - width_y + 1, width_y + 9)]
        if len(anchors) > samples_per_size:
            selected = rng.choice(len(anchors), samples_per_size, replace=False)
            anchors = [anchors[int(i)] for i in sorted(selected)]
        for sample_id, (x0, y0) in enumerate(anchors):
            members = [node for node, (x, y, _) in coords.items()
                       if x0 <= x < x0 + width_x and y0 <= y < y0 + width_y]
            if len(members) > max_lg_nodes:
                skipped_large += 1
                continue
            sub = graph.subgraph(members)
            values = {node: float(graph.nodes[node]["value"]) for node in members}
            features = partition_features(sub, values)

            precedence = {node: list(sub.successors(node)) for node in members}
            lg, lg_seconds, lg_timed_out = _timed_lg(values, precedence, lg_timeout_s)
            lg_npv = None
            if not lg_timed_out:
                lg_set = lg.closure()
                lg_npv = float(sum(values[node] for node in lg_set))

            started = time.perf_counter()
            hpf_set = _solve_hpf(sub, values)
            hpf_seconds = time.perf_counter() - started
            hpf_npv = float(sum(values[node] for node in hpf_set))
            label = "HPF" if lg_timed_out or hpf_seconds <= lg_seconds else "LG"
            rows.append({
                "dataset": data.stem, "sample_id": f"side{side}_{sample_id}", "side": side,
                "x0": x0, "y0": y0, "region_group": (x0 // 60) * 2 + (y0 // 60),
                **features, "lg_runtime_s": lg_seconds, "hpf_runtime_s": hpf_seconds,
                "lg_npv": lg_npv, "hpf_npv": hpf_npv,
                "lg_timed_out": lg_timed_out,
                "objective_delta_lg_minus_hpf": (lg_npv - hpf_npv) if lg_npv is not None else None,
                "fastest_solver": label,
            })
            print(f"{rows[-1]['sample_id']} n={features['node_count']} m={features['arc_count']} "
                  f"LG={lg_seconds:.4f}s HPF={hpf_seconds:.4f}s label={label}", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    kind = "training" if max(sizes) <= 4 else "crossover"
    csv_path = outdir / f"adaptive_solver_{kind}_{data.stem}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    labels = sorted({row["fastest_solver"] for row in rows})
    result = {
        "dataset": data.stem, "dimensions": list(dims), "samples": len(rows), "sample_sizes": list(sizes),
        "samples_skipped_above_lg_limit": skipped_large, "lg_sample_node_limit": max_lg_nodes,
        "labels": {label: sum(row["fastest_solver"] == label for row in rows) for label in labels},
        "mean_absolute_objective_delta": float(np.mean([abs(r["objective_delta_lg_minus_hpf"]) for r in rows if r["objective_delta_lg_minus_hpf"] is not None])) if any(r["objective_delta_lg_minus_hpf"] is not None for r in rows) else None,
        "csv": str(csv_path), "ml_model": None,
        "scope_warning": "single-model spatial-patch pilot; not an independent validation set",
    }
    if len(rows) >= 12 and len(labels) >= 2:
        import joblib
        import pandas as pd
        from sklearn.model_selection import GroupKFold, cross_val_score
        from sklearn.tree import DecisionTreeClassifier

        x = pd.DataFrame([{name: row[name] for name in FEATURE_NAMES} for row in rows], columns=FEATURE_NAMES)
        y = np.asarray([row["fastest_solver"] for row in rows])
        groups = np.asarray([row["region_group"] for row in rows])
        model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2,
                                       class_weight="balanced", random_state=seed)
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            cv = GroupKFold(n_splits=min(4, len(unique_groups)))
            scores = cross_val_score(model, x, y, groups=groups, cv=cv, scoring="accuracy")
            result["group_cv_accuracy_mean"] = float(scores.mean())
            result["group_cv_accuracy_std"] = float(scores.std())
        model.fit(x, y)
        model_path = outdir / f"solver_selector_{data.stem}.joblib"
        joblib.dump(model, model_path)
        result["ml_model"] = str(model_path)
        result["model_classes"] = list(model.classes_)
    else:
        result["ml_status"] = "not fitted: need at least 12 samples and both solver labels"
    json_path = outdir / f"adaptive_solver_training_{data.stem}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--outdir", type=Path, default=ROOT / "results")
    parser.add_argument("--samples-per-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--dataset", choices=tuple(DATASET_DIMS), default="bauxitemed")
    parser.add_argument("--dimensions", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    parser.add_argument("--max-lg-nodes", type=int, default=512)
    parser.add_argument("--lg-timeout-s", type=float, default=10.0)
    parser.add_argument("--sizes", type=int, nargs="+", default=(2, 3, 4))
    args = parser.parse_args()
    data = args.data or ROOT / "data/BauxiteMed" / f"{args.dataset}.dat"
    dims = tuple(args.dimensions) if args.dimensions else DATASET_DIMS[args.dataset]
    run(data, args.outdir, args.samples_per_size, seed=args.seed,
        dims=dims, max_lg_nodes=args.max_lg_nodes,
        lg_timeout_s=args.lg_timeout_s, sizes=tuple(args.sizes))


if __name__ == "__main__":
    main()
