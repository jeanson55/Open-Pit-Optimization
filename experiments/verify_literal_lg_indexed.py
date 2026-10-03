"""Check indexed LiteralLG against exhaustive maximum-closure enumeration."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpit.lg_literal import LiteralLG


def _exact(values: dict[int, float], precedence: dict[int, list[int]]):
    nodes = list(values)
    best = float("-inf")
    for mask in range(1 << len(nodes)):
        selected = {nodes[i] for i in range(len(nodes)) if mask & (1 << i)}
        if any(source in selected and target not in selected
               for source, targets in precedence.items() for target in targets):
            continue
        best = max(best, sum(values[node] for node in selected))
    return best


def run(cases: int = 200, seed: int = 3651) -> dict:
    rng = random.Random(seed)
    total_seconds = 0.0
    merger_count = prune_count = 0
    for case in range(cases):
        n = rng.randint(4, 11)
        values = {node: rng.randint(-1200, 1200) / 100.0 for node in range(1, n + 1)}
        precedence = {node: [] for node in values}
        for source in values:
            for target in range(source + 1, n + 1):
                if rng.random() < 0.27:
                    precedence[source].append(target)
        expected = _exact(values, precedence)
        engine = LiteralLG(values, precedence)
        started = time.perf_counter()
        solution = engine.solve()
        total_seconds += time.perf_counter() - started
        actual = sum(values[node] for node in solution)
        violations = engine.violations()
        if violations:
            raise AssertionError(f"case {case}: returned {violations} precedence violations")
        if abs(actual - expected) > 1e-8:
            raise AssertionError(
                f"case {case}: LiteralLG objective {actual} != exact closure {expected}"
            )
        merger_count += engine.merger_count
        prune_count += engine.prune_count
    return {
        "seed": seed,
        "cases": cases,
        "max_nodes_per_case": 11,
        "objective_matches": cases,
        "feasible_solutions": cases,
        "mergers": merger_count,
        "prunes": prune_count,
        "solve_seconds_total": total_seconds,
        "status": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3651)
    parser.add_argument("--output", type=Path,
                        default=Path("results/literal_lg_indexed_verification.json"))
    args = parser.parse_args()
    result = run(args.cases, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
