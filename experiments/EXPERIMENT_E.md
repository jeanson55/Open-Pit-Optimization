# Experiment E: Intelligent Partitioning

`experiment_e_partitioning.py` compares the 9-way spatial baseline, balanced spectral partitioning, METIS, and spatial-plus-graph hybrid partitioning on the bauxitemed precedence graph. Each partition is solved as a maximum-weight closure, the local pits are merged, merge violations are measured and clustered, and global feasibility is restored by adding required successor blocks.

The active graph contains **240,502 slope-valid blocks** and **1,130,500 precedence arcs**. EPV comes from the bauxitemed input values. The graph follows the gPit arc implication: selecting `u` on `u -> v` requires `v`.

## Full Ubuntu run

The run used the project's existing Linux virtual environment, with SciPy 1.18.1, scikit-learn 1.9.1, and pymetis 2025.2.2 installed into Ubuntu-native `/home/jeanson/pit-exp-deps` to avoid slow package writes on the Windows-mounted virtual environment. It ran with that dependency directory on `PYTHONPATH`; the JSON report has no fallback markers, confirming the genuine methods ran.

| Method | Balance (max/mean) | Cut arcs | Cut fraction | Initial violations | Repair additions | Cascade depth | Final NPV | Partition time | Local solve | Repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial | 1.557 | 18,800 | 1.663% | 878 | 3,864 | 19 | 28,610,119 | 1.9 s | 145.2 s | 0.57 s |
| Spectral, balanced | 1.000 | 70,691 | 6.253% | 2,990 | 15,662 | 20 | 19,590,009 | 140.7 s | 133.4 s | 0.61 s |
| METIS | 1.000 | 22,631 | 2.002% | 1,172 | 7,046 | 20 | 25,472,696 | 8.8 s | 222.1 s | 0.78 s |
| Hybrid | 1.557 | 18,800 | 1.663% | 878 | 3,864 | 19 | 28,610,119 | 16.6 s | 241.3 s | 0.69 s |

Every refined solution had zero precedence violations. Spatial and hybrid results matched; the current hybrid refinement did not improve on the spatial assignment. METIS achieved much better size balance than spatial, with a modestly higher cut fraction, but its repair additions reduced final NPV further. Balanced spectral had the worst cuts, violations, and final NPV in this run. Thus, balance alone did not predict a better repaired objective.

The baseline global pit NPV is **29,690,715** and has zero violations under the graph's directed convention. Partitioned NPV values are sums of the input EPV over each reported repaired selection; they are not discounted mine-life cash flows.

## Reproduction

From the project root in Ubuntu, activate/use the Linux environment containing `pseudoflow`, then run:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps .venv-linux/bin/python experiments/experiment_e_partitioning.py
```

Adjust `.venv-linux/bin/python` to the project's Linux environment path if needed. The JSON report is written to `results/experiment_e_partitioning.json`. It contains partition sizes, cut arc cluster statistics, violation clusters, initial/final NPV and block counts, cascade additions/depth, and phase runtimes.

The spectral method requires SciPy and scikit-learn; METIS requires pymetis. The driver marks dependency fallbacks explicitly if those packages are unavailable.

## Hybrid V4.1: spatial seed with graph-boundary refinement and swaps

The earlier `hybrid_spatial_graph` result above used V3.2, which generated Spatial and METIS assignments and selected one; on the reported run it selected Spatial, so its labels were identical to the Spatial baseline. Hybrid V4 initially made no single-block moves on full bauxitemed, showing that the spatial cut was a one-node local minimum. The current `partition/hybrid_spatial_graph.py` is V4.1: it keeps the spatial seed, tries graph-based single-block moves at cut boundaries, and then tries equal-size swaps across cut arcs to escape that local minimum. A single-block move is accepted only when the directed cut strictly decreases or the cut stays equal while size balance improves; pair swaps must strictly reduce the cut. Single-block moves obey a hard 1.15 x mean-size receiving cap, and swaps preserve all part sizes. V3.2 and the first V4 run remain reproducible as historical results.

The dependency-light verifier passed 100 seeded small directed graphs: 4,833 move-delta and 4,691 swap-delta calculations matched full cut recomputation; all 100 refinements were deterministic and none increased the cut. A constructed single-move local minimum was improved by one equal-size swap. These checks validate the local mechanics, not full-model performance. A full bauxitemed Spatial/V4.1 comparison has now completed:

| Method | Balance (max/mean) | Cut arcs | Initial violations | Repair additions | Final violations | Final NPV | Partition time | Local solve | Repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spatial | 1.557 | 18,800 | 878 | 3,864 | 0 | 28,610,119 | 1.58 s | 126.52 s | 0.49 s |
| Hybrid V4.1 | 1.557 | 18,800 | 878 | 3,864 | 0 | 28,610,119 | 5.16 s | reused identical Spatial solve | 0.49 s |

V4.1 made zero single-block moves and zero pair swaps on this graph, so its partition and repaired solution exactly matched Spatial. It cost about 3.6 s more partition time and gave no solution gain. The search is locally stable under its one-block and adjacent equal-size-swap moves; further refinement is not pursued here. Full metrics are in `results/experiment_e_hybrid_v4_1_bauxitemed.json`.

Reproduce the paired Spatial/V4.1 comparison in Ubuntu from the project root:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps .venv-linux/bin/python experiments/experiment_e_partitioning.py \
  --methods bench_spatial hybrid_spatial_graph_v4_1 \
  --output results/experiment_e_hybrid_v4_1_bauxitemed.json \
  2>&1 | tee results/experiment_e_hybrid_v4_1_bauxitemed.log
```

Use the same Linux Python environment path as the earlier full run if it is not `.venv-linux/bin/python`. This paired run focuses on the Spatial baseline and V4.1; compare its result to the existing genuine-METIS result above. The new JSON should include V4.1 cut/move/swap/sweep diagnostics alongside initial violations, cascade additions, final violations, feasible NPV, and solver/refinement runtimes.
