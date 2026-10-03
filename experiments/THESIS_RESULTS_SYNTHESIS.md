# Thesis results synthesis (working draft)

This draft consolidates the strongest measured evidence for the five thesis objectives. It reports negative findings as results and does not imply that an experimental selector is useful when it loses to a simple baseline. Final thesis formatting and references still need to be added.

## Overall summary

The project implements and evaluates maximum-closure pit optimization using LiteralLG and HPF, spatial/METIS/spectral partitioning, adaptive local solving, feasibility repair, boundary refinement, and ML strategy selectors. On the tested block models, repair and boundary refinement restored precedence feasibility. Partitioning can approach the global HPF reference, but did not show an end-to-end runtime advantage. Direct full-model LiteralLG/HPF comparisons on bauxitemed and cucase returned identical block membership and NPV; HPF was substantially faster on both.

## Objective 1 — Intelligent graph partitioning

Four-deposit comparisons covered Spatial and METIS at matched partition counts; spectral partitioning was additionally tested at matched k=4 across the four 3D deposits. Experiment E compared Spatial, METIS, balanced spectral, and the earlier spatial/graph candidate selector on full bauxitemed. In its nine-part run, Spatial produced the best repaired NPV (28,610,119), followed by METIS (25,472,696) and balanced spectral (19,590,009); the earlier hybrid matched Spatial (28,610,119) because it selected the spatial assignment. The global HPF reference was 29,690,715. All reported repaired solutions were feasible.

Balanced spectral partitioning achieved exact size balance but incurred 70,691 cut arcs (6.253%) and a final NPV well below Spatial. METIS achieved exact balance with 22,631 cut arcs (2.002%) and also underperformed Spatial economically. This evidence shows that partition balance alone did not predict the repaired objective. Hybrid V4.1 adds graph-boundary moves and equal-size swaps. It made zero moves on full bauxitemed, returning the same 18,800 cut arcs, 1.557 max/mean balance, 878 initial violations, 3,864 repair additions, zero final violations, and NPV 28,610,119 as Spatial; its partitioning took 5.16 s versus Spatial's 1.58 s. Local verification passed 100 directed graphs, with 4,833 single-move and 4,691 swap-delta checks. The earlier selector pilot did not beat the simple Spatial baseline.

## Objective 2 — Adaptive LG/HPF selection

The adaptive solver includes heuristic and learned policies with an LG size guard. The collected solver timing data contain 147 paired samples (130 LG-faster and 17 HPF-faster). On the four 3D deposits, the runtime-regression model achieved 84.8% accuracy compared with 86.4% for always-LG; its observed selected runtime sum was 72.02 s versus 26.16 s for always-HPF. The decision tree reached 90.4% accuracy and 77.1% balanced accuracy, but its observed runtime sum was 38.67 s, also slower than always-HPF. These results do not justify deploying the learned selector. The selector infrastructure is implemented and evaluated; a reliable held-out runtime advantage remains unproven.

## Objective 3 — Dynamic boundary refinement

Boundary refinement was evaluated on four full 3D block models. The best tested settings improved NPV over add-only repair by 0.028%–1.215% while leaving zero precedence violations. A deterministic verification on 100 seeded small DAGs matched exhaustive maximum-closure values in all 100 cases and retained a feasible solution without reducing the add-only incumbent in all 100 cases. This supports the implementation’s feasibility and conditional-optimality behavior; it does not establish a global optimum for the full model or prove scaling beyond the tested instances.

## Objective 4 — ML strategy selection

The ML experiments include local solver-runtime prediction and partition method/count selection. The V3.6 composite partition selector matched 7 of 12 paired outcomes and had 3.890% mean regret, compared with 8 of 12 and 0.198% regret for always-Spatial. The V3.7 joint selector pilot used 28 candidate outcomes and grouped leave-one-deposit-out validation; it reached 75% winner accuracy and 0.102% regret, exactly matching the training-selected Spatial count and fixed Spatial k=4. It selected Spatial k=4 in every held-out fold. The pilot therefore did not demonstrate benefit over its simple baseline and is not deployed. Reporting the negative result is part of the evaluation; further model tuning without additional independent data is not a priority for project closeout.

## Objective 5 — Standalone solver comparison on industrial data

On full bauxitemed (240,502 graph nodes; 1,130,500 precedence arcs), LiteralLG and HPF each selected 73,419 blocks, returned NPV 29,690,715, had zero precedence violations, and had exact membership agreement (Jaccard 1.0). After a normalization traversal-reuse optimization, LiteralLG took 13,206.654 s, a 4.34% reduction from its 13,805.995 s baseline; HPF took 23.391 s in the repeated run. The observed HPF/LiteralLG runtime ratio was 564.6x. Peak whole-process RSS was 2,207 MiB.

On full cucase (1,045,952 graph nodes; 5,047,000 precedence arcs), both solvers selected 327,235 blocks, returned NPV 19,444,156, had zero violations, and had exact membership agreement (Jaccard 1.0). LiteralLG took 5,939.303 s and HPF took 105.417 s (56.3x speedup); peak whole-process RSS was 6,266.4 MiB. The exactness verifier also matched exhaustive closure values and returned feasible solutions on 200 seeded DAGs of 4–11 nodes.

Four-model global HPF references and partitioned results are available for the supplied 3D deposits. Direct full-scale LiteralLG/HPF comparisons are complete for bauxitemed and cucase. The historical full-model solver named “LG” in the companion report is NetworkX max-flow, not LiteralLG, so it must be identified separately. LiteralLG’s multi-hour runtime makes additional full-model comparisons high-cost; direct LiteralLG results are not available for CUPIPE or McLaughlinGeo.

## Interpretation and limitations

- Every result is specific to the supplied precedence construction, EPV values, and datasets. The reported NPV is the sum of selected input EPV and is not a discounted mine-life cash-flow valuation.
- SIM2D76 is excluded from industrial precedence conclusions because its supplied geometry yields a degenerate graph under the current builder.
- Spatial performed strongly on these samples, but the number of independent deposits is too small to claim general superiority.
- Runtime measurements depend on the Ubuntu host, installed solvers, and execution environment. The two bauxitemed LiteralLG timings show a modest optimization; they do not make LiteralLG runtime-competitive with HPF.
- Where ML failed to beat a simple baseline, the evidence supports a negative finding and a recommendation not to deploy the model.

## Evidence files

- Consolidated exact-vs-partitioned comparison, initial/final refinement metrics, current input checksums, environment provenance, NPV comparison, and 2D/3D bauxitemed partition/pit views: `experiments/THESIS_ADDITIONS.md`, `results/thesis_additions_comparison.csv`, `results/thesis_additions_npv_comparison.svg`, `results/thesis_additions_bauxitemed_partition_map.svg`, `results/bauxitemed_3d_block_model.png`, and `results/thesis_additions_provenance.json`. Reporting files regenerate with `experiments/build_thesis_additions.py`; the 3D PNG regenerates with `experiments/build_bauxitemed_3d_block_model.py`.

- Objective matrix: `experiments/THESIS_OBJECTIVE_STATUS.md`
- Partition comparisons: `experiments/PARTITION_GRID_RESEARCH.md`, `experiments/EXPERIMENT_E.md`
- Adaptive solver and boundary refinement: `experiments/ADAPTIVE_SOLVER.md`
- Direct bauxitemed comparison before optimization: `results/global_solver_benchmark_bauxitemed.json`
- Direct bauxitemed comparison after optimization: `results/global_solver_benchmark_bauxitemed_normreuse.json`
- Direct cucase LiteralLG/HPF comparison: `results/global_solver_benchmark_cucase.json`
- Boundary-refinement correctness checks: `results/boundary_refinement_verification.json`
- Full bauxitemed Spatial/Hybrid V4.1 comparison: `results/experiment_e_hybrid_v4_1_bauxitemed.json`
- Hybrid V4.1 local mechanics checks: `results/hybrid_spatial_graph_verification.json`
- LiteralLG exactness checks: `results/literal_lg_indexed_verification.json`
- V3.7 selector metrics: `results/hybrid_v37_joint_selector_metrics.json`
- Expanded solver crossover analysis: `results/solver_crossover_expansion_20260929_01/`
