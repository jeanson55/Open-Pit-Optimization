# Adaptive solver selection experiment

This experiment adds an LG/HPF decision point inside the existing partitioned solve. It first restores feasibility, then reoptimizes the cut and cascade boundary while fixing outside decisions. It is a research implementation for formal objectives 2, 3, and 4. The results below separate full-model evidence from patch-level ML validation.

## Implementation

`optimization/adaptive_solver.py` extracts graph and economic features for each partition, chooses LiteralLG or the installed Pseudoflow engine, merges local solutions, repairs the precedence cascade, and reoptimizes a boundary graph. Nodes outside the mutable boundary stay fixed. Selected outside predecessors force boundary successors in; unselected outside successors force boundary predecessors out. This is an exact maximum-closure solve conditional on the outside decisions, so it cannot reduce the feasible repaired seed objective when the HPF value precision matches the block values. The HPF adapter compacts arbitrary partition node identifiers and preserves directed precedence arcs. As in gPit, HPF rounds block values to cents. LG is capped at 4,096 nodes per partition because the current LiteralLG implementation can have a long runtime on large graphs.

The initial heuristic sends partitions with at most 128 nodes and at most 2,048 combined nodes and arcs to LG, and larger partitions to HPF. A learned runtime model can replace the heuristic with `--policy ml --model <joblib-file>`; the LG size cap remains active as a runtime guard. The model predicts log runtime for both solvers from graph and economic features, then selects the lower prediction.

## Add-only bauxitemed baseline

The run used METIS with 9 partitions on the 120 x 120 x 26 bauxitemed model. All nine parts contained about 26,700 slope-valid blocks, so the selector chose HPF for all nine. Partitioning took 6.92 s, local HPF solves took 26.33 s in total, and boundary repair took 0.51 s.

The merged candidate had 1,790 precedence violations. Add-only repair added 13,953 blocks through 25 cascade levels and left zero violations. Its final NPV was 25,472,696, compared with 29,690,715 for the standalone global HPF baseline. Conditional boundary refinement later raised it to 26,556,907 while retaining zero violations.

The learned-model run selected HPF for all nine full-size partitions and produced the same final NPV and zero-violation result. These partitions are much larger than the training patches, so this run validates execution and the LG runtime cap, not model generalization.

## Spatial partition with boundary refinement

To check whether boundary refinement helps the strongest add-only partition result, the bauxitemed spatial 9-part assignment was rerun with the boundary refiner. It finished at NPV 28,804,730 with zero final violations, up 194,611 from the spatial add-only result (28,610,119) and 2.98% below the global HPF objective (29,690,715). The initial merge had 1,691 violations; the refiner optimized a 27,405-node, 109,874-arc mutable region in 7.36 s. Full metrics are in `results/experiment_boundary_refinement_bauxitemed_spatial.json`.

This closes some of the objective gap but does not establish a speed advantage. The partitioned run took 45.39 s after graph loading (1.69 s partitioning, 27.54 s local HPF, 7.36 s refinement, plus other overhead). The paired global HPF record took 27.52 s to solve after its 16.73 s graph build. Adding the same graph-build time to the partitioned run gives about 62.12 s end-to-end, versus 44.25 s for global HPF, roughly 40% slower. A separate historical companion report records different HPF timings; those are not mixed into this run-paired comparison.

## Evaluation across the supplied block models

The same METIS 9-part run was completed for the four three-dimensional models. The table compares the earlier add-only repair, the new conditional boundary refinement, and the standalone global HPF reference from the companion validation. The source data files have matching SHA-256 hashes and graph counts.

| Model | Active nodes | Add-only NPV | Boundary-refined NPV | Gain over add-only | Global HPF NPV | Remaining gap |
|---|---:|---:|---:|---:|---:|---:|
| bauxitemed | 240,502 | 25,472,696 | 26,556,907 | 1,084,211 | 29,690,715 | -10.6% |
| cucase | 1,045,952 | 19,025,532 | 19,099,563 | 74,031 | 19,444,156 | -1.8% |
| cupipe | 988,042 | 57,332,742 | 59,155,671 | 1,822,929 | 142,915,803 | -58.6% |
| mclaughlingeo | 1,241,546 | 1,172,258,710 | 1,173,518,708 | 1,259,998 | 1,190,881,614 | -1.5% |

Every boundary-refined solution has zero precedence violations. The add-only repair cascades were 13,953 blocks across 25 levels on bauxitemed, 21,031 across 26 levels on cucase, 497,200 across 78 levels on cupipe, and 78,558 across 39 levels on McLaughlinGeo. Boundary reoptimization removed 1,986, 726, 6,163, and 1,937 blocks respectively; the corresponding mutable regions contained 32,853, 78,523, 521,256, and 141,404 nodes. Refinement took 6.57, 22.14, 92.34, and 32.28 seconds respectively. The method improves on add-only repair, but does not recover the global optimum; CUPIPE remains 58.6% below its global reference and needs wider or iterative coordination.

The companion full-model comparison is `../../lg/benchmark_four_real_final_summary.csv`. Its solver named `LG` is a NetworkX max-flow/min-cut maximum-closure implementation; it is distinct from the literal tree-merger implementation `gpit.lg_literal.LiteralLG` in this folder. Its five source `.dat` files have SHA-256 matches with this project's five inputs. On the four three-dimensional models, the recorded report measured HPF / max-flow-reference solve times of 3.75 / 47.40 s (bauxitemed), 17.14 / 113.25 s (cucase), 17.83 / 899.45 s (cupipe), and 21.34 / 565.66 s (McLaughlinGeo), with equal global NPV.

### Direct full-scale LiteralLG / HPF comparison: bauxitemed

The direct benchmark `results/global_solver_benchmark_bauxitemed.json` completed on the full 120 x 120 x 26 bauxitemed model. The graph contained 240,502 slope-valid nodes and 1,130,500 precedence arcs. LiteralLG and HPF returned exactly the same closure: 73,419 selected blocks, NPV 29,690,715, zero precedence violations, and Jaccard membership 1.0. LiteralLG took 13,805.995 s (3 h 50 min 6 s); HPF took 20.158 s. The repeated optimized run is described below. Graph construction took 14.519 s in the baseline run. Peak process RSS was 2,211.1 MiB; this is the whole benchmark process, not solver-specific memory.

### Direct full-scale LiteralLG / HPF comparison: cucase

The direct benchmark `results/global_solver_benchmark_cucase.json` completed on the full 170 x 215 x 50 cucase model. Its graph contained 1,045,952 nodes and 5,047,000 precedence arcs. LiteralLG and HPF returned exactly the same closure: 327,235 selected blocks, NPV 19,444,156, zero precedence violations, and Jaccard membership 1.0. LiteralLG took 5,939.303 s (1 h 38 min 59 s); HPF took 105.417 s, making HPF 56.3 times faster in this run. Graph construction took 64.525 s. Peak process RSS was 6,266.4 MiB for the whole benchmark process, not solver-specific memory.

Direct full-model LiteralLG/HPF agreement is now demonstrated on two industrial datasets, bauxitemed and cucase. The partitioned framework has separate full-model comparisons against global HPF references across four 3D datasets. The direct results do not imply LiteralLG runtime competitiveness; HPF was much faster on both models.

The captured LiteralLG phase counters attribute 10,030.5 s to normalization and 3,732.2 s to merger-arc finding in the baseline. Reusing the post-order mass-recomputation traversal to seed the normalization violation heap reduced the repeated-run LiteralLG time from 13,805.995 s to 13,206.654 s, saving 599.341 s (4.34%). The normalization counter fell from 10,030.5 s to 9,439.1 s (591.4 s, 5.90%); merger finding remained near 3,725 s. The before/after runs both returned the identical 73,419-block closure and NPV 29,690,715, with zero violations. The optimized run's HPF time was 23.391 s, so its measured speedup was 564.6x; HPF timing varied from 20.158 s in the baseline and is not the optimization target. The 200-case exact small-DAG verifier also passed after the change. Detailed optimized metrics are in `results/global_solver_benchmark_bauxitemed_normreuse.json`.

SIM2D76 was also run, but its supplied 75 x 1 x 40 geometry is degenerate under the current gPit precedence builder: only 77 graph nodes are produced and the direct loader reports no real precedence arcs, while the construction excludes 97.5% of its 3,000 blocks. Its 754 NPV and exact LG/HPF agreement are therefore not counted as meaningful slope-constrained validation.

## Timing-label pilot

`benchmark_solver_selector.py` sampled vertical spatial patches from all five models and compared measured LiteralLG and HPF runtimes. Across 99 eligible samples, LG was faster in 91 and HPF in 8; three LG samples reached the 10-second timeout, so their exact runtimes are right-censored. All completed solver pairs matched in objective at the recorded precision.

Two random-forest regressors predict LG and HPF log runtime from 13 graph and economic features. Leave-one-model-out validation gave 89.9% accuracy and 66.0% balanced accuracy, compared with 91.9% accuracy and 50.0% balanced accuracy for always choosing LG. The runtime model improves minority-class recall but does not improve overall accuracy. Only eight samples favor HPF, and three LG labels are timeout lower bounds; more balanced large-patch samples are needed before claiming a reliable general selector.

An additional bounded bauxitemed crossover batch tested six larger spatial patches (sides 5, 6, and 7; 1,600-node LG cap; 5-second LG limit). Four favored LG and two favored HPF after LG timed out; all completed LG/HPF objective comparisons agreed exactly. A separate augmented training snapshot contains 105 samples (95 LG, 10 HPF; five timeout-censored LG samples). The first augmented fit reported 88.6% accuracy and 62.4% balanced accuracy, but it incorrectly treated timed-out LG durations as exact 5-second regression targets. `experiments/train_multidataset_selector.py` now excludes censored LG rows from LG runtime regression while retaining their HPF-faster classification labels. The corrected fit has 88.6% accuracy, 48.9% balanced accuracy, and 47.0% macro-F1, below the always-LG baseline of 90.5% / 50.0%; it predicts no HPF for four of the five held-out models. This corrected augmented model is diagnostic, not ready for deployment. Its CSVs, metrics, and model are preserved under `results/selector_augmented_20260927_01/`; the pre-existing model in `results/` was not replaced.

## Small correctness check

A three-node closure with a known optimum returned the same objective from LiteralLG and Pseudoflow. A cut-boundary regression case produced NPV 2 after add-only repair and NPV 7 after boundary reoptimization, with zero final violations and no reduction below the feasible seed.

## LiteralLG indexing optimization and verification

`gpit/lg_literal.py` now indexes original precedence arcs and keeps a versioned
heap of weak-over-strong merger candidates. Heap keys preserve the original
source/list scan order. Normalization similarly uses an ordered candidate
heap for locally strong mid-branch edges and refreshes only nodes whose mass
or root status changes. A subsequent review removed an accidental O(n)
per-normalization version-map initialization, leaving that map sparse. Branch
mass recomputation after each merger remains branch-linear and can still be
expensive; this is not the incremental production LG data structure.

The seeded verification script `experiments/verify_literal_lg_indexed.py`
compared 200 random DAGs (4–11 nodes each) against exhaustive maximum-closure
enumeration. All 200 objectives matched exactly; every returned closure was
feasible. A 1,200-node, 1,690-arc synthetic DAG solved in 0.13 seconds with
zero precedence violations. These checks establish correctness on small
graphs and a limited synthetic smoke timing, not on industrial mining graphs.
A 6,000-node synthetic stress case was interrupted after repeated branch
recomputation remained expensive. The full LiteralLG bauxitemed/HPF
comparison remains unrun: the Ubuntu execution escalation was rejected after
the environment usage limit was reached, and the available bundled local
Python does not include the project's NetworkX/scientific dependencies.
Detailed recorded results are in
`results/literal_lg_indexed_verification.json`.

## Reproduction

From the project root in Ubuntu, using the Linux environment and optional dependency directory used for the experiments:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/experiment_adaptive_solver.py --dataset bauxitemed --method metis --partitions 9
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/experiment_adaptive_solver.py --dataset cupipe --method metis --partitions 9 --refinement boundary
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/benchmark_solver_selector.py --dataset bauxitemed --samples-per-size 6
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/train_multidataset_selector.py
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/experiment_adaptive_solver.py --dataset bauxitemed --method metis --partitions 9 --policy ml --model results/solver_selector_multidataset.joblib --output results/experiment_adaptive_solver_ml.json
```

The machine-learning artifacts are `results/adaptive_solver_training_<dataset>.csv`, `results/adaptive_solver_crossover_<dataset>.csv`, `results/solver_selector_multidataset.json`, and `results/solver_selector_multidataset.joblib`. Add-only metrics are in `results/experiment_adaptive_solver_<dataset>.json`; boundary-refined results are in `results/experiment_boundary_refinement_<dataset>.json`.

## Hybrid V3.6 economic-aware partition ranking

V3.6 adds a pre-solve selector between Spatial and METIS. It combines a classifier probability with economic-boundary exposure, size balance, and precedence-cut quality. The model uses only pre-solve graph/partition features; realized NPV, runtime, and post-solve violations are excluded from model inputs. Training outcomes are historical Spatial/METIS NPVs from `results/final_hybrid_comparison.csv`; the grouped validation leaves one deposit out at a time.

The first four-deposit leave-one-dataset-out result is 2/4 correct (50%) for the composite selector, 0/4 for ML probability alone, and 2/4 for either always-Spatial or always-METIS. This is a pilot, not evidence that V3.6 generalizes or beats a simple baseline. The score currently normalizes economic, balance, and cut measures only relative to the two candidates in each deposit. Spatial wins both economic and cut comparisons on all four deposits, which gives it two full normalized advantages; as a result, the current selector chooses Spatial on all four. Expanding validation data is needed before selecting new score weights from this small sample.

### Full adaptive add-only runs

All four runs completed in Ubuntu with eight partitions, the heuristic local solver policy, and add-only feasibility repair. The initial violations and cluster/cascade fields below are from each run's measured output. Final precedence violations are zero in every case. `Gap` is relative to the best historical Spatial/METIS NPV in the existing comparison file.

| Model | Selected partition | Initial violations | Violation clusters | Blocks added | Cascade depth | Repair time (s) | Final violations | Final NPV | Gap to best historical | Partition time (s) | Total time (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bauxitemed | Spatial | 1,465 | 1,438 | 9,409 | 25 | 0.49 | 0 | 28,652,908 | 0.000% | 14.01 | 36.23 |
| cucase | Spatial | 2,309 | 2,260 | 11,060 | 34 | 2.10 | 0 | 19,359,730 | 0.212% | 59.58 | 165.83 |
| cupipe | Spatial | 12,257 | 12,096 | 385,662 | 60 | 4.76 | 0 | 57,815,484 | 1.681% | 58.21 | 184.98 |
| mclaughlingeo | Spatial | 6,350 | 6,321 | 79,063 | 38 | 3.59 | 0 | 1,179,217,403 | 0.000% | 77.13 | 222.05 |

V3.6 matches the best historical NPV for bauxitemed and McLaughlinGeo, but misses by 0.212% on cucase and 1.681% on CUPIPE because it chooses Spatial where historical METIS results were higher. Final feasibility was restored for every case. These results fulfill the end-to-end execution/measurement step while showing that partition selection still needs work.

Detailed per-run metrics are in `results/adaptive_<dataset>_hybrid_v36_add.json`; the aggregate is `results/hybrid_v36_final_summary.json`; a flat comparison is `results/hybrid_v36_final_comparison.csv`. Training features/labels and grouped validation are in `results/hybrid_v36_training_data.csv` and `results/hybrid_v36_training_metrics.json`.

Reproduce the training and all four full solves from the project root in Ubuntu:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/train_hybrid_v36_selector.py
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/run_adaptive_hybrid_v36_all.py
```

## Expanded larger-patch LG/HPF crossover pilot

To improve the runtime-label coverage, 48 additional paired solves were collected on bauxitemed, cucase, CUPIPE, and McLaughlinGeo: 12 spatial regions per deposit across footprint sizes 8, 9, and 10, with a 4,096-node LiteralLG cap and a five-second LG timeout. LG and HPF returned identical objectives on all 48 pairs. The fastest-solver labels were 8 LG / 4 HPF on bauxitemed, 9 / 3 on cucase, 12 / 0 on CUPIPE, and 10 / 2 on McLaughlinGeo.

Combining these with the existing samples produced 147 rows across five model groups (130 LG-faster, 17 HPF-faster). Leave-one-deposit-out validation of the runtime-regression selector reached 79.6% accuracy and 52.7% balanced accuracy, versus 88.4% and 50.0% for always-LG. A balanced decision tree reached 85.0% accuracy and 68.5% balanced accuracy. However, the runtime-regression selector's observed held-out runtime sum was 72.02 s, nearly the 72.35 s always-LG total and much slower than the 26.17 s always-HPF total. The tree's observed selected runtime was 38.74 s, also slower than always-HPF. These runtime sums treat five-second LG timeouts as lower bounds; the runtime selector selected timed-out LG in 11 held-out cases, so its actual total may be higher. Neither model is ready to replace the current policy.

Because SIM2D76 is degenerate under the project's current precedence builder, a separate industrial-only validation excluded its 22 LG-only patch rows. Across the four 3D deposits (125 rows), the runtime-regression selector reached 84.8% accuracy / 56.5% balanced accuracy versus 86.4% / 50.0% for always-LG. The decision tree reached 90.4% / 77.1% but its observed held-out runtime sum was 38.67 s, still above the 26.16 s always-HPF sum. This is a classification improvement, not yet a runtime-policy improvement. The industrial-only model and metrics are isolated under `results/solver_crossover_expansion_20260929_01/fit_industrial_only/`.

The raw larger-patch data, merged training snapshots, model, and metrics are isolated under `results/solver_crossover_expansion_20260929_01/`. The original `results/solver_selector_multidataset.joblib` was not overwritten. Reproduce the expansion and separate fit with:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/run_solver_crossover_expansion.py
```
