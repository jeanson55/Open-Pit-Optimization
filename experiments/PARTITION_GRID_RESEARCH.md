# Paired partition count study

This study extends the partition-selection pilot with paired full-model runs. For each of four block models, Spatial and METIS assignments were independently solved at 4, 8, and 12 partitions using the same eight-part local HPF policy and add-only global feasibility repair. The completed V3.6 Spatial and V3.5 METIS eight-partition runs were reused; four- and twelve-partition runs were newly executed. Every completed result has zero final precedence violations.

## Paired NPV results

| Model | Partitions | Spatial NPV | METIS NPV | Higher result |
|---|---:|---:|---:|---|
| bauxitemed | 4 | 29,065,667 | 28,958,649 | Spatial |
| bauxitemed | 8 | 28,652,908 | 28,251,171 | Spatial |
| bauxitemed | 12 | 26,655,857 | 26,055,755 | Spatial |
| cucase | 4 | 19,435,800 | 19,423,683 | Spatial |
| cucase | 8 | 19,359,730 | 19,400,896 | METIS |
| cucase | 12 | 19,356,103 | 19,370,695 | METIS |
| cupipe | 4 | 138,208,894 | 76,919,897 | Spatial |
| cupipe | 8 | 57,815,484 | 58,803,806 | METIS |
| cupipe | 12 | 91,613,358 | 61,158,175 | Spatial |
| mclaughlingeo | 4 | 1,184,853,440 | 1,189,699,445 | METIS |
| mclaughlingeo | 8 | 1,179,217,403 | 1,174,540,950 | Spatial |
| mclaughlingeo | 12 | 1,183,559,746 | 1,163,395,262 | Spatial |

The better partitioner changes with deposit and partition count. Four partitions gave the largest repaired NPV on three of the four models; McLaughlinGeo's best paired result was four-part METIS. CUPIPE is especially sensitive: four-part Spatial produced 138.2 million NPV, versus 57.8 million at eight-part Spatial. These are measured outcomes for the tested settings, not a general rule that four partitions are optimal.

## Feasibility repair and boundary refinement

The best observed add-only method/count per deposit was rerun with the existing dynamic boundary refiner. The best-setting choice is made from observed outcomes for this diagnostic comparison, so it is not a prospective selector result.

| Model | Best setting | Add-only NPV | Boundary-refined NPV | Gain | Boundary nodes | Refinement time | Final violations |
|---|---|---:|---:|---:|---:|---:|---:|
| bauxitemed | Spatial, 4 | 29,065,667 | 29,250,744 | 185,077 (0.637%) | 11,109 | 4.43 s | 0 |
| cucase | Spatial, 4 | 19,435,800 | 19,441,249 | 5,449 (0.028%) | 30,254 | 17.27 s | 0 |
| cupipe | Spatial, 4 | 138,208,894 | 139,888,165 | 1,679,271 (1.215%) | 139,883 | 31.22 s | 0 |
| mclaughlingeo | METIS, 4 | 1,189,699,445 | 1,189,971,620 | 272,175 (0.023%) | 34,018 | 19.91 s | 0 |

The refiner improved each selected add-only incumbent while preserving feasibility. It is conditional local reoptimization, not proof of global optimality. Against the recorded standalone global HPF NPVs (29,690,715; 19,444,156; 142,915,803; and 1,190,881,614 respectively), the refined results remain lower by approximately 1.481%, 0.015%, 2.119%, and 0.076%.

## V3.6 grouped validation

`experiments/train_hybrid_v36_grid_selector.py` uses the 24 method outcomes as 12 matched settings, with dataset identity grouped in leave-one-deposit-out validation. Features are recomputed from the corresponding pre-solve Spatial and METIS partitions. The pilot model and its results are separate from the original V3.6 selector.

| Selector/baseline | Held-out pair accuracy | Mean relative NPV regret |
|---|---:|---:|
| V3.6 composite | 58.3% (7/12) | 3.890% |
| V3.6 ML probability only | 66.7% (8/12) | not the primary fitted-score output |
| Always Spatial | 66.7% (8/12) | 0.198% |
| Always METIS | 33.3% (4/12) | 6.981% |

The composite rule does not beat the simple Spatial baseline on accuracy or regret. Its 3.89% mean regret is inflated by the missed CUPIPE four-part Spatial case. The ML-only result ties the accuracy baseline but has not demonstrated lower economic regret. Therefore `results/hybrid_v36_grid_selector.pkl` remains a research artifact and is not wired into the runner; `results/hybrid_v36_selector.pkl` remains unchanged.

There are 12 matched settings but still only four independent deposits. Counts on the same deposit are correlated and do not replace validation on additional geological models. The recorded eight-partition result files predate the new partition telemetry; the trainer recomputed their candidate features from the deterministic partitioners.

## Experiment E multi-count comparison on bauxitemed

Experiment E was also repeated at 4 and 12 partitions, retaining its Spatial, spectral, METIS, and spatial/graph hybrid candidates. All methods used the same local NetworkX maximum-closure engine within this experiment; these values are not mixed into the V3.6 training outcomes, whose full runs use the adaptive Pseudoflow backend.

| Partitions | Method | Balance (max/mean) | Cut fraction | Initial violations | Repair additions | Final NPV | Partition time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4 | Spatial | 1.000 | 0.831% | 201 | 993 | 29,065,667 | 1.75 s |
| 4 | Spectral | 1.000 | 5.528% | 2,267 | 19,756 | 15,715,341 | 73.73 s |
| 4 | METIS | 1.000 | 0.950% | 336 | 1,299 | 28,958,649 | 4.35 s |
| 4 | Hybrid spatial/graph | 1.000 | 0.831% | 201 | 993 | 29,065,667 | 12.91 s |
| 12 | Spatial | 1.557 | 2.079% | 1,366 | 7,931 | 26,649,405 | 1.92 s |
| 12 | Spectral | 1.000 | 7.465% | 3,618 | 24,123 | 13,510,593 | 180.68 s |
| 12 | METIS | 1.002 | 2.548% | 1,696 | 7,761 | 26,057,222 | 6.41 s |
| 12 | Hybrid spatial/graph | 1.557 | 2.079% | 1,366 | 7,931 | 26,649,405 | 11.77 s |

Every final pit was feasible. Spectral gave very even part sizes but much larger cuts, repair cascades, and lower NPV. The current hybrid generator returned the exact Spatial labels at both counts and paid extra partitioning time without improving the solution. This supports using cut/repair/economic outcomes alongside balance when evaluating partition methods. The complete nine-part Experiment E results remain in `experiments/EXPERIMENT_E.md`; the new four- and twelve-part results are in `results/experiment_e_partitioning_k4.json`, `results/experiment_e_partitioning_k12.json`, and `results/experiment_e_partition_counts_summary.json`.

## Four-deposit spectral extension at four partitions

The multi-method study was extended beyond bauxitemed by adding a genuine
spectral partition option to `experiment_adaptive_solver.py`. The resumable
runner `run_spectral_partition_extension.py` executed spectral, local HPF,
and add-only global feasibility repair for all four deposits at k=4. This
uses the same adaptive solver path as the matched Spatial/METIS k=4 results;
the old Experiment E NetworkX closure figures are not used in this comparison.

| Deposit | Spectral NPV | Best matched Spatial/METIS NPV | Spectral regret | Cut fraction | Balance (max/mean) | Initial violations | Final violations | Partition seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bauxitemed | 15,715,341 | 29,065,667 (Spatial) | 45.93% | 5.528% | 1.000 | 8,944 | 0 | 71.03 |
| cucase | 19,136,484 | 19,435,800 (Spatial) | 1.54% | 2.610% | 1.000 | 13,448 | 0 | 341.36 |
| cupipe | 81,074,506 | 138,208,894 (Spatial) | 41.34% | 2.995% | 1.000 | 16,612 | 0 | 301.93 |
| mclaughlingeo | 1,134,417,979 | 1,189,699,445 (METIS) | 4.65% | 3.168% | 1.000 | 27,342 | 0 | 410.64 |

Spectral labels were essentially exactly balanced, but they had higher cut
fractions and lower repaired NPV than the better matched candidate on all
four deposits. It beat METIS on CUPIPE by about 5.4%, but still trailed
Spatial by 41.3%. Partition generation took 71–411 seconds, versus 1.87–9.77
seconds for Spatial and 7.38–40.10 seconds for METIS. The approximate LOBPCG
embedding emitted convergence-tolerance warnings for cucase, CUPIPE, and
McLaughlinGeo; it still returned labels and all final pits had zero
violations. The current result therefore does not support routing these
industrial-scale instances through spectral partitioning without further
algorithm/runtime work.

A transparent leave-one-deposit-out baseline selected the most frequent
winner among the other three deposits. It selected Spatial in all four
holdouts, scored 3/4 exact winner selections (75%), and had 0.102% mean
relative NPV regret, identical to always-Spatial. This is only a baseline
comparison, not ML training: four independent deposits at one partition
count are too few to support a new fitted selector. Full metrics are in
`results/spectral_partition_extension_comparison.json`; the four individual
adaptive runs and summary are in `results/partition_grid_<dataset>_spectral_k4_add.json`
and `results/spectral_partition_extension_summary.csv`.

Reproduce in Ubuntu from the project root:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps:../lg:. ../lg/.venv-linux/bin/python experiments/run_spectral_partition_extension.py --partitions 4
PYTHONPATH=../lg:. ../lg/.venv-linux/bin/python experiments/analyze_spectral_partition_extension.py
```

## V3.7 joint method/count selector pilot

`train_hybrid_v37_joint_selector.py` combines the measured Spatial/METIS
outcomes at k=4, 8, and 12 with the new spectral k=4 outcomes. For Spatial
and METIS, pre-solve partition features are deterministically regenerated,
including the historical k=8 cases whose old run files did not contain cut
telemetry. Spectral features use the saved cut/balance telemetry; spectral
labels were not retained, so economic boundary exposure is omitted from the
common feature schema. The model is a regularized logistic classifier with
deposit identity excluded and leave-one-deposit-out validation.

The 28 measured candidates cover seven settings per deposit. The joint model
selected Spatial k=4 on all four held-out deposits, reaching 3/4 exact winner
accuracy (75%) and 0.102% mean relative NPV regret. This exactly matches a
training-selected Spatial partition count and fixed Spatial k=4. It misses
the small METIS advantage on McLaughlinGeo (0.407% regret). A fitted pilot
was saved for analysis, but it is not integrated into the runner because it
does not beat the simple baseline. The mean selected-run time (127.4 s) is
not a fair prospective selector runtime: it sums only the selected method's
partition/solve/repair time and excludes the cost to generate candidates and
their features. Generating spectral candidates alone takes 71–411 seconds on
these deposits.

This grouped validation adds candidate methods/counts, not independent
geological samples: there are still only four deposit groups. It is evidence
against deploying the current model and indicates that more independent
datasets or a better pre-partition predictor are needed. The complete folds,
fixed baselines, and limitations are in
`results/hybrid_v37_joint_selector_metrics.json`; the feature table and
isolated pilot are in `results/hybrid_v37_joint_training_data.csv` and
`results/hybrid_v37_joint_selector_pilot.pkl`.

Reproduce from the project root in Ubuntu:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps:../lg:. ../lg/.venv-linux/bin/python experiments/train_hybrid_v37_joint_selector.py
```

## Reproduction and artifacts

From the project root in Ubuntu:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/run_hybrid_v36_partition_grid.py
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/train_hybrid_v36_grid_selector.py
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/run_best_partition_boundary_refinement.py
PYTHONPATH=/home/jeanson/pit-exp-deps ../gpit1/lg/.venv-linux/bin/python experiments/run_experiment_e_partition_counts.py
```

The sweep is resumable and preserves completed per-case JSON files. Its main artifacts are:

- `results/hybrid_v36_partition_grid_outcomes.csv` and `.json`
- `results/hybrid_v36_grid_training_data.csv`
- `results/hybrid_v36_grid_training_metrics.json`
- `results/hybrid_v36_grid_selector.pkl`
- `results/hybrid_v36_best_boundary_summary.csv` and `.json`
- `results/partition_grid_<dataset>_<method>_k<count>_<add|boundary>.json`

The adaptive experiment runner now also records partition quality (cut fraction, cut arc count, balance, and minimum/maximum part sizes) separately from actual pit feasibility violations.

## Boundary-refinement small-graph verification

`verify_boundary_refinement.py` generates seeded precedence DAGs with 4–10
nodes. For each graph it compares the HPF local closure result, including
compatible required/forbidden pins, against exhaustive enumeration of every
node subset. It then starts from an arbitrary potentially infeasible pit,
applies add-only repair and boundary refinement, and checks final slope
feasibility and that refinement does not reduce the repaired incumbent NPV.
The 100-case Ubuntu run passed all 100 exhaustive closure comparisons with
zero objective error and all 100 feasibility/non-regression checks. This is
deterministic small-instance verification evidence; it is not a formal proof
for all graphs and says nothing by itself about industrial-scale runtime.

Reproduce from the project root in Ubuntu:

```bash
PYTHONPATH=/home/jeanson/pit-exp-deps:../lg:. ../lg/.venv-linux/bin/python experiments/verify_boundary_refinement.py --cases 100
```
