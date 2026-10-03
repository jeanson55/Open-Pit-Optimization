# Open-Pit-Optimization

This repository contains the implementation, benchmark block-model datasets, experiment outputs, and thesis-supporting documentation for the MSc research on partitioned open-pit optimization using maximum closure, LiteralLG, and HPF/Pseudoflow.

## Project contents

- `graph/`, `gpit/`, `optimization/`, and `partition/`: graph construction, solver implementations, feasibility repair, partitioners, refinement, and metrics.
- `experiments/`: benchmark drivers, verification scripts, research notes, synthesis, and thesis objective status.
- `data/`: block-model datasets and economic parameters used by the experiments. The BauxiteMed file is present at both `data/bauxitemed.dat` and `data/BauxiteMed/bauxitemed.dat` because existing drivers use both paths.
- `results/`: recorded JSON/CSV results, logs, model artifacts, ParaView trace files, and figures used in the analysis.
- `visualization/`: scripts and figures for thesis visuals.
- `paraview_bauxitemed_output/`: a structured-grid ParaView export; the point-based trace export is also stored under `results/paraview_bauxitemed/`.

## Reproducibility notes

The recorded experiment environment was Ubuntu under WSL with Python 3.12.3. Recorded core package versions include NetworkX 3.6.1, SciPy 1.18.1, scikit-learn 1.9.1, and pymetis 2025.2.2. See `experiments/THESIS_RESULTS_SYNTHESIS.md`, `experiments/PARTITION_GRID_RESEARCH.md`, and the thesis draft for experiment-specific context and caveats.

The repository includes the benchmark datasets and saved results, but no single dependency lock file currently captures every experiment environment. Install the recorded versions in the WSL virtual environment before reproducing partitioning or solver runs. Some experiment scripts use optional libraries such as SciPy, scikit-learn, and pymetis.

Run experiment commands from the repository root. For example, the BauxiteMed partitioning comparison is driven by `experiments/experiment_e_partitioning.py`; its defaults and available methods are documented in that script's argument help and `experiments/EXPERIMENT_E.md`.

## Important interpretation

Partitioned/refined NPV results are feasible local solutions and are not automatically global optima. The saved global HPF/LiteralLG benchmark results provide exact maximum-closure references only for the datasets and runs where exact solver agreement was recorded. Consult the experiment notes before comparing metrics across different partition counts or refinement modes.

