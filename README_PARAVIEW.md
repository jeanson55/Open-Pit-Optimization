# Bauxitemed ParaView Thesis Visualization

This package is generated from the verified adaptive-solver trace. It does **not**
rerun the optimization.

## Verified trace

- Block model: 120 × 120 × 26 = 240,500 blocks
- Local selected: 71,831
- Repaired selected: 85,784
- Final selected: 83,800
- Repair additions: 13,953
- Maximum repair level: 25
- Boundary members: 32,853
- Boundary additions: 2
- Boundary removals: 1,986
- Initial NPV: 31,801,369
- Final NPV: 26,556,907

## Run

From ParaView's Python environment:

```bash
pvpython paraview_bauxitemed.py --trace /path/to/bauxitemed_adaptive_trace.npz
```

This creates:

- `bauxitemed_adaptive_trace.vtp` — point-based block model with all trace arrays
- `final_pit.csv`
- `repair_cascade.csv`
- `boundary_changes.csv`
- `initial_violations.csv`
- `trace_summary.txt`

Open `bauxitemed_adaptive_trace.vtp` in ParaView. Each real block is represented by one point/vertex because the trace contains 240,500 real blocks rather than all 374,400 Cartesian cells.

## Recommended thesis views

1. **Economic block model:** color by `epv`.
2. **METIS partition:** color by `partition_id`; use categorical coloring.
3. **Initial violations:** threshold `initial_violation == 1`.
4. **Repair cascade:** threshold `repair_added == 1`, color by `repair_level`.
5. **Boundary region:** threshold `boundary_member == 1`.
6. **Boundary changes:** compare `boundary_added` and `boundary_removed`.
7. **Final pit:** threshold `final_selected == 1`, optionally color by `epv`.

For the repair figure, use `repair_level` as the scalar. The maximum recorded
repair level is 25, and the trace contains all 13,953 repair additions.

## Thesis interpretation

The visualization should be described as a spatial explanation of the experiment:
local partitioned optimization -> precedence repair -> boundary refinement -> final
feasible pit. It should not be presented as evidence of runtime superiority.
