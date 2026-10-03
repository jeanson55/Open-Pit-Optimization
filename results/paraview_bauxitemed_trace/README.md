# Bauxitemed adaptive-trace figures

Generated from `results/bauxitemed_adaptive_trace.npz`; the optimization was not rerun.

## ParaView source

Open `bauxitemed_adaptive_trace.vtp`. It contains one point and vertex for each of
the 240,500 real blocks, with coordinates and the arrays `epv`, `partition_id`,
`local_selected`, `repaired_selected`, `final_selected`, `initial_violation`,
`repair_added`, `repair_level`, `boundary_member`, `boundary_added`, and
`boundary_removed`.

For a uniform block display, apply **Glyph** to the data and use:

| Setting | Value |
|---|---|
| Glyph Type | Box |
| X / Y / Z Length | 1 / 1 / 1 |
| Scale Array | No scale array |
| Scale Factor | 0.9 |
| Glyph Mode | Every Nth Point |
| Stride | 1 |
| Representation | Surface |
| Colour | `partition_id` (categorical) |

To isolate another view, threshold the relevant binary array at 1. Colour the
repair additions by `repair_level`; colour boundary edits by `boundary_added` or
`boundary_removed`. The categorical partition ID is a colour only, never a scale.

## Included images

- `figures/bauxitemed_full_block_model_by_partition.png`: all 240,500 real block
  locations, shown as a partition-coloured point view matching the current
  ParaView import.
- `figures/bauxitemed_full_block_model_isometric.png`: exposed unit-block faces
  for the complete model, coloured by partition.
- `figures/bauxitemed_final_pit_isometric_blocks.png`: exposed unit faces of the
  final selected blocks, categorical partition colour.
- `figures/bauxitemed_trace_stages.png`: local, repaired and boundary-refined
  selection stages.
- `figures/bauxitemed_final_pit_by_partition.png`: final membership by partition.
- `figures/bauxitemed_repair_cascade.png`: recursive repair additions by level.
- `figures/bauxitemed_initial_violations.png`: pre-repair violating blocks.
- `figures/bauxitemed_boundary_changes.png`: boundary additions and removals.

The figures describe this nine-part METIS adaptive trace: 71,831 local selections,
85,784 after repair, 83,800 final selections; 3,137 initial violating blocks,
13,953 repair additions, maximum cascade level 25, two boundary additions and
1,986 removals; recorded final EPV sum 26,556,907. This is a distinct run from
the spatial nine-part Experiment E and the unrestricted exact global pit. Do not
combine its value or membership with those other outputs. The z coordinate is
bench index, not surveyed topography.

Rebuild with `visualization/build_bauxitemed_thesis_figures.py` using Python 3.12,
ParaView 6.2.0's bundled VTK/Matplotlib libraries, and the recorded NPZ trace.
