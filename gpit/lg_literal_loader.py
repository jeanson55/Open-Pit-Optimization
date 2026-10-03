#lg_literal_loader module is part of the gPit software
#Copyright (c) 2023 gPit Developers
#Distributed under the terms of MIT License

"""
Builds the plain (values, precedence) dicts LiteralLG needs, using the
SAME graph-construction path as gpit's Pseudoflow/LG (max-flow), so the
literal LG algorithm can be validated on an IDENTICAL graph -- per the
agreed correctness hierarchy:

  primary:    LG objective    == MaxFlowBaseline objective
  required:   LG violations   == 0
  secondary:  LG block count  (vs MaxFlowBaseline block count)
  diagnostic: Jaccard(LG, MaxFlowBaseline)   -- NOT a pass/fail test

IMPORTANT (bug fixed here): an earlier version of this loader computed
its own "included" node filter using CreateExtArcs' t_z-shrink rule,
then filtered precedence arcs against that filter. But the precedence
TEMPLATE loop (matching Pseudoflow.UPL's own bounds) shrinks by a
DIFFERENT formula (ind_z+1). These two independently-derived boundary
rules do not agree pointwise, so the old filter silently dropped real
precedence arcs whose endpoints fell outside its own (wrong) filter --
confirmed directly: prec.IA1x5(pos_x=1, pos_y=0, pos_z=38, nx=75, ny=1)
produces real edges to nodes 2926-3002, all of which the old filter
excluded.

Fixed by building the graph exactly as gpit's own CreateExtArcs +
precedence-template loop do, and then reading values/precedence
DIRECTLY off the resulting graph object -- no separate re-derivation of
which nodes "should" be included. Nodes and arcs are whatever gpit's
own code actually produced, guaranteed consistent by construction.
"""

import networkx as NetX

from gpit import precedence as prec
from gpit.pseudoflow import Pseudoflow


def build_from_gpit(blockmodel_df, bmparms, EPV_column, precedence_template="1x5"):
    nx_, ny_, nz_ = bmparms["nx"], bmparms["ny"], bmparms["nz"]
    epv = blockmodel_df[EPV_column].to_numpy()

    # Build the graph EXACTLY as gpit's own solvers do -- CreateExtArcs for
    # source/sink membership, then the precedence template loop with its
    # OWN bounds (matching Pseudoflow.UPL, not an independently-derived
    # boundary formula).
    ext = Pseudoflow(blockmodel_df, bmparms)
    ext.EPV = epv
    g = NetX.DiGraph()
    g = ext.CreateExtArcs(g)

    if precedence_template in ("1x5", "1x9"):
        for ind_z in range(nz_ - 1):
            pos_z = nz_ - ind_z - 2
            for pos_y in range(ind_z + 1, ny_ - ind_z - 1):
                for pos_x in range(ind_z + 1, nx_ - ind_z - 1):
                    fn = prec.IA1x5 if precedence_template == "1x5" else prec.IA1x9
                    g = fn(pos_x, pos_y, pos_z, nx_, ny_, g)
    else:
        fn_map = {"1x5x13": prec.IA1x5x13, "1x5x21": prec.IA1x5x21, "1x1x5": prec.IA1x1x5}
        fn = fn_map[precedence_template]
        for ind_z in range(nz_ - 2):
            pos_z = nz_ - ind_z - 3
            for pos_y in range(ind_z + 2, ny_ - ind_z - 2):
                for pos_x in range(ind_z + 2, nx_ - ind_z - 2):
                    g = fn(pos_x, pos_y, pos_z, nx_, ny_, g)

    SOURCE = 0
    SINK = ext.sink

    # Nodes/arcs come from whatever the graph actually contains -- no
    # independent recomputation of "which nodes should be included".
    real_nodes = set(g.nodes()) - {SOURCE, SINK}
    values = {i: float(epv[i - 1]) for i in real_nodes}

    precedence = {i: [] for i in real_nodes}
    self_loops = 0
    for a, b in g.edges():
        if a == b:
            self_loops += 1  # degenerate artifact (e.g. ny=1 aliasing in IA1x5's
            continue          # unbounded-checked formula) -- never a real constraint
        if a in real_nodes and b in real_nodes:
            precedence[a].append(b)

    diagnostics = {
        "graph_nodes": g.number_of_nodes(),
        "graph_edges": g.number_of_edges(),
        "real_nodes": len(real_nodes),
        "precedence_arcs": sum(len(v) for v in precedence.values()),
        "self_loops_dropped": self_loops,
    }
    if self_loops:
        import warnings
        warnings.warn(
            f"{self_loops} self-loop arc(s) encountered (node requires itself) -- "
            f"this is a degenerate artifact, not a real constraint, and usually "
            f"signals invalid model geometry for this precedence template (e.g. "
            f"ny=1 or nx/ny too small relative to the template's offsets)."
        )

    return values, precedence, diagnostics