"""
Hybrid LG + HPF Optimizer

Stage 1:
    LiteralLG solves maximum closure directly.

Stage 2:
    Original gPit HPF/Pseudoflow solves the same maximum-closure
    problem using the validated low-level hpf() interface.

Stage 3:
    Reconcile the two solutions using both objective and membership
    diagnostics.
"""

import time
import numpy as np

# Initialise gPit's headless environment when its helper is available.
# A minimal fallback keeps this module usable from the Pit_Optimization
# checkout, which does not include the sibling gpit1 bootstrap script.
try:
    import bootstrap_headless  # noqa: F401
except ImportError:
    import gpit
    gpit.LOGPATH = ""

from gpit.lg_literal import LiteralLG
from gpit.pseudoflow import Pseudoflow


class HybridOptimizer:

    def __init__(
        self,
        values,
        precedence,
        blockmodel,
        bmParms
    ):

        self.values = values
        self.precedence = precedence

        self.blockmodel = blockmodel
        self.bmParms = bmParms

        self.lg_solution = None
        self.hpf_solution = None

        self.lg_npv = None
        self.hpf_npv = None

        self.lg_time_s = None
        self.hpf_time_s = None

    # ============================================================
    # STAGE 1 — LITERAL LG
    # ============================================================

    def solve_lg(self):

        print("\n[HYBRID] Running LiteralLG...")

        t0 = time.perf_counter()

        lg = LiteralLG(
            values=self.values,
            precedence=self.precedence
        )

        lg.solve()

        self.lg_solution = lg.closure()

        self.lg_npv = float(lg.objective())

        self.lg_time_s = time.perf_counter() - t0

        print(
            f"[HYBRID] LiteralLG blocks: "
            f"{len(self.lg_solution):,}"
        )

        print(
            f"[HYBRID] LG objective: "
            f"{self.lg_npv:,.6f}"
        )

        print(
            f"[HYBRID] LG time: "
            f"{self.lg_time_s:.3f}s"
        )

        return self.lg_solution

    # ============================================================
    # STAGE 2 — HPF
    # ============================================================

    # ============================================================
    # STAGE 2 — HPF
    # ============================================================

    def solve_hpf(
            self,
            EPV_column="EPV",
            precedence="1x5"
    ):

        print("\n[HYBRID] Running HPF...")

        t0 = time.perf_counter()

        # --------------------------------------------------------
        # Empty closure handling.
        #
        # If LiteralLG returns no blocks, the maximum closure is
        # the empty set. There is no reason to call HPF because
        # the graph may not contain a valid source node.
        # --------------------------------------------------------

        if (
                self.lg_npv is not None
                and abs(self.lg_npv) < 1e-12
        ):
            self.hpf_solution = np.zeros(
                len(self.blockmodel),
                dtype=np.uint8
            )

            self.hpf_npv = 0.0

            self.hpf_time_s = (
                    time.perf_counter() - t0
            )

            print(
                "[HYBRID] HPF skipped: "
                "empty optimal closure"
            )

            print(
                "[HYBRID] HPF blocks: 0"
            )

            print(
                "[HYBRID] HPF objective: 0.000000"
            )

            print(
                f"[HYBRID] HPF time: "
                f"{self.hpf_time_s:.3f}s"
            )

            return self.hpf_solution

        # --------------------------------------------------------
        # Build HPF graph.
        # --------------------------------------------------------

        pf = Pseudoflow(
            self.blockmodel,
            self.bmParms
        )

        pf.EPV = np.asarray(
            self.blockmodel[EPV_column],
            dtype=np.float64
        )

        import networkx as nx

        graph = pf.CreateExtArcs(
            nx.DiGraph()
        )

        # --------------------------------------------------------
        # Add precedence arcs.
        # --------------------------------------------------------

        from gpit import precedence as prec

        nx_ = self.bmParms["nx"]
        ny_ = self.bmParms["ny"]
        nz_ = self.bmParms["nz"]

        for ind_z in range(nz_ - 1):

            pos_z = nz_ - ind_z - 2

            for pos_y in range(
                    ind_z + 1,
                    ny_ - ind_z - 1
            ):

                for pos_x in range(
                        ind_z + 1,
                        nx_ - ind_z - 1
                ):
                    graph = prec.IA1x5(
                        pos_x,
                        pos_y,
                        pos_z,
                        nx_,
                        ny_,
                        graph
                    )

        # --------------------------------------------------------
        # Safety check.
        # --------------------------------------------------------

        sink = (
                nx_ * ny_ * nz_
                + 1
        )

        if (
                0 not in graph.nodes
                or sink not in graph.nodes
        ):
            print(
                "[HYBRID] HPF skipped: "
                "invalid source/sink graph"
            )

            self.hpf_solution = np.zeros(
                len(self.blockmodel),
                dtype=np.uint8
            )

            self.hpf_npv = 0.0

            self.hpf_time_s = (
                    time.perf_counter() - t0
            )

            return self.hpf_solution

        # --------------------------------------------------------
        # Run original gPit HPF.
        # --------------------------------------------------------

        import pseudoflow as pf_solver

        _, cuts, _ = pf_solver.hpf(
            graph,
            0,
            sink,
            const_cap="const",
            mult_cap="mult",
            lambdaRange=[0],
            roundNegativeCapacity=False
        )

        # --------------------------------------------------------
        # Convert cut to block vector.
        # --------------------------------------------------------

        self.hpf_solution = np.zeros(
            len(self.blockmodel),
            dtype=np.uint8
        )

        for node, side in cuts.items():

            if (
                    side == [1]
                    and node != 0
            ):

                block_index = int(node) - 1

                if (
                        0 <= block_index
                        < len(self.hpf_solution)
                ):
                    self.hpf_solution[
                        block_index
                    ] = 1

        # --------------------------------------------------------
        # Objective.
        # --------------------------------------------------------

        epv = np.asarray(
            self.blockmodel[EPV_column],
            dtype=np.float64
        )

        self.hpf_npv = float(
            (
                    self.hpf_solution
                    * epv
            ).sum()
        )

        self.hpf_time_s = (
                time.perf_counter() - t0
        )

        print(
            f"[HYBRID] HPF blocks: "
            f"{int(self.hpf_solution.sum()):,}"
        )

        print(
            f"[HYBRID] HPF objective: "
            f"{self.hpf_npv:,.6f}"
        )

        print(
            f"[HYBRID] HPF time: "
            f"{self.hpf_time_s:.3f}s"
        )

        return self.hpf_solution

    # ============================================================
    # STAGE 3 — RECONCILIATION
    # ============================================================

    def compare(self):

        if self.lg_solution is None:
            raise RuntimeError(
                "Run solve_lg first"
            )

        if self.hpf_solution is None:
            raise RuntimeError(
                "Run solve_hpf first"
            )

        epv = np.asarray(
            self.blockmodel["EPV"],
            dtype=np.float64
        )

        # --------------------------------------------------------
        # Convert LiteralLG closure to a 0/1 vector.
        # --------------------------------------------------------

        lg_vector = np.zeros(
            len(epv),
            dtype=np.uint8
        )

        for node in self.lg_solution:

            index = int(node) - 1

            if (
                0 <= index
                < len(lg_vector)
            ):

                lg_vector[index] = 1

        hpf_vector = self.hpf_solution

        # --------------------------------------------------------
        # Membership.
        # --------------------------------------------------------

        diff = (
            lg_vector
            != hpf_vector
        )

        all_diff = int(diff.sum())

        intersection = int(
            (
                (lg_vector == 1)
                &
                (hpf_vector == 1)
            ).sum()
        )

        union = int(
            (
                (lg_vector == 1)
                |
                (hpf_vector == 1)
            ).sum()
        )

        jaccard = (
            intersection / union
            if union
            else 1.0
        )

        # --------------------------------------------------------
        # Economic categories.
        # --------------------------------------------------------

        nonzero_mask = epv != 0
        positive_mask = epv > 0
        negative_mask = epv < 0
        zero_mask = epv == 0

        def category(mask):

            lg = lg_vector & mask
            hp = hpf_vector & mask

            lg_only = (
                (lg == 1)
                & (hp == 0)
            )

            hp_only = (
                (hp == 1)
                & (lg == 0)
            )

            category_diff = int(
                (lg != hp).sum()
            )

            category_union = int(
                (
                    (lg == 1)
                    |
                    (hp == 1)
                ).sum()
            )

            category_intersection = int(
                (
                    (lg == 1)
                    &
                    (hp == 1)
                ).sum()
            )

            category_jaccard = (
                category_intersection
                / category_union
                if category_union
                else 1.0
            )

            return {
                "lg_only": int(lg_only.sum()),
                "hpf_only": int(hp_only.sum()),
                "diff": category_diff,
                "jaccard": category_jaccard,
                "lg_only_value": float(
                    epv[lg_only].sum()
                ),
                "hpf_only_value": float(
                    epv[hp_only].sum()
                ),
            }

        nonzero = category(nonzero_mask)
        positive = category(positive_mask)
        negative = category(negative_mask)
        zero = category(zero_mask)

        # --------------------------------------------------------
        # Objective agreement.
        # --------------------------------------------------------

        delta_npv = (
            self.lg_npv
            - self.hpf_npv
        )

        objective_ok = (
            abs(delta_npv)
            < 1e-9
        )

        exact_membership = (
            all_diff == 0
        )

        economically_neutral = (
            abs(
                nonzero["lg_only_value"]
            ) < 1e-9
            and
            abs(
                nonzero["hpf_only_value"]
            ) < 1e-9
        )

        # --------------------------------------------------------
        # Status.
        # --------------------------------------------------------

        if objective_ok and exact_membership:

            status = (
                "EXACT ECONOMIC AND "
                "MEMBERSHIP AGREEMENT"
            )

        elif (
            objective_ok
            and economically_neutral
        ):

            status = (
                "EXACT ECONOMIC AGREEMENT "
                "(DEGENERATE OPTIMAL CLOSURE)"
            )

        elif objective_ok:

            status = (
                "OBJECTIVE AGREEMENT WITH "
                "ECONOMIC MEMBERSHIP DIFFERENCES"
            )

        else:

            status = (
                "OBJECTIVE DISAGREEMENT"
            )

        # --------------------------------------------------------
        # Report.
        # --------------------------------------------------------

        print("\n[HYBRID] Reconciliation")
        print("=" * 78)

        print(
            f"LiteralLG blocks : "
            f"{int(lg_vector.sum()):,}"
        )

        print(
            f"HPF blocks       : "
            f"{int(hpf_vector.sum()):,}"
        )

        print(
            f"LiteralLG NPV    : "
            f"{self.lg_npv:,.6f}"
        )

        print(
            f"HPF NPV          : "
            f"{self.hpf_npv:,.6f}"
        )

        print(
            f"Delta NPV        : "
            f"{delta_npv:.12f}"
        )

        print(
            f"All-block diff    : "
            f"{all_diff:,}"
        )

        print(
            f"Jaccard           : "
            f"{jaccard:.6f}"
        )

        print("\nEconomic membership")
        print("-" * 78)

        for name, data in [
            ("Nonzero EPV", nonzero),
            ("Positive EPV", positive),
            ("Negative EPV", negative),
            ("Zero EPV", zero),
        ]:

            print(
                f"{name:<18}"
                f"LG-only={data['lg_only']:>10,} "
                f"HPF-only={data['hpf_only']:>10,} "
                f"Diff={data['diff']:>10,} "
                f"J={data['jaccard']:.6f}"
            )

        print("\nFINAL STATUS")
        print("-" * 78)
        print(status)

        return {
            "lg_blocks": int(
                lg_vector.sum()
            ),

            "hpf_blocks": int(
                hpf_vector.sum()
            ),

            "lg_npv": self.lg_npv,
            "hpf_npv": self.hpf_npv,
            "delta_npv": delta_npv,

            "jaccard": jaccard,
            "all_diff": all_diff,

            "nonzero_diff":
                nonzero["diff"],

            "positive_diff":
                positive["diff"],

            "negative_diff":
                negative["diff"],

            "zero_diff":
                zero["diff"],

            "nonzero_lg_only_value":
                nonzero["lg_only_value"],

            "nonzero_hpf_only_value":
                nonzero["hpf_only_value"],

            "zero_lg_only":
                zero["lg_only"],

            "lg_time_s":
                self.lg_time_s,

            "hpf_time_s":
                self.hpf_time_s,

            "speedup": (
                self.lg_time_s
                / self.hpf_time_s
                if self.hpf_time_s
                else float("nan")
            ),

            "objective_ok":
                objective_ok,

            "exact_membership":
                exact_membership,

            "economically_neutral":
                economically_neutral,

            "status":
                status,
        }