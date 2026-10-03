#LG-literal module is part of the gPit software (https://github.com/GuilhermeMene/gPit)
#Copyright (c) 2023 gPit Developers
#Distributed under the terms of MIT License (https://github.com/GuilhermeMene/gPit/blob/main/LICENSE)

"""
Literal Lerchs-Grossmann (1965) maximum-closure algorithm.

Implements the actual merger/pruning tree procedure -- normalized tree,
strong/weak branches, p-edges/m-edges, mass -- as described in:

  Muir, D.C.W. (2005) "Pseudoflow, New Life for Lerchs-Grossmann Pit
  Optimisation", Orebody Modelling and Strategic Mine Planning, Spectrum
  Series No 14, pp 97-104. The merger/pruning steps below are a direct
  transcription of that paper's "LERCHS-GROSSMANN ALGORITHM" section,
  which itself derives from Lerchs, H. and Grossmann, I.F. (1965)
  "Optimum design of open-pit mines", Transactions CIM 68, 17-24.

  Cross-checked against Underwood, R. & Tolwinski, B. (1998) "A
  mathematical programming viewpoint for solving the ultimate pit
  problem", EJOR 107(1), 96-107 -- an independent dual-simplex
  derivation of the identical procedure. Their paper confirms LG's
  specific convention (always sever the (rs,root) arc on merger; fix up
  any resulting negative flows via "Normalize Tree") which is the same
  operation as Muir's pruning step, described in flow terms instead of
  mass/strong-weak terms.

THIS IS NOT A MAX-FLOW/MIN-CUT IMPLEMENTATION. There is no source, sink,
or flow network anywhere in this file -- only a node-weighted directed
graph (mass = economic value) and a rooted tree restructured by repeated
merge+prune operations until no "weak-over-strong" arc remains. See
gpit/lg.py for the max-flow baseline this is explicitly NOT meant to be.

The implementation favors an auditable tree representation over the
incremental ancestor/first-child/next-sibling structures used in production
(see Muir 2024), but indexes merger arcs and normalization violations so it
does not rescan the entire graph after every local change. Branch masses are
still recomputed after a merge. Correctness remains the first requirement.
"""

import time
import heapq

from dataclasses import dataclass, field

ROOT = 0  # sentinel for the dummy root x0. Real node ids must never equal this.


@dataclass
class LGNode:
    parent: object              # ROOT if this node is a branch root
    children: set = field(default_factory=set)
    value: float = 0.0          # this block's own EPV -- constant, never changes
    mass: float = 0.0           # mass supported by this node's subtree (recomputed)
    edge_type: str = 'p'        # 'p' (downward/away-from-root) or 'm' (upward/toward-root)
    strong: bool = False        # branch-level strong/weak status (inherited from branch root)


class LiteralLG:
    """
    values: dict[node_id] -> float               block economic values
    precedence: dict[node_id] -> list[node_id]    node REQUIRES these nodes
                                                    (the ORIGINAL graph -- never mutated)
    """

    def __init__(self, values: dict, precedence: dict, debug: bool = False):
        if ROOT in values:
            raise ValueError(f"node id {ROOT} is reserved for the dummy root x0")

        self.values = dict(values)
        self.precedence = {k: list(v) for k, v in precedence.items()}  # deep-ish copy; never mutated after this
        self.debug = debug

        self.nodes = {
            v: LGNode(parent=ROOT, value=val, edge_type='p')
            for v, val in self.values.items()
        }

        self.iterations = 0
        self.merger_count = 0
        self.prune_count = 0

        self._recompute_masses_and_strength()
        self._node_order = {node: i for i, node in enumerate(self.nodes)}
        self._build_merger_index()

    def _log(self, msg):
        if self.debug:
            print(msg)

    def _build_merger_index(self):
        """Index weak-over-strong arcs in the exact legacy scan order."""
        self._merger_arcs = []
        self._outgoing_arcs = {node: [] for node in self.nodes}
        self._incoming_arcs = {node: [] for node in self.nodes}
        for source, required in self.precedence.items():
            if source not in self.nodes:
                continue
            for target in required:
                if target not in self.nodes:
                    continue
                arc_id = len(self._merger_arcs)
                self._merger_arcs.append((source, target))
                self._outgoing_arcs[source].append(arc_id)
                self._incoming_arcs[target].append(arc_id)

        self._candidate_heap = []
        self._candidate_versions = [0] * len(self._merger_arcs)
        self._active_candidates = set()
        for arc_id in range(len(self._merger_arcs)):
            self._refresh_merger_arc(arc_id)

    def _refresh_merger_arc(self, arc_id):
        """Update one indexed arc; heap order matches source/target scan order."""
        self._candidate_versions[arc_id] += 1
        version = self._candidate_versions[arc_id]
        source, target = self._merger_arcs[arc_id]
        if self.nodes[source].strong and not self.nodes[target].strong:
            self._active_candidates.add(arc_id)
            heapq.heappush(self._candidate_heap, (arc_id, version))
        else:
            self._active_candidates.discard(arc_id)

    def _refresh_node_merger_arcs(self, node):
        incident = set(self._outgoing_arcs[node])
        incident.update(self._incoming_arcs[node])
        for arc_id in incident:
            self._refresh_merger_arc(arc_id)

    def _set_branch_strength(self, root, strong):
        """Assign a branch status and refresh only arcs whose endpoint changed."""
        stack = [root]
        while stack:
            node = stack.pop()
            current = self.nodes[node]
            if current.strong != strong:
                current.strong = strong
                self._refresh_node_merger_arcs(node)
            stack.extend(current.children)

    @staticmethod
    def _locally_strong(node):
        return ((node.edge_type == 'p' and node.mass > 0)
                or (node.edge_type == 'm' and node.mass <= 0))

    # ---------------------------------------------------------------
    # Tree queries
    # ---------------------------------------------------------------

    def _root_of(self, v):
        while self.nodes[v].parent != ROOT:
            v = self.nodes[v].parent
        return v

    def _path_to_root(self, v):
        """[v, parent(v), parent(parent(v)), ..., branch_root]. Excludes ROOT itself."""
        path = [v]
        while self.nodes[path[-1]].parent != ROOT:
            path.append(self.nodes[path[-1]].parent)
        return path

    # ---------------------------------------------------------------
    # Full O(n) mass + strong/weak recompute
    # ---------------------------------------------------------------

    def _recompute_masses_and_strength(self):
        # Post-order traversal of the forest hanging off ROOT's children,
        # via an explicit stack (no recursion -- keeps this safe for deep
        # chains on larger models).
        order = []
        visited = set()

        root_children = [v for v, n in self.nodes.items() if n.parent == ROOT]

        for r in root_children:
            stack = [(r, False)]

            while stack:
                node_id, processed = stack.pop()

                if not processed:
                    if node_id in visited:
                        raise RuntimeError(
                            f"INVALID LG TREE: node {node_id} "
                            f"visited more than once during mass traversal"
                        )

                    visited.add(node_id)

                if len(visited) > len(self.nodes):
                    raise RuntimeError(
                        "INVALID LG TREE: traversal exceeded node count"
                    )

                if processed:
                    order.append(node_id)
                else:
                    stack.append((node_id, True))
                    for c in self.nodes[node_id].children:
                        stack.append((c, False))

        for v in order:
            n = self.nodes[v]
            n.mass = n.value + sum(self.nodes[c].mass for c in n.children)

        # Strong/weak is a BRANCH-level property (Muir 2005: "All nodes of
        # a strong branch are called strong") -- only the root-adjacent
        # edge is tested; the result propagates to every descendant.
        for r in root_children:
            n = self.nodes[r]
            is_strong = (n.edge_type == 'p' and n.mass > 0) or (n.edge_type == 'm' and n.mass <= 0)
            stack = [r]
            while stack:
                v = stack.pop()
                self.nodes[v].strong = is_strong
                stack.extend(self.nodes[v].children)

    # ---------------------------------------------------------------
    # Find the earliest weak-over-strong arc in the ORIGINAL graph (not tree).
    # ---------------------------------------------------------------

    def _find_merger_arc(self):
        while self._candidate_heap:
            arc_id, version = self._candidate_heap[0]
            if (version != self._candidate_versions[arc_id]
                    or arc_id not in self._active_candidates):
                heapq.heappop(self._candidate_heap)
                continue
            source, target = self._merger_arcs[arc_id]
            # Defensive validation also protects against a future mutation
            # path that fails to refresh an endpoint's incident arcs.
            if self.nodes[source].strong and not self.nodes[target].strong:
                return source, target
            heapq.heappop(self._candidate_heap)
            self._refresh_merger_arc(arc_id)
        return None

    # ---------------------------------------------------------------
    # Merger: reverse the chain from s up to its branch root, attach
    # to w. (Muir 2005, steps 1a-1d.)
    # ---------------------------------------------------------------

    def _merge(self, s, w):
        rs = self._root_of(s)
        path = self._path_to_root(s)  # [s, ..., rs]
        self._log(f"MERGE  strong_root={rs}  arc={s}->{w}  weak_root={self._root_of(w)}")

        old_edge_type = {v: self.nodes[v].edge_type for v in path}

        # detach every node on the chain from its OLD parent's children set
        for v in path:
            old_parent = self.nodes[v].parent
            if old_parent != ROOT:
                self.nodes[old_parent].children.discard(v)

        # s's new parent is w -- the brand new merger edge, ALWAYS an
        # m-edge regardless of the underlying graph arc direction
        # (Muir 2005: "The merger edge (s,w) is an m-edge since it points
        # toward the root x0").
        self.nodes[s].parent = w
        self.nodes[s].edge_type = 'm'
        self.nodes[w].children.add(s)

        # reverse the rest of the chain: path[i]'s new parent is path[i-1]
        # (its former child), and the edge type transfers from the OLD
        # owner of that edge (path[i-1]) with its type flipped.
        for i in range(1, len(path)):
            v = path[i]
            prev = path[i - 1]
            self.nodes[v].parent = prev
            self.nodes[prev].children.add(v)
            self.nodes[v].edge_type = 'm' if old_edge_type[prev] == 'p' else 'p'

        self.merger_count += 1

        # Return the root of the branch whose topology/masses changed.
        # After a merge, s's old branch has been fully absorbed into w's
        # branch -- that branch is now rooted at root_of(w).
        return self._root_of(w)

    # ---------------------------------------------------------------
    # Branch-scoped mass + strong/weak recompute (only the ONE branch
    # affected by the preceding merge -- every other root-adjacent
    # branch is untouched by that merge and doesn't need revisiting).
    # ---------------------------------------------------------------

    def _recompute_branch_masses_and_strength(self, root):
        """
        Recompute masses and strong/weak status for one root-adjacent
        branch only.

        All other root-adjacent branches are unchanged by the preceding
        merge/prune operation.
        """
        order = []
        stack = [(root, False)]
        visited = set()

        while stack:
            node_id, processed = stack.pop()

            if not processed:
                if node_id in visited:
                    raise RuntimeError(
                        f"INVALID LG TREE: node {node_id} "
                        f"visited more than once during branch traversal"
                    )

                visited.add(node_id)

                stack.append((node_id, True))

                for c in self.nodes[node_id].children:
                    stack.append((c, False))

            else:
                order.append(node_id)

        # Post-order mass calculation.
        for v in order:
            n = self.nodes[v]
            n.mass = n.value + sum(
                self.nodes[c].mass
                for c in n.children
            )

        # Branch-level strong/weak status.
        root_node = self.nodes[root]

        is_strong = (
            (root_node.edge_type == 'p' and root_node.mass > 0)
            or
            (root_node.edge_type == 'm' and root_node.mass <= 0)
        )

        self._set_branch_strength(root, is_strong)
        # The same branch nodes are needed immediately afterward to seed
        # normalization's ordered violation heap. Reuse this traversal rather
        # than walking the entire (potentially large) merged branch again.
        return order

    # ---------------------------------------------------------------
    # Pruning / normalization: repeatedly find any non-root-adjacent
    # edge that is locally "strong" (violates the normalized-tree
    # invariant) and sever it into its own root-adjacent branch.
    # (Muir 2005, steps 2a-2e -- implemented as an ordered candidate heap
    # and sever loop rather than chain-only, for simplicity/debuggability.)
    # ---------------------------------------------------------------

    def _normalize(self, affected_root):
        # Only the branch containing the preceding merger has changed;
        # recompute its masses and branch strength, leaving other branches
        # untouched.
        branch_nodes = self._recompute_branch_masses_and_strength(affected_root)

        violation_heap = []
        violation_versions = {}

        def refresh_violation(node):
            version = violation_versions.get(node, 0) + 1
            violation_versions[node] = version
            current = self.nodes[node]
            if current.parent != ROOT and self._locally_strong(current):
                heapq.heappush(
                    violation_heap,
                    (self._node_order[node], version, node),
                )

        for node in branch_nodes:
            refresh_violation(node)

        while True:
            violation = None
            while violation_heap:
                _, version, node = heapq.heappop(violation_heap)
                if version != violation_versions.get(node):
                    continue
                current = self.nodes[node]
                if current.parent == ROOT or not self._locally_strong(current):
                    refresh_violation(node)
                    continue
                violation = node
                break

            if violation is None:
                return

            old_parent = self.nodes[violation].parent

            # Save the mass of the subtree being detached.
            detached_mass = self.nodes[violation].mass

            # The old branch root is unaffected by the detach operation
            # itself, but its branch mass may change.
            old_branch_root = self._root_of(old_parent)

            # Detach violation from its old parent.
            self.nodes[old_parent].children.discard(violation)
            self.nodes[violation].parent = ROOT
            refresh_violation(violation)

            # The detached subtree keeps exactly the same internal masses.
            # Only the ancestors of old_parent lose detached_mass.
            affected_ancestors = []
            v = old_parent
            while v != ROOT:
                self.nodes[v].mass -= detached_mass
                affected_ancestors.append(v)
                v = self.nodes[v].parent
            for v in affected_ancestors:
                refresh_violation(v)

            self.prune_count += 1

            self._log(
                f"PRUNE  node={violation} re-rooted directly to x0 "
                f"(was strong mid-branch)"
            )

            # -----------------------------------------------------------
            # Recompute strong/weak status for the two affected branches.
            #
            # old_branch_root may have changed mass because the detached
            # subtree was removed.
            #
            # violation is now itself a branch root.
            # -----------------------------------------------------------

            old_root = old_branch_root
            old_root_node = self.nodes[old_root]

            old_branch_strong = (
                (old_root_node.edge_type == 'p' and old_root_node.mass > 0)
                or
                (old_root_node.edge_type == 'm' and old_root_node.mass <= 0)
            )

            self._set_branch_strength(old_root, old_branch_strong)

            new_root_node = self.nodes[violation]

            new_branch_strong = (
                (new_root_node.edge_type == 'p' and new_root_node.mass > 0)
                or
                (new_root_node.edge_type == 'm' and new_root_node.mass <= 0)
            )

            self._set_branch_strength(violation, new_branch_strong)

    # ---------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------

    def solve(self):
        start_time = time.perf_counter()

        find_time = 0.0
        merge_time = 0.0
        normalize_time = 0.0

        while True:
            t0 = time.perf_counter()
            arc = self._find_merger_arc()
            find_time += time.perf_counter() - t0

            if arc is None:
                break

            s, w = arc
            self.iterations += 1

            t0 = time.perf_counter()
            affected_root = self._merge(s, w)
            merge_time += time.perf_counter() - t0

            t0 = time.perf_counter()
            self._normalize(affected_root)
            normalize_time += time.perf_counter() - t0

            if self.iterations % 100 == 0:
                elapsed = time.perf_counter() - start_time

                print(
                    f"LG progress: "
                    f"iterations={self.iterations:,} "
                    f"mergers={self.merger_count:,} "
                    f"prunes={self.prune_count:,} "
                    f"elapsed={elapsed:.1f}s "
                    f"rate={self.iterations / elapsed:.2f} iter/s "
                    f"find={find_time:.1f}s "
                    f"merge={merge_time:.1f}s "
                    f"normalize={normalize_time:.1f}s"
                )

        self._log(
            f"\nLG converged. iterations={self.iterations} "
            f"mergers={self.merger_count} prunes={self.prune_count} "
            f"selected_blocks={len(self.closure())} objective={self.objective()}"
        )

        return self.closure()

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    def closure(self):
        return {v for v, n in self.nodes.items() if n.strong}

    def objective(self):
        S = self.closure()
        return sum(self.values[v] for v in S)

    def violations(self):
        """Precedence violations in the CURRENT closure -- must be 0 at optimum."""
        S = self.closure()
        return sum(1 for v in S for w in self.precedence.get(v, []) if w not in S)
