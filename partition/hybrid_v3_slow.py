#cat > partition/hybrid.py <<'PY'
"""
Hybrid partitioner V3.

Strategy:
1. Generate Spatial and METIS seeds.
2. Refine each seed using a common multi-objective local search.
3. Select the refined partition with the lowest objective.

The objective combines:
- normalized precedence edge cut
- node-count imbalance
- economic-value imbalance

The implementation uses incremental move evaluation and does not
rescan the entire graph for every candidate move.
"""

from collections import defaultdict

from .spatial import spatial_partition
from .metis import metis_partition


MAX_SWEEPS = 6

# Objective weights
CUT_WEIGHT = 1.00
SIZE_WEIGHT = 0.35
VALUE_WEIGHT = 0.20

# Prevent extreme partition-size changes.
BALANCE_TOLERANCE = 1.20


def _safe_scale(value):
    return max(abs(float(value)), 1.0)


def _partition_stats(graph, labels):
    parts = sorted(set(labels.values()))

    sizes = defaultdict(int)
    values = defaultdict(float)

    for node in graph.nodes:
        p = labels[node]
        sizes[p] += 1
        values[p] += float(graph.nodes[node].get("value", 0.0))

    return parts, sizes, values


def _size_penalty(sizes):
    if not sizes:
        return 0.0

    mean_size = sum(sizes.values()) / len(sizes)

    return sum(
        ((size - mean_size) / _safe_scale(mean_size)) ** 2
        for size in sizes.values()
    ) / len(sizes)


def _value_penalty(values):
    if not values:
        return 0.0

    mean_value = sum(values.values()) / len(values)

    scale = max(
        abs(mean_value),
        max((abs(v) for v in values.values()), default=1.0),
        1.0,
    )

    return sum(
        ((value - mean_value) / scale) ** 2
        for value in values.values()
    ) / len(values)


def _cut_count(graph, labels):
    return sum(
        1
        for u, v in graph.edges
        if labels[u] != labels[v]
    )


def _objective(graph, labels):
    parts, sizes, values = _partition_stats(graph, labels)

    if not parts:
        return float("inf")

    cut = _cut_count(graph, labels)
    cut_norm = cut / _safe_scale(graph.number_of_edges())

    size_pen = _size_penalty(sizes)
    value_pen = _value_penalty(values)

    return (
        CUT_WEIGHT * cut_norm
        + SIZE_WEIGHT * size_pen
        + VALUE_WEIGHT * value_pen
    )


def _candidate_parts(graph, node, labels):
    candidates = set()

    for nbr in graph.successors(node):
        candidates.add(labels[nbr])

    for nbr in graph.predecessors(node):
        candidates.add(labels[nbr])

    candidates.discard(labels[node])

    return sorted(candidates)


def _boundary_nodes(graph, labels):
    boundary = set()

    for u, v in graph.edges:
        if labels[u] != labels[v]:
            boundary.add(u)
            boundary.add(v)

    return boundary


def _move_delta_cut(graph, node, old_part, new_part, labels):
    """
    Exact change in cut count caused by moving one node.

    Only edges incident to the candidate node are examined.
    """

    delta = 0

    for nbr in graph.successors(node):
        nbr_part = labels[nbr]

        before = int(old_part != nbr_part)
        after = int(new_part != nbr_part)

        delta += after - before

    for nbr in graph.predecessors(node):
        nbr_part = labels[nbr]

        before = int(nbr_part != old_part)
        after = int(nbr_part != new_part)

        delta += after - before

    return delta


def _trial_objective(
    graph,
    labels,
    node,
    new_part,
    sizes,
    values,
    current_cut,
):
    old_part = labels[node]

    old_size = sizes[old_part]
    new_size = sizes[new_part]

    node_value = float(graph.nodes[node].get("value", 0.0))

    trial_sizes = dict(sizes)
    trial_values = dict(values)

    trial_sizes[old_part] = old_size - 1
    trial_sizes[new_part] = new_size + 1

    trial_values[old_part] = values[old_part] - node_value
    trial_values[new_part] = values[new_part] + node_value

    delta_cut = _move_delta_cut(
        graph,
        node,
        old_part,
        new_part,
        labels,
    )

    trial_cut = current_cut + delta_cut

    cut_norm = trial_cut / _safe_scale(graph.number_of_edges())

    size_pen = _size_penalty(trial_sizes)
    value_pen = _value_penalty(trial_values)

    objective = (
        CUT_WEIGHT * cut_norm
        + SIZE_WEIGHT * size_pen
        + VALUE_WEIGHT * value_pen
    )

    return objective, trial_cut, trial_sizes, trial_values


def _valid_move(
    graph,
    node,
    old_part,
    new_part,
    sizes,
    total_nodes,
):
    """
    Keep partitions from becoming excessively unbalanced.
    """

    mean_size = total_nodes / len(sizes)

    old_after = sizes[old_part] - 1
    new_after = sizes[new_part] + 1

    if old_after <= 0:
        return False

    max_allowed = BALANCE_TOLERANCE * mean_size

    if new_after > max_allowed:
        return False

    return True


def _refine(graph, initial_labels):
    labels = dict(initial_labels)

    parts, sizes, values = _partition_stats(graph, labels)

    if len(parts) < 2:
        return labels

    current_cut = _cut_count(graph, labels)
    current_objective = _objective(graph, labels)

    total_nodes = graph.number_of_nodes()

    for sweep in range(MAX_SWEEPS):

        boundary = _boundary_nodes(graph, labels)

        # Deterministic ordering makes experiments reproducible.
        node_order = sorted(boundary)

        changed = False

        for node in node_order:

            old_part = labels[node]

            candidates = _candidate_parts(
                graph,
                node,
                labels,
            )

            if not candidates:
                continue

            best_part = old_part
            best_objective = current_objective
            best_cut = current_cut
            best_sizes = sizes
            best_values = values

            for new_part in candidates:

                if not _valid_move(
                    graph,
                    node,
                    old_part,
                    new_part,
                    sizes,
                    total_nodes,
                ):
                    continue

                (
                    trial_objective,
                    trial_cut,
                    trial_sizes,
                    trial_values,
                ) = _trial_objective(
                    graph,
                    labels,
                    node,
                    new_part,
                    sizes,
                    values,
                    current_cut,
                )

                if trial_objective + 1e-12 < best_objective:

                    best_part = new_part
                    best_objective = trial_objective
                    best_cut = trial_cut
                    best_sizes = trial_sizes
                    best_values = trial_values

            if best_part != old_part:

                labels[node] = best_part

                sizes = best_sizes
                values = best_values
                current_cut = best_cut
                current_objective = best_objective

                changed = True

        print(
            f"    refinement sweep {sweep + 1}: "
            f"boundary={len(boundary)} "
            f"cut={current_cut} "
            f"objective={current_objective:.8f}"
        )

        if not changed:
            break

    return labels


def hybrid_partition(graph, k):
    """
    Multi-seed Hybrid V3 partitioning.

    Returns a node -> partition mapping.
    """

    print("  Hybrid V3: generating Spatial seed...")
    spatial_labels = spatial_partition(graph, k)

    print("  Hybrid V3: generating METIS seed...")
    metis_labels = metis_partition(graph, k)

    seeds = [
        ("spatial", spatial_labels),
        ("metis", metis_labels),
    ]

    candidates = []

    for name, seed in seeds:

        print(f"  Hybrid V3: refining {name} seed...")

        refined = _refine(graph, seed)

        objective = _objective(graph, refined)

        cut = _cut_count(graph, refined)

        candidates.append(
            (
                objective,
                name,
                refined,
                cut,
            )
        )

        print(
            f"  Hybrid V3: {name} result "
            f"objective={objective:.8f}, "
            f"cut={cut}"
        )

    candidates.sort(key=lambda x: x[0])

    best_objective, best_name, best_labels, best_cut = candidates[0]

    print(
        f"  Hybrid V3: selected {best_name} seed "
        f"(objective={best_objective:.8f}, cut={best_cut})"
    )

    return {
        int(node): int(part)
        for node, part in best_labels.items()
    }
