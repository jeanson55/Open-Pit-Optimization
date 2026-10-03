"""Convert the mining block model into a directed precedence graph.

G = (V, E)
  Nodes V = blocks   (attributes x, y, z, grade, value, cost)
  Edges  E = mining dependencies

Edge direction convention (matches gPit's block precedence arcs):
    u -> v  means selecting u requires v to also be selected.

A mining solution S is feasible iff it is closed along directed arcs:
    u in S and (u, v) in E  =>  v in S.
"""

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


def build_graph(blocks: pd.DataFrame, arcs: pd.DataFrame) -> nx.DiGraph:
    """Build the precedence digraph from the raw CSV tables."""
    G = nx.DiGraph()
    for row in blocks.itertuples(index=False):
        G.add_node(
            int(row.block_id),
            x=float(row.x), y=float(row.y), z=float(row.z),
            grade=float(row.grade),
            value=float(row.value),          # net block value (revenue - cost)
            cost=float(row.cost),
        )
    for row in arcs.itertuples(index=False):
        G.add_edge(int(row.from_block), int(row.to_block))
    return G


def save_graph(G: nx.DiGraph, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)
