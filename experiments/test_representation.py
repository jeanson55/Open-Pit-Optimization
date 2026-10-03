import numpy as np

from graph.load_graph import load_graph
from optimization.representation import PitSolution


graph, df = load_graph(
    "data/BauxiteMed/bauxitemed.dat"
)


pit = np.load(
    "results/bauxitemed_baseline_pit.npy"
)


solution = PitSolution(
    pit,
    df["EPV"].values
)


print("Blocks:", solution.blocks())
print("Objective:", solution.objective())