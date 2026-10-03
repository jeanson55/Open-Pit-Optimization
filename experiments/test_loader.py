from graph.load_graph import load_graph


path = (
    "data/BauxiteMed/"
    "bauxitemed.dat"
)

G, df = load_graph(path)


print()
print("------ CHECK ------")
print("Blocks:", len(df))
print("Nodes :", G.number_of_nodes())
print("Edges :", G.number_of_edges())
print("-------------------")