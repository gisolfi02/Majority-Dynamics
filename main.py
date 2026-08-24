import networkx as nx

from src.graph import load_graph


DATASET_PATH = "data/facebook_combined.txt"


def main():
    graph = load_graph(DATASET_PATH)

    print("=== FACEBOOK SOCIAL NETWORK ===")
    print(f"Nodi: {graph.number_of_nodes()}")
    print(f"Archi: {graph.number_of_edges()}")
    print(f"Grafo connesso: {nx.is_connected(graph)}")


if __name__ == "__main__":
    main()