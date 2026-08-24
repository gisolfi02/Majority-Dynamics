from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def plot_degree_histogram(graph: nx.Graph, output_path: str) -> None:
    """
    Genera e salva l'istogramma della distribuzione dei gradi.
    """

    degrees = [degree for _, degree in graph.degree()]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=30, edgecolor="black")
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_local_clustering_histogram(graph: nx.Graph, output_path: str) -> None:
    """
    Genera e salva l'istogramma della distribuzione
    del coefficiente di clustering locale.
    """

    clustering_dict = nx.clustering(graph)
    clustering_values = list(clustering_dict.values())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(clustering_values, bins=30, edgecolor="black")
    plt.title("Local Clustering Coefficient Distribution")
    plt.xlabel("Local Clustering Coefficient")
    plt.ylabel("Number of Nodes")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()