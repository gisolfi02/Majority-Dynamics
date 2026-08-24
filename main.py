from src.graph import load_graph, get_graph_statistics
from src.plots import (
    plot_degree_histogram,
    plot_local_clustering_histogram,
)


DATASET_PATH = "data/facebook_combined.txt"


def main():
    graph = load_graph(DATASET_PATH)

    stats = get_graph_statistics(graph)

    print("\n=== FACEBOOK SOCIAL NETWORK ===")
    print(f"Nodi: {stats['nodes']}")
    print(f"Archi: {stats['edges']}")
    print(f"Grado medio: {stats['average_degree']:.2f}")
    print(f"Grado minimo: {stats['min_degree']}")
    print(f"Grado massimo: {stats['max_degree']}")
    print(f"Densità: {stats['density']:.6f}")
    print(f"Componenti connesse: {stats['connected_components']}")
    print(f"Dimensione componente principale: {stats['largest_component_size']}")
    print(f"Clustering medio: {stats['average_clustering']:.4f}")

    plot_degree_histogram(
        graph,
        "results/figures/degree_histogram.png"
    )

    plot_local_clustering_histogram(
        graph,
        "results/figures/local_clustering_histogram.png"
    )

    print("\nGrafici salvati in results/figures/")


if __name__ == "__main__":
    main()