from src.graph import load_graph, get_graph_statistics # type: ignore
from src.diffusion import majority_cascade


DATASET_PATH = "data/facebook_combined.txt"

# Seed set temporaneo usato soltanto per verificare
# il funzionamento della Majority Cascade.
TEST_SEEDS = {0, 1, 2}


def main():
    # Caricamento del dataset
    graph = load_graph(DATASET_PATH) # pyright: ignore[reportUnknownVariableType]

    # Statistiche della rete
    stats = get_graph_statistics(graph) # type: ignore

    print("\n=== FACEBOOK SOCIAL NETWORK ===")
    print(f"Nodi: {stats['nodes']}")
    print(f"Archi: {stats['edges']}")
    print(f"Grado medio: {stats['average_degree']:.2f}")
    print(f"Grado minimo: {stats['min_degree']}")
    print(f"Grado massimo: {stats['max_degree']}")
    print(f"Densità: {stats['density']:.6f}")
    print(f"Componenti connesse: {stats['connected_components']}")
    print(
        f"Dimensione componente principale: "
        f"{stats['largest_component_size']}"
    )
    print(
        f"Clustering medio: "
        f"{stats['average_clustering']:.4f}"
    )

    # Majority Cascade
    active, rounds = majority_cascade(
        graph,
        TEST_SEEDS
    )

    influence_percentage = (
        len(active) / graph.number_of_nodes()
    ) * 100

    print("\n=== MAJORITY CASCADE ===")
    print(f"Seed set iniziale: {TEST_SEEDS}")
    print(f"Numero di seed: {len(TEST_SEEDS)}")
    print(f"Nodi finali attivati: {len(active)}")
    print(f"Round della cascade: {rounds}")
    print(f"Percentuale influenzata: {influence_percentage:.2f}%")


if __name__ == "__main__":
    main()