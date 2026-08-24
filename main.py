from src.graph import load_graph, get_graph_statistics # type: ignore
from src.diffusion import majority_cascade
from src.costs import (
    generate_random_costs,
    generate_degree_costs,
    seed_set_cost,
)


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

    # Generazione delle due funzioni di costo
    random_costs = generate_random_costs(graph)

    degree_costs = generate_degree_costs(graph)

    print("\n=== COST FUNCTIONS ===")

    print("\nPrimi 10 nodi:")

    for node in list(graph.nodes)[:10]:
        print(
            f"Nodo {node:4d} | "
            f"grado = {graph.degree[node]:3d} | "
            f"random cost = {random_costs[node]:2d} | "
            f"degree cost = {degree_costs[node]:3d}"
        )

    print("\nCosto totale della rete:")

    print(
        f"Random costs: "
        f"{sum(random_costs.values())}"
    )

    print(
        f"Degree costs: "
        f"{sum(degree_costs.values())}"
    )

    print("\nCosto del seed set temporaneo:")

    print(
        f"Random: "
        f"{seed_set_cost(TEST_SEEDS, random_costs)}"
    )

    print(
        f"Degree: "
        f"{seed_set_cost(TEST_SEEDS, degree_costs)}"
    )


if __name__ == "__main__":
    main()