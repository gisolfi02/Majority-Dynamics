from src.graph import load_graph, get_graph_statistics
from src.diffusion import majority_cascade
from src.costs import (
    generate_random_costs,
    generate_degree_costs,
    seed_set_cost,
)
from src.algorithms import (
    cost_seeds_greedy_f1,
    cost_seeds_greedy_f2
)

DATASET_PATH = "data/facebook_combined.txt"

# Seed set temporaneo usato soltanto per verificare
# il funzionamento della Majority Cascade.
TEST_SEEDS = {0, 1, 2}

# Budget temporaneo usato soltanto per verificare
# il funzionamento dell'algoritmo Cost-Seeds-Greedy.
TEST_BUDGET = 100

def main():
    # Caricamento del dataset
    graph = load_graph(DATASET_PATH)

    ''' 
    # Statistiche della rete
    stats = get_graph_statistics(graph)

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

    ''' 

    random_costs = generate_random_costs(graph)


    print("\n=== COST-SEEDS-GREEDY + f1 ===")


    f1_seeds = cost_seeds_greedy_f1(
        graph,
        TEST_BUDGET,
        random_costs
    )

    f1_cost = seed_set_cost(
        f1_seeds,
        random_costs
    )

    active, rounds = majority_cascade(
        graph,
        f1_seeds
    )

    influence_percentage = (
        len(active) / graph.number_of_nodes()
    ) * 100

    print(f"Budget: {TEST_BUDGET}")
    print(f"Numero di seed selezionati: {len(f1_seeds)}")
    print(f"Costo del seed set: {f1_cost}")
    print(f"Nodi finali attivati: {len(active)}")
    print(f"Round della cascade: {rounds}")
    print(
        f"Percentuale influenzata: "
        f"{influence_percentage:.2f}%"
    )

    print("\n=== COST-SEEDS-GREEDY + f2 ===")

    f2_seeds = cost_seeds_greedy_f2(
        graph,
        TEST_BUDGET,
        random_costs
    )

    f2_cost = seed_set_cost(
        f2_seeds,
        random_costs
    )

    f2_active, f2_rounds = majority_cascade(
        graph,
        f2_seeds
    )

    f2_influence_percentage = (
        len(f2_active) / graph.number_of_nodes()
    ) * 100

    print(f"Budget: {TEST_BUDGET}")
    print(
        f"Numero di seed selezionati: "
        f"{len(f2_seeds)}"
    )
    print(f"Costo del seed set: {f2_cost}")
    print(
        f"Nodi finali attivati: "
        f"{len(f2_active)}"
    )
    print(
        f"Round della cascade: "
        f"{f2_rounds}"
    )
    print(
        f"Percentuale influenzata: "
        f"{f2_influence_percentage:.2f}%"
    )


if __name__ == "__main__":
    main()