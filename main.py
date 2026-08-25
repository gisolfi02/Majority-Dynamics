import pandas as pd

from src.graph import load_graph
from src.costs import (
    generate_random_costs,
    generate_degree_costs,
)
from src.experiments import (
    run_budget_experiments,
    run_edge_removal_experiments,
    run_node_removal_experiments,
    compute_seed_configurations
)
from src.plots import plot_budget_influence
from src.perturbations import remove_random_edges

DATASET_PATH = "data/facebook_combined.txt"

BUDGET_PERCENTAGES = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]

EDGE_REMOVAL_PERCENTAGES = [
    0.01,
    0.05,
    0.10,
    0.20,
]

EDGE_REMOVAL_REPETITIONS = 20

NODE_REMOVAL_PERCENTAGES = [
    0.01,
    0.05,
    0.10,
    0.20,
]

NODE_REMOVAL_REPETITIONS = 20

def main():

    graph = load_graph(DATASET_PATH)

    print("\n=== GENERAZIONE COSTI ===")

    random_costs = generate_random_costs(
        graph,
        min_cost=1,
        max_cost=10,
        seed=123
    )

    degree_costs = generate_degree_costs(
        graph
    )

    print(
        f"Costo totale - Random: "
        f"{sum(random_costs.values())}"
    )

    print(
        f"Costo totale - Degree: "
        f"{sum(degree_costs.values())}"
    )

    cost_functions = {
            "Random": random_costs,
            "Degree": degree_costs,
        }

    configurations = compute_seed_configurations(
        graph=graph,
        cost_functions=cost_functions,
        budget_percentages=BUDGET_PERCENTAGES,
    )
    '''
    print("\n=== ESPERIMENTI BUDGET ===")

    budget_results = run_budget_experiments(
    configurations=configurations
    )

    budget_results.to_csv(
        "results/budget_experiments.csv",
        index=False
    )

    print(
        "\nRisultati salvati in "
        "results/csv/budget_experiments.csv"
    )


    print("\n=== GENERAZIONE GRAFICI ===")

    plot_budget_influence(
        results=results,
        cost_function="Random",
        output_path=(
            "results/figures/"
            "budget_influence_random.png"
        ),
        total_nodes=graph.number_of_nodes()
    )

    plot_budget_influence(
        results=results,
        cost_function="Degree",
        output_path=(
            "results/figures/"
            "budget_influence_degree.png"
        ),
        total_nodes=graph.number_of_nodes()
    )

    print(
        "Grafici salvati in "
        "results/figures/"
    )
    '''
    print("\n=== ESPERIMENTI RIMOZIONE ARCHI ===")


    edge_results = run_edge_removal_experiments(
        graph=graph,
        configurations=configurations,
        removal_percentages=EDGE_REMOVAL_PERCENTAGES,
        repetitions=EDGE_REMOVAL_REPETITIONS,
        perturbation_seed=0,
    )

    edge_results.to_csv(
        "results/csv/edge_removal_experiments.csv",
        index=False
    )

    print(
        "\nRisultati salvati in "
        "results/csv/edge_removal_experiments.csv"
    )

    print("\n=== ESPERIMENTI RIMOZIONE NODI ===")

    node_results = run_node_removal_experiments(
        graph=graph,
        configurations=configurations,
        removal_percentages=NODE_REMOVAL_PERCENTAGES,
        repetitions=NODE_REMOVAL_REPETITIONS,
        perturbation_seed=0,
    )

    node_results.to_csv(
        "results/csv/node_removal_experiments.csv",
        index=False
    )

    print(
        "\nRisultati salvati in "
        "results/csv/node_removal_experiments.csv"
    )


if __name__ == "__main__":
    main()