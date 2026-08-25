import pandas as pd

from src.graph import load_graph
from src.costs import (
    generate_random_costs,
    generate_degree_costs,
)
from src.experiments import (
    run_budget_experiments,
    run_edge_removal_experiments,
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
    '''
    print("\n=== ESPERIMENTI RANDOM COST ===")

    random_results = run_budget_experiments(
        graph=graph,
        costs=random_costs,
        cost_name="Random",
        budget_percentages=BUDGET_PERCENTAGES,
    )

    print("\n=== ESPERIMENTI DEGREE COST ===")

    degree_results = run_budget_experiments(
        graph=graph,
        costs=degree_costs,
        cost_name="Degree",
        budget_percentages=BUDGET_PERCENTAGES,
    )

    results = pd.concat(
        [
            random_results,
            degree_results
        ],
        ignore_index=True
    )

    results.to_csv(
        "results/csv/budget_experiments.csv",
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

    print("\n=== TEST EDGE REMOVAL ===")

    test_graph = remove_random_edges(
        graph,
        removal_percentage=0.10,
        seed=0
    )

    print(
        f"Archi originali: "
        f"{graph.number_of_edges()}"
    )

    print(
        f"Archi dopo rimozione 10%: "
        f"{test_graph.number_of_edges()}"
    )

    print(
        f"Archi rimossi: "
        f"{graph.number_of_edges() - test_graph.number_of_edges()}"
    ) 
    '''
    print("\n=== ESPERIMENTI RIMOZIONE ARCHI ===")

    cost_functions = {
        "Random": random_costs,
        "Degree": degree_costs,
    }

    edge_results = run_edge_removal_experiments(
        graph=graph,
        cost_functions=cost_functions,
        budget_percentages=BUDGET_PERCENTAGES,
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


if __name__ == "__main__":
    main()