import pandas as pd

from src.plots import (
    aggregate_node_removal_results,
    aggregate_edge_removal_results,
    plot_node_removal_composite,
    plot_edge_removal_composite,
    plot_budget_influence,
)


RESULTS_PATH1 = "results/csv/node_removal_experiments.csv"
RESULTS_PATH2 = "results/csv/edge_removal_experiments.csv"
RESULTS_PATH3 = "results/csv/budget_experiments.csv"

TOTAL_NODES = 4039


def main():
    print("\n=== GRAFICI RIMOZIONE NODI ===")

    results = pd.read_csv(RESULTS_PATH1)

    aggregated = aggregate_node_removal_results(results)

    aggregated.to_csv(
        "results/csv/node_removal_aggregated.csv",
        index=False
    )

    plot_node_removal_composite(
        aggregated_results=aggregated,
        cost_function="Random",
        output_path=(
            "results/figures/"
            "node_removal_random_composite.png"
        ),
        total_nodes=TOTAL_NODES
    )

    plot_node_removal_composite(
        aggregated_results=aggregated,
        cost_function="Degree",
        output_path=(
            "results/figures/"
            "node_removal_degree_composite.png"
        ),
        total_nodes=TOTAL_NODES
    )

    print("Grafici compositi node removal generati correttamente.")
    print(
        "File aggregato salvato in "
        "results/csv/node_removal_aggregated.csv"
    )

    print("\n=== GRAFICI RIMOZIONE ARCHI ===")


    results = pd.read_csv(RESULTS_PATH2)
    
    aggregated = aggregate_edge_removal_results(results)

    aggregated.to_csv(
        "results/csv/edge_removal_aggregated.csv",
        index=False
    )

    plot_edge_removal_composite(
        aggregated_results=aggregated,
        cost_function="Random",
        output_path=(
            "results/figures/"
            "edge_removal_random_composite.png"
        ),
        total_nodes=TOTAL_NODES
    )

    plot_edge_removal_composite(
        aggregated_results=aggregated,
        cost_function="Degree",
        output_path=(
            "results/figures/"
            "edge_removal_degree_composite.png"
        ),
        total_nodes=TOTAL_NODES
    )

    print("Grafici compositi generati correttamente.")
    print(
        "File aggregato salvato in "
        "results/csv/edge_removal_aggregated.csv"
    )


    print("\n=== GRAFICI INFLUENZA BUDGET ===")

    results = pd.read_csv(RESULTS_PATH3)
  
    plot_budget_influence(
        results=results,
        cost_function="Random",
        output_path=(
            "results/figures/"
            "budget_influence_random.png"
        ),
        total_nodes=TOTAL_NODES
    )

    plot_budget_influence(
        results=results,
        cost_function="Degree",
        output_path=(
            "results/figures/"
            "budget_influence_degree.png"
        ),
        total_nodes=TOTAL_NODES
    )

    print(
        "Grafici generati correttamente."
    )


if __name__ == "__main__":
    main()