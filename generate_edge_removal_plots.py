import pandas as pd

from src.plots import (
    aggregate_edge_removal_results,
    plot_edge_removal_composite,
)


RESULTS_PATH = "results/csv/edge_removal_experiments.csv"
TOTAL_NODES = 4039


def main():

    results = pd.read_csv(RESULTS_PATH)

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


if __name__ == "__main__":
    main()