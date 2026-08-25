import pandas as pd

from src.plots import plot_budget_influence


RESULTS_PATH = "results/csv/budget_experiments.csv"

TOTAL_NODES = 4039


def main():

    results = pd.read_csv(
        RESULTS_PATH
    )

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