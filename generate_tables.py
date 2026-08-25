from pathlib import Path

import pandas as pd


BUDGET_RESULTS_PATH = "results/csv/budget_experiments.csv"
EDGE_RESULTS_PATH = "results/csv/edge_removal_experiments.csv"
NODE_RESULTS_PATH = "results/csv/node_removal_experiments.csv"

OUTPUT_DIR = Path("results/tables")


ALGORITHM_ORDER = [
    "CSG-f1",
    "CSG-f2",
    "CGG",
]


def save_table(
    table: pd.DataFrame,
    filename: str
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = OUTPUT_DIR / filename

    table.to_csv(
        path,
        float_format="%.2f"
    )

    print(f"Salvata: {path}")


def generate_budget_tables() -> None:

    results = pd.read_csv(
        BUDGET_RESULTS_PATH
    )

    for cost_function in ["Random", "Degree"]:

        subset = results[
            results["cost_function"]
            == cost_function
        ]

        table = subset.pivot(
            index="budget_percentage",
            columns="algorithm",
            values="influence_percentage"
        )

        table = table[
            ALGORITHM_ORDER
        ]

        table.index.name = "Budget (%)"

        table.columns.name = None

        table = table.round(2)

        save_table(
            table,
            f"budget_{cost_function.lower()}.csv"
        )

        print(
            f"\n=== BUDGET - {cost_function.upper()} ==="
        )
        print(table)


def generate_edge_removal_tables() -> None:

    results = pd.read_csv(
        EDGE_RESULTS_PATH
    )

    aggregated = (
        results
        .groupby(
            [
                "cost_function",
                "budget_percentage",
                "algorithm",
                "edge_removal_percentage",
            ],
            as_index=False
        )
        ["retention_percentage"]
        .mean()
    )

    for cost_function in ["Random", "Degree"]:

        subset = aggregated[
            aggregated["cost_function"]
            == cost_function
        ]

        table = subset.pivot_table(
            index=[
                "budget_percentage",
                "edge_removal_percentage",
            ],
            columns="algorithm",
            values="retention_percentage"
        )

        table = table[
            ALGORITHM_ORDER
        ]

        table.index.names = [
            "Budget (%)",
            "Edges removed (%)"
        ]

        table.columns.name = None

        table = table.round(2)

        save_table(
            table,
            f"edge_removal_{cost_function.lower()}.csv"
        )

        print(
            f"\n=== EDGE REMOVAL - "
            f"{cost_function.upper()} ==="
        )
        print(table)


def generate_node_removal_tables() -> None:

    results = pd.read_csv(
        NODE_RESULTS_PATH
    )

    aggregated = (
        results
        .groupby(
            [
                "cost_function",
                "budget_percentage",
                "algorithm",
                "node_removal_percentage",
            ],
            as_index=False
        )
        ["retention_percentage"]
        .mean()
    )

    for cost_function in ["Random", "Degree"]:

        subset = aggregated[
            aggregated["cost_function"]
            == cost_function
        ]

        table = subset.pivot_table(
            index=[
                "budget_percentage",
                "node_removal_percentage",
            ],
            columns="algorithm",
            values="retention_percentage"
        )

        table = table[
            ALGORITHM_ORDER
        ]

        table.index.names = [
            "Budget (%)",
            "Nodes removed (%)"
        ]

        table.columns.name = None

        table = table.round(2)

        save_table(
            table,
            f"node_removal_{cost_function.lower()}.csv"
        )

        print(
            f"\n=== NODE REMOVAL - "
            f"{cost_function.upper()} ==="
        )
        print(table)


def main():

    generate_budget_tables()

    generate_edge_removal_tables()

    generate_node_removal_tables()

    print(
        "\nTutte le tabelle sono state "
        "generate correttamente."
    )


if __name__ == "__main__":
    main()