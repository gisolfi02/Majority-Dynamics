from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def plot_degree_histogram(graph: nx.Graph, output_path: str) -> None:
    """
    Genera e salva l'istogramma della distribuzione dei gradi.
    """

    degrees = [degree for _, degree in graph.degree()]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=30, edgecolor="black")
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_local_clustering_histogram(graph: nx.Graph, output_path: str) -> None:
    """
    Genera e salva l'istogramma della distribuzione
    del coefficiente di clustering locale.
    """

    clustering_dict = nx.clustering(graph)
    clustering_values = list(clustering_dict.values())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(clustering_values, bins=30, edgecolor="black")
    plt.title("Local Clustering Coefficient Distribution")
    plt.xlabel("Local Clustering Coefficient")
    plt.ylabel("Number of Nodes")
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


def plot_budget_influence(
    results: pd.DataFrame,
    cost_function: str,
    output_path: str,
    total_nodes: int
) -> None:
    """
    Rappresenta |Inf[G,S]| al variare del budget
    per i tre algoritmi, fissata una funzione di costo.
    """

    results = results.copy()
    results.columns = results.columns.str.strip()
    for column in ("cost_function", "algorithm"):
        results[column] = results[column].astype(str).str.strip()

    subset = results[
        results["cost_function"] == cost_function
    ]

    algorithms = [
        "CSG-f1",
        "CSG-f2",
        "CGG"
    ]

    plt.figure(figsize=(8, 5))

    for algorithm in algorithms:

        algorithm_results = subset[
            subset["algorithm"] == algorithm
        ].sort_values("budget_percentage")

        plt.plot(
            algorithm_results["budget_percentage"],
            algorithm_results["influenced_nodes"],
            marker="o",
            linewidth=2,
            label=algorithm
        )

    plt.xlabel("Budget (% of total network cost)")
    plt.ylabel("Influenced Nodes |Inf[G,S]|")

    plt.title(
        f"Influence vs Budget - {cost_function} Costs"
    )

    plt.xticks(
        sorted(subset["budget_percentage"].unique())
    )

    # Stessa scala nei due grafici:
    # 0 = nessun nodo attivo
    # total_nodes = intera rete attiva
    plt.ylim(0, total_nodes * 1.05)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def aggregate_edge_removal_results(
    results: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggrega i risultati della rimozione archi
    calcolando media e deviazione standard
    delle repliche.
    """

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
        .agg(
            mean_influenced_nodes=(
                "perturbed_influenced_nodes",
                "mean"
            ),
            std_influenced_nodes=(
                "perturbed_influenced_nodes",
                "std"
            ),
            baseline_influenced_nodes=(
                "baseline_influenced_nodes",
                "first"
            ),
        )
    )

    return aggregated


def plot_edge_removal_composite(
    aggregated_results: pd.DataFrame,
    cost_function: str,
    output_path: str,
    total_nodes: int
) -> None:
    """
    Genera una figura composita con 3 pannelli:
    uno per algoritmo. In ogni pannello ci sono
    5 curve, una per budget.

    X = percentuale di archi rimossi (incluso 0%)
    Y = |Inf[G', S]|
    """

    subset = aggregated_results[
        aggregated_results["cost_function"] == cost_function
    ].copy()

    algorithms = ["CSG-f1", "CSG-f2", "CGG"]
    budgets = [1, 2, 5, 10, 20]

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), sharey=True
    )

    for ax, algorithm in zip(axes, algorithms):

        algorithm_subset = subset[
            subset["algorithm"] == algorithm
        ]

        for budget in budgets:

            budget_subset = algorithm_subset[
                algorithm_subset["budget_percentage"] == budget
            ].sort_values("edge_removal_percentage")

            if budget_subset.empty:
                continue

            baseline = budget_subset[
                "baseline_influenced_nodes"
            ].iloc[0]

            x_values = [0] + budget_subset[
                "edge_removal_percentage"
            ].tolist()

            y_values = [baseline] + budget_subset[
                "mean_influenced_nodes"
            ].tolist()

            y_errors = [0] + budget_subset[
                "std_influenced_nodes"
            ].fillna(0).tolist()

            ax.errorbar(
                x_values,
                y_values,
                yerr=y_errors,
                marker="o",
                linewidth=2,
                capsize=4,
                label=f"Budget {budget}%"
            )

        ax.set_title(algorithm)
        ax.set_xlabel("Removed Edges (%)")
        ax.set_xticks([0, 1, 5, 10, 20])
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].set_ylabel("Influenced Nodes |Inf[G',S]|")
    axes[0].set_ylim(0, total_nodes * 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.05)
    )

    fig.suptitle(
        f"Edge Removal Robustness - {cost_function} Costs",
        fontsize=14
    )

    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()