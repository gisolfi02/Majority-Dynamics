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