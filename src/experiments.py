import time

import networkx as nx
import pandas as pd

from src.algorithms import (
    cost_seeds_greedy_f1,
    cost_seeds_greedy_f2,
    cascade_gain_greedy,
)
from src.costs import seed_set_cost
from src.diffusion import majority_cascade
from src.perturbations import (
  remove_random_edges,
  remove_random_nodes
)


ALGORITHMS = {
    "CSG-f1": cost_seeds_greedy_f1,
    "CSG-f2": cost_seeds_greedy_f2,
    "CGG": cascade_gain_greedy,
}


def run_budget_experiments(
    configurations: list[dict]
) -> pd.DataFrame:

    results = []

    print("\n=== BUDGET EXPERIMENTS ===")

    for config in configurations:

        results.append({
            "cost_function":
                config["cost_function"],

            "budget_percentage":
                config["budget_percentage"],

            "budget":
                config["budget"],

            "algorithm":
                config["algorithm"],

            "seed_count":
                config["seed_count"],

            "seed_cost":
                config["seed_cost"],

            "influenced_nodes":
                config["baseline_influenced_nodes"],

            "cascade_activated":
                config["cascade_activated"],

            "influence_percentage":
                config["baseline_influence_percentage"],

            "rounds":
                config["baseline_rounds"],

            "execution_time_seconds":
                config["execution_time_seconds"],
        })

        print(
            f"{config['cost_function']:6s} | "
            f"budget {config['budget_percentage']:5.0f}% | "
            f"{config['algorithm']:8s} | "
            f"seed={config['seed_count']:4d} | "
            f"Inf={config['baseline_influenced_nodes']:4d} | "
            f"{config['baseline_influence_percentage']:6.2f}%"
        )

    return pd.DataFrame(results)


def compute_seed_configurations(
    graph: nx.Graph,
    cost_functions: dict[str, dict[int, int]],
    budget_percentages: list[float],
) -> list[dict]:
    """
    Calcola tutti i seed set sul grafo originale G.

    Per ogni combinazione di:
        - funzione di costo
        - budget
        - algoritmo

    salva il seed set e le relative informazioni baseline.

    I risultati possono poi essere riutilizzati
    negli esperimenti di edge removal e node removal.
    """

    configurations = []

    print("\n=== CALCOLO SEED SET SUL GRAFO ORIGINALE ===")

    for cost_name, costs in cost_functions.items():

        total_network_cost = sum(costs.values())

        for percentage in budget_percentages:

            budget = int(
                total_network_cost * percentage
            )

            for algorithm_name, algorithm in ALGORITHMS.items():

                print(
                    f"{cost_name} | "
                    f"budget {percentage * 100:.0f}% | "
                    f"{algorithm_name}"
                )

                start_time = time.perf_counter()

                seeds = algorithm(
                    graph,
                    budget,
                    costs
                )

                execution_time = (
                    time.perf_counter() - start_time
                )

                seed_cost = seed_set_cost(
                    seeds,
                    costs
                )

                active, rounds = majority_cascade(
                    graph,
                    seeds
                )

                influenced_nodes = len(active)

                configurations.append({
                    "cost_function": cost_name,
                    "budget_percentage":
                        percentage * 100,
                    "budget": budget,
                    "algorithm": algorithm_name,

                    "seeds": seeds,

                    "seed_count": len(seeds),
                    "seed_cost": seed_cost,

                    "baseline_influenced_nodes":
                        influenced_nodes,

                    "baseline_influence_percentage":
                        (
                            influenced_nodes
                            / graph.number_of_nodes()
                        ) * 100,

                    "cascade_activated":
                        influenced_nodes - len(seeds),

                    "baseline_rounds":
                        rounds,

                    "execution_time_seconds":
                        execution_time,
                })

    return configurations

def run_edge_removal_experiments(
    graph: nx.Graph,
    configurations: list[dict],
    removal_percentages: list[float],
    repetitions: int = 20,
    perturbation_seed: int = 0,
) -> pd.DataFrame:
    """
    Valuta la robustezza dei seed set rispetto
    alla rimozione casuale di archi.

    I seed set vengono calcolati una sola volta
    sul grafo originale G e successivamente
    mantenuti invariati sui grafi perturbati G'.
    """

    results = []

    original_num_edges = graph.number_of_edges()
    total_nodes = graph.number_of_nodes()

    for removal_percentage in removal_percentages:

        print(
            f"\nRimozione archi: "
            f"{removal_percentage * 100:.0f}%"
        )

        for repetition in range(repetitions):

            # Seed diverso per ogni replica.
            current_seed = (
                perturbation_seed + repetition
            )

            perturbed_graph = remove_random_edges(
                graph,
                removal_percentage,
                current_seed
            )

            removed_edges = (
                original_num_edges
                - perturbed_graph.number_of_edges()
            )

            # ------------------------------------------
            # 3. VALUTAZIONE DEGLI STESSI SEED SET
            #    SUL GRAFO MODIFICATO G'
            # ------------------------------------------

            for config in configurations:

                perturbed_active, perturbed_rounds = (
                    majority_cascade(
                        perturbed_graph,
                        config["seeds"]
                    )
                )

                perturbed_influenced = len(
                    perturbed_active
                )

                baseline_influenced = (
                    config[
                        "baseline_influenced_nodes"
                    ]
                )

                delta_influence = (
                    perturbed_influenced
                    - baseline_influenced
                )

                baseline_percentage = (
                    baseline_influenced
                    / total_nodes
                ) * 100

                perturbed_percentage = (
                    perturbed_influenced
                    / total_nodes
                ) * 100

                retention_percentage = (
                    perturbed_influenced
                    / baseline_influenced
                ) * 100

                results.append({
                    "cost_function":
                        config["cost_function"],

                    "budget_percentage":
                        config["budget_percentage"],

                    "budget":
                        config["budget"],

                    "algorithm":
                        config["algorithm"],

                    "seed_count":
                        config["seed_count"],

                    "seed_cost":
                        config["seed_cost"],

                    "baseline_influenced_nodes":
                        baseline_influenced,

                    "baseline_influence_percentage":
                        baseline_percentage,

                    "edge_removal_percentage":
                        removal_percentage * 100,

                    "repetition":
                        repetition,

                    "removed_edges":
                        removed_edges,

                    "perturbed_influenced_nodes":
                        perturbed_influenced,

                    "perturbed_influence_percentage":
                        perturbed_percentage,

                    "delta_influence":
                        delta_influence,

                    "retention_percentage":
                        retention_percentage,

                    "perturbed_rounds":
                        perturbed_rounds,
                })

        print(
            f"Completate {repetitions} "
            f"ripetizioni."
        )

    return pd.DataFrame(results)

def run_node_removal_experiments(
    graph: nx.Graph,
    configurations: list[dict],
    removal_percentages: list[float],
    repetitions: int = 20,
    perturbation_seed: int = 0,
) -> pd.DataFrame:
    """
    Valuta la robustezza dei seed set rispetto
    alla rimozione casuale di vertici.

    I seed set vengono determinati sul grafo originale G.
    Dopo la perturbazione vengono utilizzati soltanto
    i seed ancora presenti in G'.
    """
    results = []

    original_num_nodes = graph.number_of_nodes()

    for removal_percentage in removal_percentages:

        print(
            f"\nRimozione nodi: "
            f"{removal_percentage * 100:.0f}%"
        )

        for repetition in range(repetitions):

            current_seed = (
                perturbation_seed + repetition
            )

            perturbed_graph = remove_random_nodes(
                graph,
                removal_percentage,
                current_seed
            )

            removed_nodes = (
                original_num_nodes
                - perturbed_graph.number_of_nodes()
            )

            # ------------------------------------------
            # 3. Valutazione dei seed originali
            # ------------------------------------------

            for config in configurations:

                original_seeds = config["seeds"]

                # Manteniamo soltanto i seed
                # sopravvissuti alla perturbazione.
                surviving_seeds = (
                    original_seeds
                    & set(perturbed_graph.nodes)
                )

                removed_seeds = (
                    len(original_seeds)
                    - len(surviving_seeds)
                )

                perturbed_active, perturbed_rounds = (
                    majority_cascade(
                        perturbed_graph,
                        surviving_seeds
                    )
                )

                perturbed_influenced = len(
                    perturbed_active
                )

                baseline_influenced = (
                    config[
                        "baseline_influenced_nodes"
                    ]
                )

                delta_influence = (
                    perturbed_influenced
                    - baseline_influenced
                )

                retention_percentage = (
                    perturbed_influenced
                    / baseline_influenced
                ) * 100

                # Percentuale rispetto ai nodi
                # ancora presenti in G'
                remaining_network_percentage = (
                    perturbed_influenced
                    / perturbed_graph.number_of_nodes()
                ) * 100

                results.append({
                    "cost_function":
                        config["cost_function"],

                    "budget_percentage":
                        config["budget_percentage"],

                    "budget":
                        config["budget"],

                    "algorithm":
                        config["algorithm"],

                    "original_seed_count":
                        config["seed_count"],

                    "surviving_seed_count":
                        len(surviving_seeds),

                    "removed_seed_count":
                        removed_seeds,

                    "seed_cost":
                        config["seed_cost"],

                    "baseline_influenced_nodes":
                        baseline_influenced,

                    "node_removal_percentage":
                        removal_percentage * 100,

                    "repetition":
                        repetition,

                    "removed_nodes":
                        removed_nodes,

                    "remaining_nodes":
                        perturbed_graph.number_of_nodes(),

                    "perturbed_influenced_nodes":
                        perturbed_influenced,

                    "remaining_network_influence_percentage":
                        remaining_network_percentage,

                    "delta_influence":
                        delta_influence,

                    "retention_percentage":
                        retention_percentage,

                    "perturbed_rounds":
                        perturbed_rounds,
                })

        print(
            f"Completate {repetitions} ripetizioni."
        )

    return pd.DataFrame(results)