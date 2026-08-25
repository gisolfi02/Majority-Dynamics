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


ALGORITHMS = {
    "CSG-f1": cost_seeds_greedy_f1,
    "CSG-f2": cost_seeds_greedy_f2,
    "CGG": cascade_gain_greedy,
}


def run_budget_experiments(
    graph: nx.Graph,
    costs: dict[int, int],
    cost_name: str,
    budget_percentages: list[float],
) -> pd.DataFrame:
    """
    Esegue i tre algoritmi al variare del budget.

    Il budget viene espresso come percentuale
    del costo totale della rete.
    """

    results = []

    total_network_cost = sum(costs.values())

    for percentage in budget_percentages:

        budget = int(
            total_network_cost * percentage
        )

        print(
            f"\n--- {cost_name} | "
            f"Budget {percentage * 100:.0f}% "
            f"(k = {budget}) ---"
        )

        for algorithm_name, algorithm in ALGORITHMS.items():

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

            influence_percentage = (
                influenced_nodes
                / graph.number_of_nodes()
            ) * 100

            cascade_activated = (
                influenced_nodes - len(seeds)
            )

            results.append({
                "cost_function": cost_name,
                "budget_percentage":
                    percentage * 100,
                "budget": budget,
                "algorithm": algorithm_name,
                "seed_count": len(seeds),
                "seed_cost": seed_cost,
                "influenced_nodes":
                    influenced_nodes,
                "cascade_activated":
                    cascade_activated,
                "influence_percentage":
                    influence_percentage,
                "rounds": rounds,
                "execution_time_seconds":
                    execution_time,
            })

            print(
                f"{algorithm_name:8s} | "
                f"seed={len(seeds):4d} | "
                f"c(S)={seed_cost:6d} | "
                f"Inf={influenced_nodes:4d} | "
                f"{influence_percentage:6.2f}% | "
                f"time={execution_time:.2f}s"
            )

    return pd.DataFrame(results)

def run_edge_removal_experiments(
    graph: nx.Graph,
    cost_functions: dict[str, dict[int, int]],
    budget_percentages: list[float],
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

    from src.perturbations import remove_random_edges

    results = []

    # --------------------------------------------------
    # 1. CALCOLO DEI SEED SET SUL GRAFO ORIGINALE G
    # --------------------------------------------------

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

                seeds = algorithm(
                    graph,
                    budget,
                    costs
                )

                seed_cost = seed_set_cost(
                    seeds,
                    costs
                )

                baseline_active, baseline_rounds = (
                    majority_cascade(
                        graph,
                        seeds
                    )
                )

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
                        len(baseline_active),
                    "baseline_rounds":
                        baseline_rounds,
                })

    # --------------------------------------------------
    # 2. CREAZIONE DEI GRAFI PERTURBATI G'
    # --------------------------------------------------

    print("\n=== EDGE REMOVAL EXPERIMENTS ===")

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