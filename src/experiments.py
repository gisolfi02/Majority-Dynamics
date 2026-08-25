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