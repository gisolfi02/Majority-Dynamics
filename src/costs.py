import random

import networkx as nx


def generate_random_costs(
    graph: nx.Graph,
    min_cost: int = 1,
    max_cost: int = 10,
    seed: int = 123
) -> dict[int, int]:
    """
    Assegna a ogni nodo un costo intero casuale
    compreso tra min_cost e max_cost.

    L'uso di un seed fisso rende l'esperimento
    riproducibile.
    """

    rng = random.Random(seed)

    return {
        node: rng.randint(min_cost, max_cost)
        for node in graph.nodes
    }


def generate_degree_costs(
    graph: nx.Graph
) -> dict[int, int]:
    """
    Assegna a ogni nodo il costo:

        c(u) = ceil(d(u) / 2)

    dove d(u) è il grado del nodo.
    """

    return {
        node: (graph.degree[node] + 1) // 2
        for node in graph.nodes
    }


def seed_set_cost(
    seeds: set[int],
    costs: dict[int, int]
) -> int:
    """
    Calcola il costo complessivo di un seed set:

        c(S) = sum_{u in S} c(u)
    """

    return sum(costs[node] for node in seeds)