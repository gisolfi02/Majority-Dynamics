import random

import networkx as nx


def remove_random_edges(
    graph: nx.Graph,
    removal_percentage: float,
    seed: int
) -> nx.Graph:
    """
    Restituisce una copia del grafo dopo aver rimosso
    casualmente una percentuale degli archi.

    Parameters
    ----------
    graph : nx.Graph
        Grafo originale G.

    removal_percentage : float
        Percentuale di archi da rimuovere, espressa
        come valore tra 0 e 1.
        Esempio: 0.10 = 10%.

    seed : int
        Seed del generatore pseudo-casuale.

    Returns
    -------
    nx.Graph
        Grafo perturbato G'.
    """

    if not 0 <= removal_percentage <= 1:
        raise ValueError(
            "removal_percentage deve essere compreso tra 0 e 1."
        )

    perturbed_graph = graph.copy()

    edges = list(graph.edges())

    num_edges_to_remove = int(
        len(edges) * removal_percentage
    )

    rng = random.Random(seed)

    edges_to_remove = rng.sample(
        edges,
        num_edges_to_remove
    )

    perturbed_graph.remove_edges_from(
        edges_to_remove
    )

    return perturbed_graph

def remove_random_nodes(
    graph: nx.Graph,
    removal_percentage: float,
    seed: int
) -> nx.Graph:
    """
    Restituisce una copia del grafo dopo aver rimosso
    casualmente una percentuale dei vertici.

    La rimozione di un vertice comporta automaticamente
    la rimozione di tutti i suoi archi incidenti.
    """

    if not 0 <= removal_percentage <= 1:
        raise ValueError(
            "removal_percentage deve essere compreso tra 0 e 1."
        )

    perturbed_graph = graph.copy()

    nodes = list(graph.nodes())

    num_nodes_to_remove = int(
        len(nodes) * removal_percentage
    )

    rng = random.Random(seed)

    nodes_to_remove = rng.sample(
        nodes,
        num_nodes_to_remove
    )

    perturbed_graph.remove_nodes_from(
        nodes_to_remove
    )

    return perturbed_graph