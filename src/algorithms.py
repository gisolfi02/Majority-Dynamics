import networkx as nx

from src.costs import seed_set_cost


def cost_seeds_greedy_f1(
    graph: nx.Graph,
    budget: int,
    costs: dict[int, int]
) -> set[int]:
    """
    Implementa Cost-Seeds-Greedy utilizzando la funzione f1.

    A ogni iterazione viene scelto il nodo u che massimizza:

        Delta_u f1(S) / c(u)

    L'algoritmo termina quando l'aggiunta del nodo successivo
    farebbe superare il budget.

    Parameters
    ----------
    graph : nx.Graph
        Grafo su cui viene eseguito l'algoritmo.

    budget : int
        Budget massimo k.

    costs : dict[int, int]
        Costo associato a ciascun nodo.

    Returns
    -------
    set[int]
        Seed set S con costo <= budget.
    """

    seed_set = set()

    # Numero di vicini appartenenti al seed set
    # attualmente presenti per ciascun nodo.
    seed_neighbors = {
        node: 0
        for node in graph.nodes
    }

    # Soglie majority pre-calcolate.
    thresholds = {
        node: (graph.degree[node] + 1) // 2
        for node in graph.nodes
    }

    current_cost = 0

    while True:

        best_node = None
        best_ratio = -1.0

        for candidate in graph.nodes:

            if candidate in seed_set:
                continue

            # Calcolo di Delta_candidate f1(S).
            marginal_gain = 0

            for neighbor in graph.neighbors(candidate):

                if seed_neighbors[neighbor] < thresholds[neighbor]:
                    marginal_gain += 1

            ratio = marginal_gain / costs[candidate]

            if ratio > best_ratio:
                best_ratio = ratio
                best_node = candidate

        # Non ci sono più candidati.
        if best_node is None:
            break

        # La slide aggiunge il nodo e si ferma quando
        # il nuovo seed set supera il budget.
        # Equivalentemente, restituiamo direttamente il
        # seed set precedente.
        if current_cost + costs[best_node] > budget:
            break

        seed_set.add(best_node)
        current_cost += costs[best_node]

        # L'aggiunta di best_node fa aumentare di 1
        # il numero di vicini-seed di tutti i suoi vicini.
        for neighbor in graph.neighbors(best_node):
            seed_neighbors[neighbor] += 1

    return seed_set


def cost_seeds_greedy_f2(
    graph: nx.Graph,
    budget: int,
    costs: dict[int, int]
) -> set[int]:
    """
    Implementa Cost-Seeds-Greedy utilizzando la funzione f2.

    A ogni iterazione viene scelto il nodo u che massimizza:

        Delta_u f2(S) / c(u)

    dove:

        Delta_u f2(S) =
            sum_{v in N(u)}
            max(t(v) - |N(v) intersect S|, 0)

    con:

        t(v) = ceil(d(v) / 2)
    """

    seed_set = set()

    # Numero di vicini appartenenti al seed set
    # per ciascun nodo.
    seed_neighbors = {
        node: 0
        for node in graph.nodes
    }

    # Soglie majority.
    thresholds = {
        node: (graph.degree[node] + 1) // 2
        for node in graph.nodes
    }

    current_cost = 0

    while True:

        best_node = None
        best_ratio = -1.0

        for candidate in graph.nodes:

            if candidate in seed_set:
                continue

            marginal_gain = 0

            for neighbor in graph.neighbors(candidate):

                remaining = (
                    thresholds[neighbor]
                    - seed_neighbors[neighbor]
                )

                if remaining > 0:
                    marginal_gain += remaining

            ratio = marginal_gain / costs[candidate]

            if ratio > best_ratio:
                best_ratio = ratio
                best_node = candidate

        if best_node is None:
            break

        if current_cost + costs[best_node] > budget:
            break

        seed_set.add(best_node)
        current_cost += costs[best_node]

        for neighbor in graph.neighbors(best_node):
            seed_neighbors[neighbor] += 1

    return seed_set