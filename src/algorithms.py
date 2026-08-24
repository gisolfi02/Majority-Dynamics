import networkx as nx

from src.costs import seed_set_cost
from src.diffusion import majority_cascade
from collections import deque



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

def _incremental_cascade(
    candidate: int,
    current_active: set[int],
    active_neighbors: dict[int, int],
    thresholds: dict[int, int],
    neighborhoods: dict[int, tuple[int, ...]]
) -> set[int]:
    """
    Calcola solamente le NUOVE attivazioni provocate
    dall'aggiunta di candidate come seed.

    current_active contiene già Inf[G, S].

    Restituisce:
        Inf[G, S U {candidate}] - Inf[G, S]
    """

    if candidate in current_active:
        return set()

    # Il candidato diventa direttamente attivo perché seed.
    newly_active = {candidate}

    queue = deque([candidate])

    # Incrementi temporanei del numero di vicini attivi.
    # Non modifichiamo active_neighbors perché stiamo
    # soltanto simulando il candidato.
    neighbor_deltas: dict[int, int] = {}

    while queue:

        node = queue.popleft()

        for neighbor in neighborhoods[node]:

            neighbor_deltas[neighbor] = (
                neighbor_deltas.get(neighbor, 0) + 1
            )

            # È già attivo nella cascade corrente.
            if neighbor in current_active:
                continue

            # È già stato attivato durante questa simulazione.
            if neighbor in newly_active:
                continue

            total_active_neighbors = (
                active_neighbors[neighbor]
                + neighbor_deltas[neighbor]
            )

            if total_active_neighbors >= thresholds[neighbor]:
                newly_active.add(neighbor)
                queue.append(neighbor)

    return newly_active


def cascade_gain_greedy(
    graph: nx.Graph,
    budget: int,
    costs: dict[int, int]
) -> set[int]:
    """
    Cascade-Gain Greedy (CGG).

    A ogni iterazione seleziona il nodo u che massimizza:

        (|Inf[G, S U {u}]| - |Inf[G, S]|) / c(u)

    La valutazione della cascade viene effettuata
    incrementalmente per evitare di ricalcolarla
    completamente per ogni candidato.
    """

    nodes = list(graph.nodes)

    # Pre-calcoliamo i vicinati.
    # Evitiamo migliaia di chiamate ripetute a
    # graph.neighbors().
    neighborhoods = {
        node: tuple(graph.neighbors(node))
        for node in nodes
    }

    # Pre-calcoliamo le soglie majority.
    thresholds = {
        node: (graph.degree[node] + 1) // 2
        for node in nodes
    }

    seed_set = set()

    current_cost = 0

    # Cascade relativa al seed set iniziale vuoto.
    current_active, _ = majority_cascade(
        graph,
        seed_set
    )

    # Numero di vicini attivi di ogni nodo nella
    # cascade corrente Inf[G, S].
    active_neighbors = {
        node: 0
        for node in nodes
    }

    for active_node in current_active:
        for neighbor in neighborhoods[active_node]:
            active_neighbors[neighbor] += 1

    while True:

        best_node = None
        best_ratio = -1.0
        best_gain = -1

        # Memorizziamo anche le nuove attivazioni del
        # candidato migliore, così non dobbiamo
        # risimularlo dopo la selezione.
        best_newly_active = None

        for candidate in nodes:

            if candidate in seed_set:
                continue

            # Se è già stato attivato dalla cascade,
            # renderlo seed non porta alcun guadagno.
            if candidate in current_active:
                continue

            # Deve rispettare il budget residuo.
            if (
                current_cost + costs[candidate]
                > budget
            ):
                continue

            newly_active = _incremental_cascade(
                candidate,
                current_active,
                active_neighbors,
                thresholds,
                neighborhoods
            )

            # Questo equivale a:
            #
            # |Inf[G,S U {candidate}]|
            # -
            # |Inf[G,S]|
            #
            marginal_gain = len(newly_active)

            ratio = (
                marginal_gain
                / costs[candidate]
            )

            if (
                ratio > best_ratio
                or (
                    ratio == best_ratio
                    and marginal_gain > best_gain
                )
            ):
                best_ratio = ratio
                best_gain = marginal_gain
                best_node = candidate
                best_newly_active = newly_active

        # Nessun altro candidato compatibile.
        if best_node is None:
            break

        # Aggiungiamo definitivamente il nodo scelto.
        seed_set.add(best_node)

        current_cost += costs[best_node]

        # Le attivazioni simulate per il best_node
        # diventano ora reali.
        current_active.update(best_newly_active)

        # Aggiorniamo il numero di vicini attivi
        # solamente per i nuovi nodi attivati.
        for active_node in best_newly_active:
            for neighbor in neighborhoods[active_node]:
                active_neighbors[neighbor] += 1

    return seed_set