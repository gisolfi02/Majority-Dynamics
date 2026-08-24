import networkx as nx


def majority_threshold(graph: nx.Graph, node: int) -> int:
    """
    Restituisce la soglia majority del nodo.

    Un nodo v si attiva quando almeno ceil(d(v) / 2)
    dei suoi vicini sono attivi.
    """

    degree = graph.degree[node]

    return (degree + 1) // 2


def majority_cascade(
    graph: nx.Graph,
    seeds: set[int]
) -> tuple[set[int], int]:
    """
    Simula il processo di Majority Cascade.

    Parameters
    ----------
    graph : nx.Graph
        Grafo su cui viene eseguita la diffusione.

    seeds : set[int]
        Insieme dei nodi inizialmente attivi.

    Returns
    -------
    active : set[int]
        Insieme finale Inf[G, S] dei nodi attivati.

    rounds : int
        Numero di round necessari affinché la cascade termini.
    """

    seeds = set(seeds)

    # Controlliamo che tutti i seed appartengano al grafo.
    invalid_seeds = seeds - set(graph.nodes)

    if invalid_seeds:
        raise ValueError(
            f"I seguenti seed non appartengono al grafo: "
            f"{invalid_seeds}"
        )

    # Inf[S, 0] = S
    active = set(seeds)

    rounds = 0

    # Per ogni nodo manteniamo il numero di vicini attivi.
    active_neighbors = {
        node: 0
        for node in graph.nodes
    }

    # Inizializziamo i conteggi considerando i seed.
    for seed in active:
        for neighbor in graph.neighbors(seed):
            active_neighbors[neighbor] += 1

    # Nodi che possono attivarsi al primo round.
    frontier = {
        node
        for node in graph.nodes
        if node not in active
        and active_neighbors[node]
        >= majority_threshold(graph, node)
    }

    while frontier:

        # Tutti questi nodi si attivano nello stesso round.
        active.update(frontier)

        rounds += 1

        affected_nodes = set()

        # L'attivazione dei nodi del round corrente
        # modifica il conteggio dei loro vicini.
        for node in frontier:

            for neighbor in graph.neighbors(node):

                active_neighbors[neighbor] += 1

                if neighbor not in active:
                    affected_nodes.add(neighbor)

        # Determiniamo i nodi che potranno attivarsi
        # nel round successivo.
        frontier = {
            node
            for node in affected_nodes
            if active_neighbors[node]
            >= majority_threshold(graph, node)
        }

    return active, rounds