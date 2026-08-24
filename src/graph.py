from pathlib import Path

import networkx as nx


def load_graph(file_path: str) -> nx.Graph:
    """
    Carica un grafo non orientato da un file edge list.

    Ogni riga del dataset deve contenere due identificativi
    di nodi separati da uno spazio:

        u v

    Parameters
    ----------
    file_path : str
        Percorso del file contenente gli archi.

    Returns
    -------
    nx.Graph
        Il grafo non orientato caricato dal dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {path}"
        )

    graph = nx.read_edgelist(
        path,
        nodetype=int,
        create_using=nx.Graph()
    )

    return graph

def get_graph_statistics(graph: nx.Graph) -> dict:
    """
    Calcola alcune statistiche descrittive della rete.
    """

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    degrees = [degree for _, degree in graph.degree()]

    connected_components = list(nx.connected_components(graph))
    largest_component_size = max(
        len(component)
        for component in connected_components
    )

    statistics = {
        "nodes": num_nodes,
        "edges": num_edges,
        "average_degree": sum(degrees) / num_nodes,
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "density": nx.density(graph),
        "connected_components": len(connected_components),
        "largest_component_size": largest_component_size,
        "average_clustering": nx.average_clustering(graph),
    }

    return statistics