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