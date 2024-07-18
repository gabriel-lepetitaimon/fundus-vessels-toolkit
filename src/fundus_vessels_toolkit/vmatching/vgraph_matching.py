import math
from itertools import product as iter_product
from typing import Iterable

import numpy as np

from ..vascular_data_objects import VGraph
from .node_matching import junctions_matching, ransac_refine_node_matching


def multi_vgraph_registration(vgraphs: Iterable[VGraph], ransac_retry=4) -> list[np.ndarray]:
    """
    Register multiple vascular graphs together.

    Parameters
    ----------
    vgraphs : Iterable[VGraph]
        The vascular graphs to register.

    ransac_retry : int, optional
        The number of retries for RANSAC convergence. By default 4.

    Returns
    -------
    list[np.ndarray]
        The transformations to apply to each vascular graph to register them together.

    Raises
    ------
    ValueError
        If the provided vascular graphs do not overlap enough to be registered together

    """
    import networkx as nx

    n_graph = len(vgraphs)
    matching = {}

    # Find the transformation between each pairs of graphs
    for i1, i2 in iter_product(range(n_graph), repeat=2):
        if i1 >= i2:
            continue
        g1, g2 = vgraphs[i1], vgraphs[i2]
        n1, n2 = junctions_matching(g1, g2)
        while ransac_retry > 0:
            try:
                T, matched_nodes, mean_error = ransac_refine_node_matching(g1, g2, n1, n2, return_mean_error=True)
                break
            except ValueError:
                ransac_retry -= 1
                continue
        else:
            continue

        matching[(i1, i2)] = (T, matched_nodes.shape[1], mean_error)

    # Find the spanning tree of graphs providing the least registration mean error
    G = nx.Graph()
    G.add_nodes_from(range(n_graph))
    G.add_weighted_edges_from([(i1, i2, -error * math.sqrt(n)) for (i1, i2), (_, n, error) in matching.items()])
    ST = nx.minimum_spanning_tree(G)

    # Check that ST is a single connected component
    subtrees = list(nx.connected_components(ST))
    if len(subtrees) != 1:
        raise ValueError(
            "The provided vascular graph does not overlap enough to be registered together.\n"
            f"The registration identified the following clusters: {subtrees}."
        )

    # Accumulate the transformation from all other graphs to the center of the spanning tree
    root = nx.center(ST)[0]
    projection_type = type(next(iter(matching.values()))[0])
    transformations = {root: projection_type.identity()}
    for i0, i1 in nx.bfs_edges(ST, root):
        T1 = matching[(i0, i1)][0]
        if i0 in transformations:
            T0 = transformations[i0]
            T1 = T1.invert()
            transformations[i1] = T0.compose(T1)
        else:
            T0 = transformations[i1]
            transformations[i0] = T0.compose(T1)

    return list(transformations[_] for _ in range(n_graph))
