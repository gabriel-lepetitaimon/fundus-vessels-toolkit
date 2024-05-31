from typing import Tuple

import numpy as np

from ..vascular_graph import VascularGraph


def simple_graph_matching(
    node1_yx: tuple[np.ndarray, np.ndarray],
    node2_yx: tuple[np.ndarray, np.ndarray],
    max_matching_distance: float | None = None,
    min_distance: float | None = None,
    density_sigma: float | None = None,
    gamma: float = 0,
    return_distance: bool = False,
):
    """
    Match nodes from two graphs based on the euclidean distance between nodes.

    Each node from the first graph is matched to the closest node from the second graph, if their distance is below
      max_matching_distance. When multiple nodes from both graph are near each other, minimize the sum of the distance
      between matched nodes.

    This implementation use the Hungarian algorithm to maximize the sum of the inverse of the distance between
        matched nodes.


    Args:
        node1_yx: tuple (y, x), where y and x are vectors of the same length and encode the coordinates of the nodes
                    of the first graph.
        node2_yx: same format as node1_yx but for the second graph.
        max_matching_distance: maximum distance between two nodes to be considered as a match.
        min_distance: set a lower bound to the distance between two nodes to prevent immediate match of superposed nodes.
        gamma: parameter to exponentially increase the penalty of large distance.
            If 0 (default), the algorithm simply minimize the sum of the distance between matched nodes.
        return_distance: if True, return the distance between each matched nodes.
    Returns:
        A tuple (node1_matched, node2_matched), where node1_matched and node2_matched are vectors of the same length.
          Every (node1_matched[i], node2_matched[i]) encode a match, node1_matched being the index of a node from the
          first graph and node2_matched the index of the corresponding node from the second graph.

        If return_distance is True, returns ((node1_matched, node2_matched), nodes_distance), where nodes_distance
          is a vector of the same length as node1_matched and node2_matched, and encode the distance between each
          matched nodes.
    """
    from pygmtools.linear_solvers import hungarian

    if isinstance(node1_yx, tuple) and len(node1_yx) == 2:
        node1_yx = np.stack(node1_yx, axis=1)
    if isinstance(node2_yx, tuple) and len(node2_yx) == 2:
        node2_yx = np.stack(node2_yx, axis=1)
    n1 = len(node1_yx)
    n2 = len(node2_yx)

    # Compute the euclidean distance between each node
    cross_euclidean_distance = np.linalg.norm(node1_yx[:, None] - node2_yx[None, :], axis=2)
    max_distance = cross_euclidean_distance.max()

    if gamma > 0:

        def dist2weight(dist):
            return np.exp(-gamma * dist / 1e3)

    else:

        def dist2weight(dist):
            return max_distance - dist

    # Compute the weight as the distance inverse (the Hungarian method maximise the sum of the weight)
    weight = dist2weight(cross_euclidean_distance)

    # Set the cost of not matched nodes to half the inverse of the maximum distance,
    #  so that nodes separated by more than max_distance are better left unmatched.
    if density_sigma is None:
        min_weight = dist2weight(max_matching_distance) / 2 if max_matching_distance is not None else 0
        min_weight1 = np.repeat([min_weight], n1, axis=0)
        min_weight2 = np.repeat([min_weight], n2, axis=0)
    else:
        if min_distance is None:
            min_distance = np.clip(max_matching_distance / 20, 1)

        # Compute node density as the sum of the Gaussian kernel centered on each node
        def influence(x):
            return np.exp(-(x**2) / (2 * density_sigma**2))

        self_euclidean_distance1 = np.linalg.norm(node1_yx[:, None] - node1_yx[None, :], axis=2)
        self_euclidean_distance2 = np.linalg.norm(node2_yx[:, None] - node2_yx[None, :], axis=2)
        self_density1 = np.clip(influence(self_euclidean_distance1).sum(axis=1) - 1, 0, 1)
        self_density2 = np.clip(influence(self_euclidean_distance2).sum(axis=1) - 1, 0, 1)

        cross_density = influence(cross_euclidean_distance)
        cross_density1 = np.clip(cross_density.sum(axis=1), 0, 1)
        cross_density2 = np.clip(cross_density.sum(axis=0), 0, 1)

        density1 = (self_density1 + cross_density1) / 2
        density2 = (self_density2 + cross_density2) / 2

        # Compute a maximum matching distance per node based on the density of its closest neighbor in the other graph
        closest_node1 = cross_euclidean_distance.argmin(axis=1)
        closest_node2 = cross_euclidean_distance.argmin(axis=0)
        factor1 = 1 - density2[closest_node1]
        factor2 = 1 - density1[closest_node2]

        max_matching_distance1 = np.clip(max_matching_distance * factor1, min_distance, max_matching_distance)
        max_matching_distance2 = np.clip(max_matching_distance * factor2, min_distance, max_matching_distance)

        # for i, (max_d, closest_node) in enumerate(zip(max_matching_distance1, closest_node1, strict=True)):
        #     print(
        #         f"[{str(i): >3}] : {max_d:.2f} ({str(closest_node): >3} -> {density2[closest_node]:.2f}: "
        #         f"{cross_density2[closest_node]:.2f} & {self_density2[closest_node]:.2f})"
        #     )

        min_weight1 = dist2weight(max_matching_distance1) / 2
        min_weight2 = dist2weight(max_matching_distance2) / 2

    # Compute the Hungarian matching
    matched_nodes = hungarian(
        weight,
        n1,
        n2,
        min_weight1,
        min_weight2,
    )
    matched_nodes = np.where(matched_nodes)

    if return_distance:
        return matched_nodes, cross_euclidean_distance[matched_nodes]
    return matched_nodes


def shortest_unmatched_path(
    adj_list1: np.ndarray, adj_list2: np.ndarray, matched_nodes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the shortest path connecting unmatched nodes to matched nodes, without using matched nodes.
    This function perform the same operation individually on the two graphs.

    Parameters
    ----------
    adj_list1 :
        The adjacency list of the first graph of shape(E1, 2) and maximum value N1.

        The adjacency list is a 2D array of shape (E1, 2) where N is the number of edges in the graph. Each row contains the index of the two nodes connected by the edge.

    adj_list2 :
        The adjacency list of the second graph of shape (E2, 2) and maximum value N2.

    matched_nodes : int
        The number of matched nodes between the two graphs. ``matched_nodes`` must be lower to N1 and N2.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

        - A distance matrix of shape ``(matched_nodes, N1)`` with the distance between matched nodes and each node of graph 1. (If no path exists between the two nodes, the matrix contains -1.)
        - A distance matrix of shape ``(matched_nodes, N2)`` with the distance between matched nodes and each node of graph 2. (If no path exists between the two nodes, the matrix contains -1.)
        - A backtrack matrix of shape ``(matched_nodes, N1, 2)`` with the index of the edge and the index of the next node on the path from the node N1 to the primary node N. If no path exists between the two nodes, the matrix contains (-1, -1).
        - A backtrack matrix of shape ``(matched_nodes, N2, 2)`` with the index of the edge and the index of the next node on the path from the node N2 to the primary node N. If no path exists between the two nodes, the matrix contains (-1, -1).
    """  # noqa: E501
    from .edit_distance_cy import shortest_secondary_path as cython_shortest_path

    primary_nodes = np.arange(matched_nodes)

    nb_node1 = adj_list1.max() + 1
    dist1, backtrack1 = cython_shortest_path(adj_list1, primary_nodes, np.arange(matched_nodes, nb_node1))

    nb_node2 = adj_list2.max() + 1
    dist2, backtrack2 = cython_shortest_path(adj_list2, primary_nodes, np.arange(matched_nodes, nb_node2))

    return dist1, dist2, backtrack1, backtrack2


def backtrack_edges(from_node: int, to_primary_node: int, backtrack: np.ndarray) -> list[int]:
    """Compute the list of edges between two nodes based on the backtrack matrix.

    Parameters
    ----------
    from_node :
        Index of the starting node.

    to_primary_node :
        Index of the ending node. The ending node must be a primary node, ie its index must be lower than backtrack.shape[0].

    backtrack :
        The backtrack matrix as returned by shortest_unmatched_path. The matrix must be of shape (N, m, 2) with N<m where N is the number of primary (or matched) nodes and m the total number of nodes in the graph.
        For each pair (N, m) the matrix contains the index of the edge and the index of the next node on the path from the node m to the primary node N. If no path exists between the two nodes, the matrix contains (-1, -1).


    Returns
    -------
    list[int]
        The list of edges between the two nodes. If no path exists between the two nodes, returns an empty list.
    """  # noqa: E501
    edges = []
    backtrack = backtrack[to_primary_node]
    node = from_node

    if backtrack[node, 1] == -1:
        return []

    while node != to_primary_node:
        next_edge, next_node = backtrack[node]
        edges.append(next_edge)
        node = next_node

    return edges


def label_edge_diff(graph_pred, graph_true, n_match):
    dist_pred, dist_true, backtrack_pred, backtrack_true = shortest_unmatched_path(
        graph_pred.node_adjacency_list(), graph_true.node_adjacency_list(), n_match
    )
    edge_id_pred = backtrack_pred[..., 0]
    edge_id_true = backtrack_true[..., 0]

    prim_adj_true = dist_true[:, :n_match]
    prim_adj_pred = dist_pred[:, :n_match]

    valid_edges = np.where((prim_adj_true == 1) & (prim_adj_pred == 1))
    fused_edges = np.where((prim_adj_true > 1) & (prim_adj_pred == 1))
    split_edges = np.where((prim_adj_true == 1) & (prim_adj_pred > 1))

    # Assign labels to prediction graph edges
    #  - False positive (default)
    pred_edge_labels = np.zeros((graph_pred.branches_count), dtype=np.int8)
    #  - True positive
    pred_edge_labels[edge_id_pred[valid_edges]] = 1
    #  - Split edges (single branch in true, multiple branch in pred)
    fused_pred_edge = np.concatenate(
        [backtrack_edges(*e, backtrack=backtrack_pred) for e in zip(*split_edges, strict=True)]
    )
    pred_edge_labels[fused_pred_edge] = 2
    #  - Fused edges (multiple branch in true, single branch in pred)
    pred_edge_labels[edge_id_pred[fused_edges]] = 3

    # Assign labels to true graph edges
    #  - False negative (default)
    true_edge_labels = np.zeros((graph_true.branches_count), dtype=np.int8)
    #  - True positive
    true_edge_labels[edge_id_true[valid_edges]] = 1
    #  - Fused edges (multiple branch in true, single branch in pred)
    fused_true_edge = np.concatenate(
        [backtrack_edges(*e, backtrack=backtrack_true) for e in zip(*fused_edges, strict=True)]
    )
    true_edge_labels[fused_true_edge] = 2
    #  - Split edges (single branch in true, multiple branch in pred)
    true_edge_labels[edge_id_true[split_edges]] = 3

    return pred_edge_labels, true_edge_labels


def naive_edit_distance(
    graph1: VascularGraph,
    graph2: VascularGraph,
    max_matching_distance: float | None = None,
    density_matching_sigma: float | None = None,
    min_distance: float | None = None,
    return_labels: bool = False,
) -> tuple[int, int] | tuple[int, int, tuple[int, np.ndarray, np.ndarray]]:
    """Compute the naive edit distance between two graphs. The two graphs must be geometrically similar as the node will first be paired based on their euclidean distance.

    Parameters
    ----------
    graph1 :
        The first graph to compare.

    graph2 :
        The second graph to compare.

    max_matching_distance : float | None, optional
        The maximum distance between two nodes to be considered as a match.
        By default: None.

    density_matching_sigma : float | None, optional
        The standard deviation of the Gaussian kernel used to compute the node density used to reduce the maximum matching distance for cluttered nodes.

    min_distance : float | None, optional
        Set a lower bound to the distance between two nodes to prevent immediate match of superposed nodes.

    return_labels : bool, optional
        If True, return the labels of the edges of the two graphs.
        By default: False.

    Returns
    -------
    tuple[int, int] | tuple[int, int, tuple[int, np.ndarray, np.ndarray]]
        The number of not-matched edges in each graph. If return_labels is True, also returns the a tuple with the number of matched edges and two 1D binary arrays which indicate for each edge of graph 1 and 2 if it is matched (0) or not-matched (1).

    """  # noqa: E501
    # Match nodes
    node_match_id1, node_match_id2 = simple_graph_matching(
        graph1.nodes_yx_coord,
        graph2.nodes_yx_coord,
        max_matching_distance=max_matching_distance,
        density_sigma=density_matching_sigma,
        min_distance=min_distance,
    )
    graph1.shuffle_nodes(node_match_id1)
    graph2.shuffle_nodes(node_match_id2)
    nb_match = len(node_match_id1)

    # Match edges
    dist1, dist2, backtrack1, backtrack2 = shortest_unmatched_path(
        graph1.node_adjacency_list(), graph2.node_adjacency_list(), nb_match
    )

    branches_id1 = backtrack1[..., 0]
    branches_id2 = backtrack2[..., 0]
    match_dist1 = dist1[:, :nb_match]
    match_dist2 = dist2[:, :nb_match]

    connected_matched_nodes = np.where((match_dist1 == 1) & (match_dist2 == 1))
    connected_unmatched_nodes1 = np.where((match_dist1 > 1) & (match_dist2 >= 1))
    connected_unmatched_nodes2 = np.where((match_dist1 >= 1) & (match_dist2 > 1))

    # Count branch unique to each graph
    #  - Branches are considered unique by default
    unique_branches1 = np.ones((graph1.branches_count), dtype=np.int8)
    unique_branches2 = np.ones((graph2.branches_count), dtype=np.int8)

    #  Remove branches that are matched
    unique_branches1[branches_id1[connected_matched_nodes]] = 0
    unique_branches2[branches_id2[connected_matched_nodes]] = 0

    #  Remove branches that connect matched nodes
    edges = [backtrack_edges(*e, backtrack=backtrack1) for e in zip(*connected_unmatched_nodes1, strict=True)]
    if len(edges) > 0:
        unique_branches1[np.concatenate(edges)] = 0
    edges = [backtrack_edges(*e, backtrack=backtrack2) for e in zip(*connected_unmatched_nodes1, strict=True)]
    if len(edges) > 0:
        unique_branches2[np.concatenate(edges)] = 0

    edges = [backtrack_edges(*e, backtrack=backtrack1) for e in zip(*connected_unmatched_nodes2, strict=True)]
    if len(edges) > 0:
        unique_branches1[np.concatenate(edges)] = 0
    edges = [backtrack_edges(*e, backtrack=backtrack2) for e in zip(*connected_unmatched_nodes2, strict=True)]
    if len(edges) > 0:
        unique_branches2[np.concatenate(edges)] = 0

    n_diff1 = unique_branches1.sum()
    n_diff2 = unique_branches2.sum()

    if return_labels:
        return n_diff1, n_diff2, (nb_match, unique_branches1, unique_branches2)
    return n_diff1, n_diff2
