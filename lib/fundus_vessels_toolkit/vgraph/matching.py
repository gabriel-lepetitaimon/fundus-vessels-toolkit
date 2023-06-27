import numpy as np

from .vascular_graph import VascularGraph


def simple_graph_matching(
    node1_yx: tuple[np.ndarray, np.ndarray],
    node2_yx: tuple[np.ndarray, np.ndarray],
    max_matching_distance: float | None = None,
    return_distance: bool = False,
    gamma: float = 0,
):
    """
    Match nodes from two graphs based on the euclidian distance between nodes.

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
        return_distance: if True, return the distance between each matched nodes.
        gamma: parameter to exponentially increase the penalty of large distance.
            If 0 (default), the algorithm simply minimize the sum of the distance between matched nodes.
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

    # Compute the euclidian distance between each node
    euclidian_distance = np.linalg.norm(node1_yx[:, None] - node2_yx[None, :], axis=2)
    max_distance = euclidian_distance.max()

    def dist2weight(dist):
        return np.exp(-gamma * dist / 1e3) if gamma != 0 else max_distance - dist

    # Compute the weight as the distance inverse (the hungarian method maximise the sum of the weight)
    weight = dist2weight(euclidian_distance)

    # Set the cost of unmatch nodes to half the inverse of the maximum distance,
    #  so that nodes separated by more than max_distance are better left unmatched.
    min_weight = dist2weight(max_matching_distance) / 2 if max_matching_distance is not None else 0

    # Compute the hungarian matching
    matched_nodes = hungarian(
        weight[None, ...],
        [n1],
        [n2],
        np.repeat([[min_weight]], n1, axis=1),
        np.repeat([[min_weight]], n2, axis=1),
    )[0]
    matched_nodes = np.where(matched_nodes)

    if return_distance:
        return matched_nodes, euclidian_distance[matched_nodes]
    return matched_nodes


def shortest_unmatched_path(adj_list1, adj_list2, matched_nodes: int):
    from .edit_distance_cy import shortest_secondary_path as cython_shortest_path

    primary_nodes = np.arange(matched_nodes)

    nb_node1 = adj_list1.max() + 1
    dist1, backtrack1 = cython_shortest_path(adj_list1, primary_nodes, np.arange(matched_nodes, nb_node1))

    nb_node2 = adj_list2.max() + 1
    dist2, backtrack2 = cython_shortest_path(adj_list2, primary_nodes, np.arange(matched_nodes, nb_node2))

    return dist1, dist2, backtrack1, backtrack2


def backtrack_edges(from_node, to_primary_node, backtrack):
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


def label_edge_diff(vgraph_pred, vgraph_true, nmatch):
    dist_pred, dist_true, backtrack_pred, backtrack_true = shortest_unmatched_path(
        vgraph_pred.node_adjacency_list(), vgraph_true.node_adjacency_list(), nmatch
    )
    edge_id_pred = backtrack_pred[..., 0]
    edge_id_true = backtrack_true[..., 0]

    prim_adj_true = dist_true[:, :nmatch]
    prim_adj_pred = dist_pred[:, :nmatch]

    valid_edges = np.where((prim_adj_true == 1) & (prim_adj_pred == 1))
    fused_edges = np.where((prim_adj_true > 1) & (prim_adj_pred == 1))
    split_edges = np.where((prim_adj_true == 1) & (prim_adj_pred > 1))

    # Assign labels to prediction graph edges
    #  - False positive (default)
    pred_edge_labels = np.zeros((vgraph_pred.branches_count), dtype=np.int8)
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
    true_edge_labels = np.zeros((vgraph_true.branches_count), dtype=np.int8)
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
    return_labels: bool = False,
):
    # Match nodes
    node_match_id1, node_match_id2 = simple_graph_matching(
        graph1.nodes_yx_coord,
        graph2.nodes_yx_coord,
        max_matching_distance=max_matching_distance,
        gamma=2,
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
    unique_branches1[np.concatenate(edges)] = 0
    edges = [backtrack_edges(*e, backtrack=backtrack2) for e in zip(*connected_unmatched_nodes1, strict=True)]
    unique_branches2[np.concatenate(edges)] = 0

    edges = [backtrack_edges(*e, backtrack=backtrack1) for e in zip(*connected_unmatched_nodes2, strict=True)]
    unique_branches1[np.concatenate(edges)] = 0
    edges = [backtrack_edges(*e, backtrack=backtrack2) for e in zip(*connected_unmatched_nodes2, strict=True)]
    unique_branches2[np.concatenate(edges)] = 0

    n_diff1 = unique_branches1.sum()
    n_diff2 = unique_branches2.sum()

    if return_labels:
        return n_diff1, n_diff2, (nb_match, unique_branches1, unique_branches2)
    return n_diff1, n_diff2
