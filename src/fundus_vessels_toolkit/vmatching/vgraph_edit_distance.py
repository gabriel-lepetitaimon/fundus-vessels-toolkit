from typing import Optional

import numpy as np
import torch

from ..utils.cpp_extensions.fvt_cpp import backtrack_edges as backtrack_edges_cpp
from ..utils.cpp_extensions.fvt_cpp import shortest_secondary_path as shortest_sec_path_cpp
from ..utils.torch import autocast_torch
from ..utils.typing import Int2DArray, IntPairArray, IntPairMap
from ..vascular_data_objects.vgraph import VGraph
from .node_matching import match_nodes_by_distance


@autocast_torch
def shortest_secondary_path(
    edge_list: torch.Tensor,
    primary_nodes: torch.Tensor,
    n_nodes: Optional[int] = None,
    directed_edge: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute shortest path between every primary nodes without going through other primary nodes.

    Parameters
    ----------
    edge_list : torch.Tensor
        The edge list of the graph as a 2D tensor of shape (E, 2) where E is the number of edges in the graph. Each row contains the index of the two nodes connected by the edge.
    primary_nodes : torch.Tensor
        The nodes in the graph that are considered primary.
    n_nodes : Optional[int], optional
        The total number of nodes in the graph. If None (by default), it is inferred from the edge list.

    directed_edge : bool, optional
        If True, the edges in the edge list are considered as directed.

    Returns
    -------
    dist: torch.Tensor
        The distance matrix between primary nodes as an integer matrix of shape (nP, nP) where nP is the number of primary nodes. If no path exists between two primary nodes, the corresponding entry in the matrix is -1.

    backtrack: torch.Tensor
        The backtrack matrix as an integer tensor of shape (nP, N, 2) where nP is the number of primary nodes and N the total number of nodes in the graph. For each pair (p, n) the matrix contains the index of the edge and the index of the next node on the path from the node n to the primary node p. If no path exists between the two nodes, the matrix contains (-1, -1).
    """  # noqa: E501
    if n_nodes is None:
        n_nodes = int(edge_list.max()) + 1
    return shortest_sec_path_cpp(edge_list.cpu().int(), primary_nodes.cpu().int(), n_nodes, directed_edge)


def shortest_unmatched_path(
    adj_list1: IntPairArray, adj_list2: IntPairArray, matched_nodes: int
) -> tuple[Int2DArray, Int2DArray, IntPairMap, IntPairMap]:
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
    primary_nodes = np.arange(matched_nodes)

    nb_node1 = adj_list1.max() + 1
    dist1, backtrack1 = shortest_secondary_path(adj_list1, primary_nodes, np.arange(matched_nodes, nb_node1))

    nb_node2 = adj_list2.max() + 1
    dist2, backtrack2 = shortest_secondary_path(adj_list2, primary_nodes, np.arange(matched_nodes, nb_node2))

    return dist1, dist2, backtrack1, backtrack2


@autocast_torch
def backtrack_edges(
    src_dst_nodes: torch.Tensor, backtrack: torch.Tensor, primary_nodes: torch.Tensor
) -> list[list[int]]:
    """Compute the list of edges between pair of nodes based on the backtrack matrix.

    Parameters
    ----------
    src_dst_nodes : torch.Tensor
        The source and destination nodes as a 2D tensor of shape (P, 2) where P is the number of pairs of nodes.

        The first element of each pair must be a primary node index according to the first dimension of the backtrack matrix, the second element can be any node index according to the second dimension of the backtrack matrix.

    backtrack :
        The backtrack matrix as returned by shortest_unmatched_path. The matrix must be of shape (nP, N, 2) with nP<N where nP is the number of primary (or matched) nodes and N the total number of nodes in the graph.


    Returns
    -------
    list[list[int]]
        A list of P lists of edge indices corresponding to the path between the source and destination nodes. If no path exists between the two nodes, the corresponding list is empty.
    """  # noqa: E501
    assert backtrack.dim() == 3, "Backtrack matrix must be of shape (nP, N, 2)"
    nP, N, _ = backtrack.shape
    assert src_dst_nodes.dim() == 2 and src_dst_nodes.size(1) == 2, "src_dst_nodes must be of shape (P, 2)"
    assert primary_nodes.shape == (nP,), "primary_nodes must be of shape (nP,)"
    assert torch.all(src_dst_nodes[:, 0] < nP), (
        "The first element of each pair in src_dst_nodes must be a primary node index according to the first dimension of the backtrack matrix"  # noqa: E501
    )
    assert torch.all(src_dst_nodes[:, 1] < N), (
        "The second element of each pair in src_dst_nodes must be a node index according to the second dimension of the backtrack matrix"  # noqa: E501
    )
    return backtrack_edges_cpp(backtrack.cpu().int(), src_dst_nodes.cpu().int(), primary_nodes.cpu().int())


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
    pred_edge_labels = np.zeros((graph_pred.branch_count), dtype=np.int8)
    #  - True positive
    pred_edge_labels[edge_id_pred[valid_edges]] = 1
    #  - Split edges (single branch in true, multiple branch in pred)
    fused_pred_edge = np.concatenate(
        backtrack_edges(split_edges, backtrack=backtrack_pred, primary_nodes=np.arange(n_match))
    )
    pred_edge_labels[fused_pred_edge] = 2
    #  - Fused edges (multiple branch in true, single branch in pred)
    pred_edge_labels[edge_id_pred[fused_edges]] = 3

    # Assign labels to true graph edges
    #  - False negative (default)
    true_edge_labels = np.zeros((graph_true.branch_count), dtype=np.int8)
    #  - True positive
    true_edge_labels[edge_id_true[valid_edges]] = 1
    #  - Fused edges (multiple branch in true, single branch in pred)
    fused_true_edge = np.concatenate(
        backtrack_edges(fused_edges, backtrack=backtrack_true, primary_nodes=np.arange(n_match))
    )
    true_edge_labels[fused_true_edge] = 2
    #  - Split edges (single branch in true, multiple branch in pred)
    true_edge_labels[edge_id_true[split_edges]] = 3

    return pred_edge_labels, true_edge_labels


def naive_edit_distance(
    graph1: VGraph,
    graph2: VGraph,
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
    node_match_id1, node_match_id2 = match_nodes_by_distance(
        graph1.node_coord(),
        graph2.node_coord(),
        max_matching_distance=max_matching_distance,
        density_sigma=density_matching_sigma,
        min_distance=min_distance,
    )
    graph1.reindex_nodes(node_match_id1, inverse_lookup=True, inplace=True)
    graph2.reindex_nodes(node_match_id2, inverse_lookup=True, inplace=True)
    nb_match = len(node_match_id1)

    # Match edges
    dist1, dist2, backtrack1, backtrack2 = shortest_unmatched_path(graph1.branch_list, graph2.branch_list, nb_match)

    branches_id1 = backtrack1[..., 0]
    branches_id2 = backtrack2[..., 0]
    match_dist1 = dist1[:, :nb_match]
    match_dist2 = dist2[:, :nb_match]

    connected_matched_nodes = np.where((match_dist1 == 1) & (match_dist2 == 1))
    connected_unmatched_nodes1 = np.where((match_dist1 > 1) & (match_dist2 >= 1))
    connected_unmatched_nodes2 = np.where((match_dist1 >= 1) & (match_dist2 > 1))

    # Count branch unique to each graph
    #  - Branches are considered unique by default
    unique_branches1 = np.ones((graph1.branch_count), dtype=np.int8)
    unique_branches2 = np.ones((graph2.branch_count), dtype=np.int8)

    #  Remove branches that are matched
    unique_branches1[branches_id1[connected_matched_nodes]] = 0
    unique_branches2[branches_id2[connected_matched_nodes]] = 0

    #  Remove branches that connect matched nodes
    edges = [
        backtrack_edges(*e, backtrack=backtrack1, primary_nodes=np.arange(nb_match))
        for e in zip(*connected_unmatched_nodes1, strict=True)
    ]
    if len(edges) > 0:
        unique_branches1[np.concatenate(edges)] = 0
    edges = [
        backtrack_edges(*e, backtrack=backtrack2, primary_nodes=np.arange(nb_match))
        for e in zip(*connected_unmatched_nodes1, strict=True)
    ]
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


########################################################################################################################
