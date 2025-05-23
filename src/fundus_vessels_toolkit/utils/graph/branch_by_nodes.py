import warnings
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..binary_mask import index_to_mask
from ..geometric import Point, Rect
from ..lookup_array import add_empty_to_lookup, apply_lookup, apply_lookup_on_coordinates, invert_lookup


def branch_by_nodes_to_adjacency_list(branches_by_nodes: np.ndarray, sorted=False) -> np.ndarray:
    """
    Convert a connectivity matrix of branches and nodes to an adjacency list of branch. Each branch is represented
    by a tuple (node1, node2) where node1 and node2 are the two nodes connected by the branch.

    Args:
        branches_by_nodes: A 2D array of shape (nb_branches, nb_nodes) containing the connectivity matrix of branches
                             and nodes. branches_by_nodes[i, j] is True if the branch i is connected to the node j.
        sorted: If True, the adjacency list is sorted by the first node id.

    Returns:
        A 2d array of shape (nb_branches, 2) containing the adjacency list of branches. The first node id is always
        lower than the second node id.
    """
    node1_ids = np.argmax(branches_by_nodes, axis=1)
    node2_ids = branches_by_nodes.shape[1] - np.argmax(branches_by_nodes[:, ::-1], axis=1) - 1

    if sorted:
        sort_id = np.argsort(node1_ids)
        node1_ids = node1_ids[sort_id]
        node2_ids = node2_ids[sort_id]

    return np.stack([node1_ids, node2_ids], axis=1)


def branches_by_nodes_to_edge_list(branches_by_nodes):
    assert branches_by_nodes.ndim == 2, "branches_by_nodes must be a 2D array"
    assert np.all(np.sum(branches_by_nodes, axis=1) == 2), "Each branch must connect exactly 2 nodes"

    n1 = np.argmax(branches_by_nodes, axis=1)
    n2 = np.argmax(branches_by_nodes[:, ::-1], axis=1)
    return np.stack((n1, n2), axis=1)


def branches_by_nodes_to_node_graph(branches_by_nodes, node_pos=None):
    branches = np.arange(branches_by_nodes.shape[0]) + 1
    branches_by_nodes = branches_by_nodes.astype(bool)
    node_adjacency = branches_by_nodes.T @ (branches_by_nodes * branches[:, None])
    graph = nx.from_numpy_array((node_adjacency > 0) & (~np.eye(branches_by_nodes.shape[1], dtype=bool)))
    if node_pos is not None:
        node_y, node_x = node_pos
        nx.set_node_attributes(graph, node_y, "y")
        nx.set_node_attributes(graph, node_x, "x")
    for edge in graph.edges():
        graph.edges[edge]["branch"] = node_adjacency[edge[0], edge[1]] - 1
    return graph


def compute_is_endpoints(branches_by_nodes) -> npt.NDArray[np.bool_]:
    """
    Compute a boolean array indicating if each node is an endpoint (i.e. connected to only one branch).
    """
    return np.sum(branches_by_nodes, axis=0) == 1


def delete_nodes(branches_by_nodes, nodes_id):
    """
    Delete a node and its connected branch from the connectivity matrix of branches and nodes.
    """
    # Check parameters and compute the initial branch lookup
    branch_lookup, nodes_mask, nodes_id = _prepare_node_deletion(branches_by_nodes, nodes_id)

    deleted_branches = branches_by_nodes[:, nodes_id].any(axis=1)
    branches_id = np.where(deleted_branches)[0]

    branches_by_nodes = np.delete(branches_by_nodes, branches_id, axis=0)
    branches_by_nodes = np.delete(branches_by_nodes, nodes_id, axis=1)

    branch_lookup = add_empty_to_lookup(branch_lookup)
    branch_lookup[1:][deleted_branches] = 0
    branch_shift_lookup = np.cumsum(np.concatenate([[True], ~deleted_branches])) - 1

    return branches_by_nodes, branch_shift_lookup[branch_lookup], nodes_mask


def fuse_nodes(
    branches_by_nodes: npt.NDArray[np.bool_],
    nodes_id: npt.NDArray[np.int64],
    node_coord: Tuple[np.ndarray, np.ndarray] | None = None,
):
    """
    Fuse a node from the connectivity matrix of branches and nodes.
    The node is removed and the two branches connected to it are fused into one.

    Parameters
    ----------
    branches_by_nodes:
        A (N, M) boolean matrix where N is the number of branches and M the number of nodes.

    nodes_id:
        The index of the node to remove.

    node_coord:
        The coordinates of the nodes. This is not used by the function but remove the need to filter the nodes_coord
        manually.

    Returns
    -------
    branches_by_nodes:
        A (N', M') boolean matrix where N' is the number of branches after merging.

    branch_lookup:
        A (N,) array containing the new branch labels.

    nodes_mask:
        A (M,) boolean array indicating the nodes that have been removed.

    nodes_labels: (optional, if node_coord is not None)
        A tuple (y, x, branch) of three 1D arrays of length M' containing the coordinates of the nodes and the label of
        the branch they are connected to.

    """

    # Check parameters and compute the initial branch lookup
    branch_lookup, nodes_mask, nodes_id = _prepare_node_deletion(branches_by_nodes, nodes_id)
    nb_branches = len(branch_lookup)

    branches_by_fused_nodes = branches_by_nodes[:, nodes_id]
    invalid_fused_node = np.sum(branches_by_fused_nodes, axis=0) > 2
    if np.any(invalid_fused_node):
        warnings.warn("Warning: some nodes are connected to more than 2 branches and won't be fused.", stacklevel=2)
        branches_by_fused_nodes = branches_by_fused_nodes[:, ~invalid_fused_node]
    branch1_ids = np.argmax(branches_by_fused_nodes, axis=0)
    branch2_ids = nb_branches - np.argmax(branches_by_fused_nodes[::-1], axis=0) - 1

    sort_id = np.argsort(branch1_ids)[::-1]
    branch1_ids = branch1_ids[sort_id]
    branch2_ids = branch2_ids[sort_id]
    if node_coord is not None:
        node_coord = tuple(c[nodes_id[sort_id]] for c in node_coord)
    branches_to_delete = []

    # Sequential merge is required when a branch appear both in branch1_ids and branch2_ids
    #  (because 2 adjacent nodes are fused)
    for b1, b2 in zip(branch1_ids, branch2_ids, strict=True):
        b2_id = branch_lookup[b2]
        branches_by_nodes[b1] |= branches_by_nodes[b2_id]
        branches_to_delete.append(b2_id)

        branch_lookup = apply_lookup(branch_lookup, {b2_id: b1})

    branches_by_nodes = np.delete(branches_by_nodes, branches_to_delete, axis=0)
    branches_by_nodes = np.delete(branches_by_nodes, nodes_id, axis=1)

    branch_shift_lookup = np.cumsum(index_to_mask(branches_to_delete, len(branch_lookup), invert=True)) - 1
    branch_lookup = add_empty_to_lookup(branch_shift_lookup[branch_lookup])

    if node_coord is not None:
        nodes_labels = node_coord[0], node_coord[1], branch_lookup[branch1_ids + 1]
        return branches_by_nodes, branch_lookup, nodes_mask, nodes_labels
    else:
        return branches_by_nodes, branch_lookup, nodes_mask


def fuse_node_pairs(branches_by_nodes: npt.NDArray[np.bool_], pairs_node_id: npt.NDArray):
    """
    Fuse pairs of nodes, removing them from the connectivity matrix and merging their connected branches.
    Each nodes must be connected to a single branch.

    Parameters
    ----------
    branches_by_nodes : npt.NDArray[np.bool_]
        The connectivity matrix as a (B, N) boolean matrix where B is the number of branches and N the number of nodes.
    pairs_node_id : npt.NDArray
        A (P, 2) array containing P pairs of nodes to fuse.

    Returns
    -------
    branches_by_nodes : npt.NDArray[np.bool_]
        The new connectivity matrix of shape (B-P, N-2P)
    branch_lookup : npt.NDArray[np.int64], shape (B,)
        The branch lookup table to propagate the merging to the branches labels.
    merged_branche_id : npt.NDArray[np.int64], shape (P,)
        For each node pair, the new label of their branches after the fuse.
    """
    assert pairs_node_id.ndim == 2 and pairs_node_id.shape[1] == 2, "pairs_node_id must be a 2D array of shape (N, 2)"
    assert len(np.unique(pairs_node_id.flatten())) == pairs_node_id.size, "Each node must be unique in pairs_node_id"
    assert np.all(
        np.sum(branches_by_nodes[:, pairs_node_id.flatten()], axis=0) == 1
    ), "Each node in pairs_node_id must be connected to a single branch"
    nb_branches, N = branches_by_nodes.shape

    # --- Compute the branches pairs to merge ---
    pairs_branch_id = np.argmax(branches_by_nodes[:, pairs_node_id], axis=0)

    # --- Remove the nodes to fuse ---
    branches_by_nodes = np.delete(branches_by_nodes, pairs_node_id.flatten(), axis=1)

    # --- Merge the branches ---
    # Sort the pairs by the first node id
    pairs_branch_id.sort(axis=1)
    branch_target = pairs_branch_id[:, 0]
    branch_to_delete = pairs_branch_id[:, 1]

    # Merge the branches
    branches_by_nodes[branch_target] += branches_by_nodes[branch_to_delete]
    branches_by_nodes = np.delete(branches_by_nodes, branch_to_delete, axis=0)

    # Create the branch_lookup
    branch_lookup = np.arange(nb_branches, dtype=np.int64)
    branch_lookup[branch_to_delete] = branch_target  # Redirect
    branch_lookup_shift = np.cumsum(~index_to_mask(branch_to_delete, nb_branches)) - 1
    branch_lookup = branch_lookup_shift[branch_lookup]

    return branches_by_nodes, add_empty_to_lookup(branch_lookup), branch_lookup[branch_target]


def _prepare_node_deletion(branches_by_nodes, nodes_id):
    assert nodes_id.ndim == 1, "nodes_id must be a 1D array"
    if nodes_id.dtype == bool:
        assert len(nodes_id) == branches_by_nodes.shape[1], (
            "nodes_id must be a boolean array of the same length as the number of nodes,"
            f" got len(nodes_id)={len(nodes_id)} instead of {branches_by_nodes.shape[1]}."
        )
        nodes_mask = ~nodes_id
        nodes_id = np.where(nodes_id)[0]
    else:
        nodes_mask = None
    assert nodes_id.dtype == np.int64, "nodes_id must be a boolean or integer array"

    if nodes_mask is None:
        nodes_mask = index_to_mask(nodes_id, branches_by_nodes.shape[1], invert=True)

    nb_branches = branches_by_nodes.shape[0]
    return np.arange(nb_branches, dtype=np.int64), nodes_mask, nodes_id


def merge_equivalent_branches(
    branches_by_nodes: np.ndarray,
    max_nodes_distance: int | None = None,
    nodes_coordinates: Tuple[np.ndarray, np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge branches that are equivalent (same nodes) and return the resulting branch_by_node matrix and a lookup table for branch labels.

    Args:
        branches_by_node: A (N, M) boolean matrix where N is the number of branches and M the number of nodes.
        max_nodes_distance: If not None, only branches connecting two nodes seperated by a distance smaller than this value
                            will be merged.
        nodes_coordinates: A (2, M) array or tuples of array containing the coordinates of the nodes.
                           Must be provided if max_nodes_distance is not None.

    Returns:
        branches_by_node: A (N', M) boolean matrix where N' is the number of branches after merging.
        branch_lookup: A (N,) array containing the new branch labels.
    """
    if max_nodes_distance is None:
        branches_by_nodes, branches_lookup = np.unique(branches_by_nodes, return_inverse=True, axis=0)
    else:
        # Select small branches
        assert nodes_coordinates is not None, "nodes_coordinates must be provided when max_nodes_distance is not None"
        branches_n1, branches_n2 = branch_by_nodes_to_adjacency_list(branches_by_nodes).T
        nodes_coordinates = np.stack(nodes_coordinates, axis=1)
        branches_node_dist = np.linalg.norm(nodes_coordinates[branches_n1] - nodes_coordinates[branches_n2], axis=1)

        branches_to_remove = np.zeros(branches_by_nodes.shape[0], dtype=bool)
        branches_lookup = np.arange(branches_by_nodes.shape[0] + 1, dtype=np.int64)

        small_branches = branches_node_dist <= max_nodes_distance
        small_branches_lookup = np.where(small_branches)[0]
        if len(small_branches_lookup) == 0:
            return branches_by_nodes, None

        small_branches_by_node = branches_by_nodes[small_branches]

        # Identify equivalent small branches
        small_branches_by_node, unique_idx, unique_count = np.unique(
            small_branches_by_node, axis=0, return_inverse=True, return_counts=True
        )
        for duplicate_id in np.where(unique_count > 1)[0]:
            duplicated_branches = small_branches_lookup[np.where(unique_idx == duplicate_id)[0]]
            branches_to_remove[duplicated_branches[1:]] = True
            branches_lookup = apply_lookup(
                branches_lookup, (duplicated_branches[1:] + 1, branches_lookup[duplicated_branches[0] + 1])
            )

        if len(branches_to_remove) == 0:
            return branches_by_nodes, None

        # Delete duplicated branches
        branches_by_nodes = branches_by_nodes[~branches_to_remove, :]

        branches_lookup_shift = np.cumsum(np.concatenate(([True], ~branches_to_remove))) - 1
        branches_lookup = branches_lookup_shift[branches_lookup]

    return branches_by_nodes, branches_lookup


def merge_nodes_by_distance(
    branches_by_nodes: np.ndarray,
    nodes_coord: Tuple[np.ndarray, np.ndarray],
    distance: float | List[Tuple[np.ndarray, float, bool]],
):
    """ """
    if isinstance(distance, float):
        distance = [(None, distance, True)]

    branch_lookup = None

    for i, (mask, dist, remove_branch) in enumerate(distance):
        if dist <= 0:
            continue
        masked_coord = np.stack(nodes_coord, axis=1)
        masked_coord = masked_coord[mask] if mask is not None else masked_coord
        proximity_matrix = np.linalg.norm(masked_coord[:, None] - masked_coord[None, :], axis=2) <= dist
        proximity_matrix &= ~np.eye(proximity_matrix.shape[0], dtype=bool)

        if proximity_matrix.sum() == 0:
            continue

        lookup = np.arange(len(nodes_coord[0]), dtype=np.int64)
        lookup = lookup[mask] if mask is not None else lookup
        nodes_clusters = [
            tuple(lookup[_] for _ in cluster)
            for cluster in adjacency_list_connected_components(np.where(proximity_matrix))
            if len(cluster) > 1
        ]
        endpoints_nodes = compute_is_endpoints(branches_by_nodes)

        if remove_branch is None:
            termination_branches = np.sum(branches_by_nodes[:, endpoints_nodes].astype(bool), axis=1) == 1
            remove_branch = np.where(termination_branches)[0]
        branches_by_nodes, branch_lookup2, node_lookup = merge_nodes_clusters(
            branches_by_nodes, nodes_clusters, erase_branches=remove_branch
        )

        nodes_coord = apply_lookup_on_coordinates(nodes_coord, node_lookup, weight=~endpoints_nodes)
        branch_lookup = apply_lookup(branch_lookup, branch_lookup2)
        inverted_lookup = invert_lookup(node_lookup)

        for j, (m, d, r) in enumerate(distance[i + 1 :]):
            if m is not None:
                distance[j + i + 1] = (m[inverted_lookup], d, r)

    return branches_by_nodes, branch_lookup, nodes_coord


def merge_nodes_clusters(
    branches_by_nodes: np.ndarray, nodes_clusters: List[set], erase_branches: bool | List[int] = True
):
    """
    Merge nodes according to a set of clusters. Return the resulting branch_by_node matrix and lookup tables for branch and  node labels.

    Parameters:
    ----------

    branches_by_nodes:
        A boolean matrix of shape (N, M) where N is the number of branches and M the number of nodes.

    nodes_clusters:
        A list of sets of nodes to merge.

        .. warning::
            The implementation assume that each nodes is only present in one cluster.

    erase_branches:

        - If True, branches connecting two nodes of the same cluster will be deleted, otherwise they will be relabeled as one of the branches incoming to the cluster.
        - If list of int, the branches with the corresponding index will be erased, while the other will be relabeled.

    Returns:
    --------
        - branches_by_node: A (N', M') boolean matrix where N' is the number of branches after merging.
        - branch_lookup: A (N,) array mapping the new branch labels.
        - node_lookup: A (M,) array mapping the new node labels.
    """
    node_lookup = np.arange(branches_by_nodes.shape[1], dtype=np.int64)
    branches_to_remove = np.zeros(branches_by_nodes.shape[0], dtype=bool)
    branches_lookup = np.arange(branches_by_nodes.shape[0] + 1, dtype=np.int64)
    node_to_remove = np.zeros(branches_by_nodes.shape[1], dtype=bool)

    # 1. Process each cluster, removing and relabeling branches and nodes in the lookup tables
    for cluster in nodes_clusters:
        cluster = np.asarray(tuple(cluster), dtype=np.int64)
        cluster.sort()  # Sort the cluster to only keep the node with the lowest id

        # 1.a Find the branches connected linking several nodes of the cluster (included in the cluster)...
        cluster_branches = np.where(np.sum(branches_by_nodes[:, cluster].astype(bool), axis=1) >= 2)[0]
        # cluster_branches.sort() # USELESS?
        branches_to_remove[cluster_branches] = True  # ... and mark them for deletion

        # 1.b Relabel the branches as backrgound

        # 1.c If erase_branches is not False, attempt to relabel the branches as one of the incoming branches
        if erase_branches is not True:
            # Find the branches connected to only one node of the cluster
            incoming_cluster_branches = np.where(np.sum(branches_by_nodes[:, cluster].astype(bool), axis=1) == 1)[0]
            incoming_label = branches_lookup[incoming_cluster_branches[0] + 1]
            if erase_branches is False or len(incoming_cluster_branches) == 0:
                branches_new_labels = [incoming_label] * len(cluster_branches)
            else:
                branches_new_labels = [0 if b in erase_branches else incoming_label for b in cluster_branches]
            branches_new_labels = np.array(branches_new_labels)
            # If any, relabel the branches as one of the incoming branches
            branches_lookup = apply_lookup(branches_lookup, (cluster_branches + 1, branches_new_labels))
        else:
            branches_lookup[cluster_branches + 1] = 0

        # 1.c Merge the node of the cluster to the lowest node id...
        node_lookup[cluster[1:]] = cluster[0]
        node_to_remove[cluster[1:]] = True  # ... and mark them for deletion.

    # 2. Remove the marked branches
    if branches_to_remove.any():
        # 2.a Simplify the branches_by_nodes matrix by removing the rows associated with the marked branches
        branches_by_nodes = branches_by_nodes[~branches_to_remove, :]
        # if erase_branches:
        #     branches_lookup[np.where(branches_to_remove)[0] + 1] = 0
        # 2.b Map the branches ids so they increase monotonously, skipping the gaps created by branches removals
        branches_lookup_shift = np.cumsum(np.concatenate(([True], ~branches_to_remove))) - 1
        branches_lookup = branches_lookup_shift[branches_lookup]
    else:
        branches_lookup = None
    nb_branches = branches_by_nodes.shape[0]

    # 3. Merge the marked nodes
    #   3.a Map the nodes ids so they increase monotonously, skipping the gaps created by nodes removals
    node_lookup_shift = np.cumsum(~node_to_remove) - 1
    nb_nodes = node_lookup_shift[-1] + 1
    node_lookup = node_lookup_shift[node_lookup]

    #   3.b Merge the appropriate columns of the branches_by_nodes matrix by summation
    nodes_by_branches = np.zeros_like(branches_by_nodes, shape=(nb_nodes, nb_branches))
    np.add.at(nodes_by_branches, node_lookup, branches_by_nodes.T)

    return nodes_by_branches.T, branches_lookup, node_lookup


def node_rank(branches_by_nodes):
    """
    Compute the rank of each node in the connectivity matrix of branches and nodes.
    """
    return np.sum(branches_by_nodes, axis=0)


def adjacency_list_connected_components(
    pairwise_connection: List[Tuple[int, int]] | Tuple[np.ndarray, np.ndarray],
) -> List[set]:
    """
    Generate a list of clusters from a list of pairwise connections.
    """

    if isinstance(pairwise_connection, tuple):
        assert len(pairwise_connection) == 2, "pairwise_connection must be a tuple of two arrays"
        assert (
            pairwise_connection[0].shape == pairwise_connection[1].shape
        ), "pairwise_connection must be a tuple of two arrays of the same shape"
        pairwise_connection = zip(*pairwise_connection, strict=True)

    # TODO: check and use the cython implementation: graph_utilities_cy.solve_clusters

    clusters = []
    for p1, p2 in pairwise_connection:
        for i, c in enumerate(clusters):
            if p1 in c:
                for j, c2 in enumerate(clusters[i + 1 :]):
                    if p2 in c2:
                        c.update(c2)
                        del clusters[j]
                        break
                else:
                    c.add(p2)
                break
            elif p2 in c:
                for j, c1 in enumerate(clusters[i + 1 :]):
                    if p1 in c1:
                        c.update(c1)
                        del clusters[j]
                        break
                else:
                    c.add(p1)
                break
        else:
            clusters.append({p1, p2})
    return clusters


def reduce_clusters(clusters: List[Tuple[int, ...]]) -> List[set]:
    """
    Merge clusters that share at least one element.
    """
    clusters = [set(c) for c in clusters]
    for i, c in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if i == j:
                continue
            if c.intersection(c2):
                c.update(c2)
                del clusters[j]
                break
    return clusters


def perimeter_from_vertices(coord: np.ndarray, close_loop: bool = True) -> float:
    """
    Compute the perimeter of a polygon defined by a list of vertices.

    Args:
        coord: A (V, 2) array or list of V vertices.
        close_loop: If True, the polygon is closed by adding an edge between the first and the last vertex.

    Returns:
        The perimeter of the polygon. (The sum of the distance between each vertex and the next one.)
    """
    coord = np.asarray(coord)
    next_coord = np.roll(coord, 1, axis=0)
    if not close_loop:
        next_coord = next_coord[:-1]
        coord = coord[:-1]
    return np.sum(np.linalg.norm(coord - next_coord, axis=1))


def nodes_tangent(
    nodes_coord: np.ndarray,
    branches_label_map: np.ndarray,
    branches_id: Iterable[int] = None,
    *,
    gaussian_offset: float = 7,
    gaussian_std: float = 7,
):
    """
    Compute the vector tangent to the skeleton at given nodes. The vector is directed as a continuation of te skeleton.

    The computed vector is the symmetric of the barycenter of the surrounding skeleton points weighted by a gaussian
    distribution on the distance to the considered node. Note that the center of the gaussian distribution is offset
    from the considered node to account for skeleton artefact near the node.


    Parameters
    ----------
    coord : np.ndarray, shape (N, 2)
        The coordinates of the nodes to consider.
    skeleton_map : np.ndarray, shape (H, W)
        The skeleton map.
    branch_id : Iterable[int], shape (N,)
        For each node, the label of the branch it is connected to.
        If 0, all the branches arround the node will be considered. (This may lead to indesirable results.)

        .. warning::

            The branch_id must follow the same indexing as the skeleton_map: especially the indexes must start at 1.

    gaussian_offset : float, optional
        The offset in pixel of the gaussian weighting the skeleton arround the node. Must be positive.
        By default: 7.
    std : float, optional
        The standard deviation in pixel of the gaussian weighting the skeleton arround the node. Must be positive.
        By default 7.

    Returns
    -------
    np.ndarray, shape (N, 2)
        The tangent vectors at the given nodes. The vectors are normalized to unit length.
    """
    assert len(nodes_coord) == len(branches_id), "coord and branch_id must have the same length"
    assert nodes_coord.ndim == 2 and nodes_coord.shape[1] == 2, "coord must be a 2D array of shape (N, 2)"
    N = len(nodes_coord)
    assert branches_label_map.ndim == 2, "skeleton_map must be a 2D array"
    assert branches_label_map.shape[0] > 0 and branches_label_map.shape[1] > 0, "skeleton_map must be non empty"
    assert gaussian_offset > 0, "gaussian_offset must be positive"
    assert gaussian_std > 0, "gaussian_std must be positive"

    tangent_vectors = np.zeros((N, 2), dtype=np.float64)
    for i, ((y, x), branch_id) in enumerate(zip(nodes_coord, branches_id, strict=True)):
        pos = Point(y, x)
        window_rect = Rect.from_center(pos, 2 * (2 * gaussian_std + gaussian_offset)).clip(branches_label_map.shape)
        window = branches_label_map[window_rect.slice()]
        pos = pos - window_rect.top_left

        skel_points = np.where(window == branch_id if branch_id != 0 else window > 0)
        skel_points = np.array(skel_points, dtype=np.float64).T
        skel_points_d = pos.distance(skel_points)
        skel_points_w = np.exp(-0.5 * (skel_points_d - gaussian_offset) ** 2 / gaussian_std**2)
        barycenter = np.sum(skel_points * skel_points_w[:, None], axis=0) / np.sum(skel_points_w) - pos
        tangent_vectors[i] = -barycenter / np.linalg.norm(barycenter)

    return tangent_vectors
