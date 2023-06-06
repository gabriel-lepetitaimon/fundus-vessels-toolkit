import warnings
import numpy as np
from typing import Dict, List, Tuple


def add_empty_to_lookup(lookup: np.ndarray) -> np.ndarray:
    """
    Add an empty entry to a lookup table.
    """
    return np.concatenate([[0], lookup+1], dtype=lookup.dtype)


def apply_lookup(array: np.ndarray | None, mapping: Dict[int, int] | Tuple[np.ndarray, np.ndarray] | np.array | None,
                 apply_inplace_on: np.ndarray | None = None) \
        -> np.ndarray:
    lookup = mapping
    if mapping is None:
        return array
    if not isinstance(mapping, np.ndarray):
        lookup = np.arange(len(array), dtype=np.int64)
        if isinstance(mapping, dict):
            mapping = tuple(zip(*mapping.items(), strict=True))
        search = mapping[0]
        replace = mapping[1]
        if not isinstance(replace, np.ndarray):
            replace = [replace] * len(search)
        for s, r in zip(search, replace, strict=False):
            lookup[lookup == s] = r

    if apply_inplace_on is not None:
        apply_inplace_on[:] = lookup[apply_inplace_on]

    return lookup if array is None else lookup[array]


def apply_node_lookup_on_coordinates(nodes_coord, nodes_lookup: np.ndarray | None,
                                     nodes_weight: np.ndarray | None = None):
    """
    Apply a lookup table on a set of coordinates to merge specific nodes and return their barycenter.

    Args:
        nodes_coord: A tuple (y, x) of two 1D arrays of the same length N containing the nodes coordinates.
        nodes_lookup: A 1D array of length N containing a lookup table of points index. Nodes with the same
                          index in this table will be merged.

    Returns:
        A tuple (y, x) of two 1D arrays of length M (max(junction_lookup) + 1) containing the merged coordinates.

    """
    if nodes_lookup is None:
        return nodes_coord
    if nodes_weight is None or nodes_weight.sum() == 0:
        nodes_weight = np.ones_like(nodes_lookup)
    nodes_weight = nodes_weight + 1e-8

    assert len(nodes_coord) == 2, f"nodes_coord must be a tuple of two 1D arrays. Got {len(nodes_coord)} elements."
    assert len(nodes_coord[0]) == len(nodes_coord[1]) == len(nodes_lookup) == len(nodes_weight), \
        f"nodes_coord, nodes_lookup and nodes_weight must have the same length. Got {len(nodes_coord[0])}, " \
        f"{len(nodes_lookup)} and {len(nodes_weight)} respectively."

    jy, jx = nodes_coord
    nodes_coord = np.zeros((np.max(nodes_lookup) + 1, 2), dtype=np.float64)
    nodes_total_weight = np.zeros((np.max(nodes_lookup) + 1,), dtype=np.float64)
    np.add.at(nodes_coord, nodes_lookup, np.asarray((jy, jx)).T * nodes_weight[:, None])
    np.add.at(nodes_total_weight, nodes_lookup, nodes_weight)
    nodes_coord = nodes_coord / nodes_total_weight[:, None]
    return nodes_coord.T


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


def compute_is_endpoints(branches_by_nodes):
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


def distance_matrix(nodes_coord: np.ndarray):
    """
    Compute the distance matrix between a set of points.

    Parameters
    ----------
    nodes_coord: A 2D array of shape (nb_nodes, 2) containing the coordinates of the nodes.

    Returns
    -------
    A 2D array of shape (nb_nodes, nb_nodes) containing the distance between each pair of nodes.
    """
    return np.linalg.norm(nodes_coord[:, None, :] - nodes_coord[None, :, :], axis=2)


def fuse_nodes(branches_by_nodes, nodes_id, node_coord: Tuple[np.ndarray, np.ndarray] | None = None):
    """
    Fuse a node from the connectivity matrix of branches and nodes.
    The node is removed and the two branches connected to it are fused into one.
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

    branch_shift_lookup = np.cumsum(index_to_mask(branches_to_delete, len(branch_lookup), invert=True))-1
    branch_lookup = add_empty_to_lookup(branch_shift_lookup[branch_lookup])

    if node_coord is not None:
        nodes_labels = node_coord[0], node_coord[1], branch_lookup[branch1_ids+1]
        return branches_by_nodes, branch_lookup, nodes_mask, nodes_labels
    else:
        return branches_by_nodes, branch_lookup, nodes_mask


def _prepare_node_deletion(branches_by_nodes, nodes_id):
    assert nodes_id.ndim == 1, "nodes_id must be a 1D array"
    if nodes_id.dtype == bool:
        assert len(nodes_id) == branches_by_nodes.shape[
            1], "nodes_id must be a boolean array of the same length as the number of nodes," \
                f" got len(nodes_id)={len(nodes_id)} instead of {branches_by_nodes.shape[1]}."
        nodes_mask = ~nodes_id
        nodes_id = np.where(nodes_id)[0]
    else:
        nodes_mask = None
    assert nodes_id.dtype == np.int64, "nodes_id must be a boolean or integer array"

    if nodes_mask is None:
        nodes_mask = index_to_mask(nodes_id, branches_by_nodes.shape[1], invert=True)

    nb_branches = branches_by_nodes.shape[0]
    return np.arange(nb_branches, dtype=np.int64), nodes_mask, nodes_id


def index_to_mask(index, length, invert=False):
    """
    Convert a list of indices to a boolean mask.
    """
    if not isinstance(length, int):
        length = len(length)
    mask = np.zeros(length, dtype=bool) if not invert else np.ones(length, dtype=bool)
    mask[index] = not invert
    return mask


def invert_lookup(lookup):
    """
    Invert a lookup array. The lookup array must be a 1D array of integers. The output array is a 1D array of length
    max(lookup) + 1. The output array[i] contains the list of indices of lookup where lookup[index] == i.
    """
    splits = np.split(np.argsort(np.argsort(lookup)), np.cumsum(np.unique(lookup, return_counts=True)[1]))[:-1]
    return np.array([np.array(s[0], dtype=np.int64) for s in splits])


def merge_equivalent_branches(branches_by_nodes: np.ndarray, 
                              max_nodes_distance: int | None = None, 
                              nodes_coordinates: Tuple[np.ndarray, np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
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
        branches_lookup = np.arange(branches_by_nodes.shape[0]+1, dtype=np.int64)

        small_branches = branches_node_dist <= max_nodes_distance
        small_branches_lookup = np.where(small_branches)[0]
        if len(small_branches_lookup) == 0:
            return branches_by_nodes, None

        small_branches_by_node = branches_by_nodes[small_branches]

        # Identify equivalent small branches
        small_branches_by_node, unique_idx, unique_count = np.unique(small_branches_by_node, axis=0,
                                                                     return_inverse=True, return_counts=True)
        for duplicate_id in np.where(unique_count > 1)[0]:
            duplicated_branches = small_branches_lookup[np.where(unique_idx == duplicate_id)[0]]
            branches_to_remove[duplicated_branches[1:]] = True
            branches_lookup = apply_lookup(branches_lookup,
                                           (duplicated_branches[1:] + 1, branches_lookup[duplicated_branches[0] + 1]))

        if len(branches_to_remove) == 0:
            return branches_by_nodes, None

        # Delete duplicated branches
        branches_by_nodes = branches_by_nodes[~branches_to_remove, :]

        branches_lookup_shift = np.cumsum(np.concatenate(([True], ~branches_to_remove))) - 1
        branches_lookup = branches_lookup_shift[branches_lookup]

    return branches_by_nodes, branches_lookup


def merge_nodes_by_distance(branches_by_nodes: np.ndarray, nodes_coord: Tuple[np.ndarray, np.ndarray],
                            distance: float | List[Tuple[np.ndarray, float, bool]]):
    """

    """
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
        nodes_clusters = [tuple(lookup[_] for _ in cluster)
                          for cluster in solve_clusters(np.where(proximity_matrix)) if len(cluster) > 1]
        endpoints_nodes = compute_is_endpoints(branches_by_nodes)

        branches_by_nodes, branch_lookup2, node_lookup = merge_nodes_clusters(branches_by_nodes, nodes_clusters,
                                                                              erase_branches=remove_branch)

        nodes_coord = apply_node_lookup_on_coordinates(nodes_coord, node_lookup, nodes_weight=endpoints_nodes)
        branch_lookup = apply_lookup(branch_lookup, branch_lookup2)
        inverted_lookup = invert_lookup(node_lookup)

        for j, (m, d, r) in enumerate(distance[i+1:]):
            if m is not None:
                distance[j+i+1] = (m[inverted_lookup], d, r)

    return branches_by_nodes, branch_lookup, nodes_coord


def merge_nodes_clusters(branches_by_nodes: np.ndarray, nodes_clusters: List[set], erase_branches=True):
    """
    Merge nodes according to a set of clusters. Return the resulting branch_by_node matrix and lookup tables for branch and  node labels. 

    Args:
        branches_by_nodes: A (N, M) boolean matrix where N is the number of branches and M the number of nodes.
        nodes_clusters: A list of sets of nodes to merge. 
                        The implementation assume that each nodes is only present in one cluster.
        erase_branches: If True, branches connecting two nodes of the same cluster will be deleted, otherwise they
                        will be relabeled as one of the branches incoming to the cluster.

    Returns:
        branches_by_node: A (N', M') boolean matrix where N' is the number of branches after merging.
        branch_lookup: A (N,) array mapping the new branch labels.
        node_lookup: A (M,) array mapping the new node labels.
    """
    node_lookup = np.arange(branches_by_nodes.shape[1], dtype=np.int64)
    branches_to_remove = np.zeros(branches_by_nodes.shape[0], dtype=bool)
    branches_lookup = np.arange(branches_by_nodes.shape[0]+1, dtype=np.int64)
    node_to_remove = np.zeros(branches_by_nodes.shape[1], dtype=bool)

    for cluster in nodes_clusters:
        cluster = np.asarray(tuple(cluster), dtype=np.int64)
        cluster.sort()
        cluster_branches = np.where(np.sum(branches_by_nodes[:, cluster].astype(bool), axis=1) >= 2)[0]
        cluster_branches.sort()
        branches_to_remove[cluster_branches] = True
        incoming_cluster_branches = np.where(np.sum(branches_by_nodes[:, cluster].astype(bool), axis=1) == 1)[0]

        if len(incoming_cluster_branches):
            branches_lookup = apply_lookup(branches_lookup,
                                           (cluster_branches + 1, branches_lookup[incoming_cluster_branches[0] + 1]))
        else:
            branches_lookup[cluster_branches+1] = 0
        node_lookup[cluster[1:]] = cluster[0]
        node_to_remove[cluster[1:]] = True

    if branches_to_remove.any():
        branches_by_nodes = branches_by_nodes[~branches_to_remove, :]
        if erase_branches:
            branches_lookup[np.where(branches_to_remove)[0]+1] = 0
        branches_lookup_shift = np.cumsum(np.concatenate(([True], ~branches_to_remove))) - 1
        branches_lookup = branches_lookup_shift[branches_lookup]
    else:
        branches_lookup = None
    nb_branches = branches_by_nodes.shape[0]

    node_lookup_shift = np.cumsum(~node_to_remove) - 1
    nb_nodes = node_lookup_shift[-1] + 1
    node_lookup = node_lookup_shift[node_lookup]

    nodes_by_branches = np.zeros_like(branches_by_nodes, shape=(nb_nodes, nb_branches))
    np.add.at(nodes_by_branches, node_lookup, branches_by_nodes.T)

    return nodes_by_branches.T, branches_lookup, node_lookup


def node_rank(branches_by_nodes):
    """
    Compute the rank of each node in the connectivity matrix of branches and nodes.
    """
    return np.sum(branches_by_nodes, axis=0)


def solve_clusters(pairwise_connection: List[Tuple[int, int]] | Tuple[np.ndarray, np.ndarray]) -> List[set]:
    """
    Generate a list of clusters from a list of pairwise connections.
    """

    if isinstance(pairwise_connection, tuple):
        assert len(pairwise_connection) == 2, "pairwise_connection must be a tuple of two arrays"
        assert pairwise_connection[0].shape == pairwise_connection[1].shape, "pairwise_connection must be a tuple of two arrays of the same shape"
        pairwise_connection = zip(*pairwise_connection, strict=True)

    # TODO: check and use the cython implementation: graph_utilities_cy.solve_clusters

    clusters = []
    for p1, p2 in pairwise_connection:
        for i, c in enumerate(clusters):
            if p1 in c:
                for j, c2 in enumerate(clusters[i+1:]):
                    if p2 in c2:
                        c.update(c2)
                        del clusters[j]
                        break
                else:
                    c.add(p2)
                break
            elif p2 in c:
                for j, c1 in enumerate(clusters[i+1:]):
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
