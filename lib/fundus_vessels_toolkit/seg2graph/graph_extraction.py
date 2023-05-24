import warnings
from typing import Mapping, Literal, TypedDict, TypeAlias

import numpy as np
import networkx as nx
import scipy.ndimage as scimage
from skimage.measure import label
from skimage.segmentation import expand_labels

from .skeleton_utilities import extract_unravelled_pattern
from .graph_utilities import apply_lookup, apply_node_lookup_on_coordinates, branch_by_nodes_to_adjacency_list, \
    compute_is_endpoints, delete_nodes, fuse_nodes, index_to_mask, merge_nodes_by_distance, \
    merge_nodes_clusters, merge_equivalent_branches, node_rank, perimeter_from_vertices, solve_clusters


class NodeMergeDistanceDict(TypedDict):
    junction: float
    termination: float
    node: float


NodeMergeDistanceParam: TypeAlias = bool | float | NodeMergeDistanceDict
SimplifyTopology: TypeAlias = Literal['node', 'branch', 'both'] | None


def seg_to_adjacency_branches_nodes(vessel_map: np.ndarray, return_label=False,
                                    max_spurs_distance: float = 0,
                                    nodes_merge_distance: NodeMergeDistanceParam = True,
                                    merge_small_cycles: float = 0,
                                    simplify_topology: SimplifyTopology = 'node'):
    """
    Extract the naive vasculature graph from a vessel map.
    If return label is True, the label map of branches and nodes are also computed and returned.

    Small topological corrections are applied to the graph:
        - nodes too close to each other are merged (max distance=5√2/2 by default)
        - cycles with small perimeter are merged (max size=15% of the image width by default)
        - if simplify_topology is True, the graph is simplified by merging equivalent branches (branches connecting the same junctions)
            and removing nodes with degree 2.

    Args:
        vessel_map: The vessel map. Shape: (H, W) where H and W are the map height and width.
        return_label: If True, return the label map of branches and nodes.
        max_spurs_distance: If larger than 0, spurs (terminal branches) with a length smaller than this value are removed (disabled by default).
        nodes_merge_distance: If larger than 0, nodes separated by less than this distance are merged (5√2/2 by default).
        merge_small_cycles: If larger than 0, cycles with a perimeter smaller than this value are merged (disabled by default).
        simplify_topology: If 'node', the graph is simplified by fusing nodes with degree 2 (connected to exactly 2 branches).
            If 'branch', the graph is simplified by merging equivalent branches (branches connecting the same junctions).
            If 'both', both simplifications are applied.

    Returns:
        The adjacency matrix of the graph.
        Shape: (nBranch, nNode) where nBranch and nNode are the number of branches and nodes (junctions or terminations).

        If return_label is True, also return the label map of branches
            (where each branch of the skeleton is labeled by a unique integer corresponding to its index in the adjacency matrix)
        and the coordinates of the nodes as a tuple of (y, x) where y and x are vectors of length nNode.
    """
    if vessel_map.dtype != np.int8:
        from skeletonization import skeletonize
        skel = skeletonize(vessel_map, fix_hollow=True, max_spurs_length=10, return_distance=False)
    else:
        skel = vessel_map

    branches_by_nodes, labeled_branches, (node_y, node_x), nb_junctions = compute_branches_by_nodes_matrix(skel)
    is_endpoint = np.arange(len(node_y)) >= nb_junctions

    branch_lookup = None
    node_labels = (np.asarray([], dtype=np.int64),) * 3

    if max_spurs_distance > 0:
        # Remove spurs (terminal branches) shorter than max_spurs_distance
        # - Identify terminal branches (branches connected to endpoints)
        spurs_branches = np.any(branches_by_nodes[:, is_endpoint], axis=1)
        # - Extract the two nodes (the first is probably a junction, the second is necessarily an endpoint) connected
        #    by each terminal branch
        spurs_junction, spurs_endpoint = branch_by_nodes_to_adjacency_list(branches_by_nodes[spurs_branches]).T
        # - Discard branches whose first node is not a junction (branches connecting 2 endpoints)
        single_branches = spurs_junction >= nb_junctions
        spurs_junction = spurs_junction[~single_branches]
        spurs_endpoint = spurs_endpoint[~single_branches]
        # - Compute the distance between those nodes for each terminal branch
        spurs_distance = np.linalg.norm((node_y[spurs_junction] - node_y[spurs_endpoint],
                                         node_x[spurs_junction] - node_x[spurs_endpoint]), axis=0)
        # - Select every endpoint connected to a terminal branch with a distance smaller than max_spurs_distance
        node_to_delete = np.unique(np.concatenate((spurs_junction[spurs_distance < max_spurs_distance],
                                                   spurs_endpoint[spurs_distance < max_spurs_distance])))
        node_to_fuse = node_to_delete[node_to_delete < nb_junctions]
        node_to_delete = node_to_delete[node_to_delete >= nb_junctions]
        # - Delete those nodes
        branches_by_nodes, branch_lookup2, nodes_mask = delete_nodes(branches_by_nodes, node_to_delete)
        # - Apply the lookup tables on nodes and branches
        branch_lookup = apply_lookup(branch_lookup, branch_lookup2)
        node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]
        node_to_fuse = (np.cumsum(nodes_mask) - 1)[node_to_fuse]

        # - Remove useless nodes
        node_to_fuse = node_to_fuse[node_rank(branches_by_nodes[:, node_to_fuse]) == 2]
        if np.any(node_to_fuse):
            branches_by_nodes, branch_lookup2, nodes_mask, node_labels2 = \
                fuse_nodes(branches_by_nodes, node_to_fuse, (node_y, node_x))

            branch_lookup = apply_lookup(branch_lookup, branch_lookup2)
            node_labels = tuple(np.concatenate((n, n1)) for n, n1 in zip(node_labels, node_labels2))
            node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]

        is_endpoint = np.arange(len(node_y)) >= nb_junctions

    if nodes_merge_distance is True:
        nodes_merge_distance = 2.5 * np.sqrt(2)
    if isinstance(nodes_merge_distance, Mapping):
        junctions_merge_distance = nodes_merge_distance.get('junction', 0)
        terminations_merge_distance = nodes_merge_distance.get('termination', 0)
        nodes_merge_distance = nodes_merge_distance.get('node', 0)
    else:
        junctions_merge_distance = 0
        terminations_merge_distance = 0

    if nodes_merge_distance > 0 or junctions_merge_distance > 0 or terminations_merge_distance > 0:
        # Merge nodes clusters smaller than nodes_merge_distance
        distances = [(~is_endpoint, junctions_merge_distance, True),      # distance only for junctions
                     (is_endpoint, terminations_merge_distance, False),  # distance only for terminations
                     (None, nodes_merge_distance, False)]                  # distance for all nodes
        branches_by_nodes, branch_lookup2, nodes_coord = merge_nodes_by_distance(branches_by_nodes, (node_y, node_x),
                                                                                 distances)
        # - Apply the lookup tables on nodes and branches
        branch_lookup = apply_lookup(branch_lookup, branch_lookup2, node_labels[2])
        node_y, node_x = nodes_coord
        is_endpoint = compute_is_endpoints(branches_by_nodes)

    if merge_small_cycles > 0:
        # Merge small cycles
        # - Identify cycles
        nodes_adjacency_matrix = branches_by_nodes.T @ branches_by_nodes - np.eye(len(node_y))
        cycles = [_ for _ in nx.chordless_cycles(nx.from_numpy_array(nodes_adjacency_matrix), length_bound=4)
                  if len(_) > 2]
        # - Select chord less cycles with small perimeter
        cycles_perimeters = [perimeter_from_vertices(np.asarray((node_y, node_x)).T[cycle]) for cycle in cycles]
        cycles = [cycle for cycle, perimeter in zip(cycles, cycles_perimeters)
                  if perimeter < merge_small_cycles]
        # - Merge cycles
        branches_by_nodes, branch_lookup_2, nodes_mask = merge_nodes_clusters(branches_by_nodes, cycles,
                                                                              erase_branches=True)
        # - Apply the lookup tables on nodes and branches
        node_y, node_x = apply_node_lookup_on_coordinates((node_y, node_x), nodes_mask)
        branch_lookup = apply_lookup(branch_lookup, branch_lookup_2, node_labels[2])

    if merge_small_cycles > 0 or simplify_topology in ('branch', 'both'):
        # Merge 2 branches cycles (branches that are connected to the same 2 nodes)
        # - If simplify_topology is neither 'branch' nor 'both', limit the merge distance to half merge_small_cycles
        max_nodes_distance = merge_small_cycles // 2 if simplify_topology not in ('branch', 'both') else None
        # - Merge equivalent branches
        branches_by_nodes, branch_lookup_2 = merge_equivalent_branches(
            branches_by_nodes, max_nodes_distance=max_nodes_distance, nodes_coordinates=(node_y, node_x))
        # - Apply the lookup tables on branches
        branch_lookup = apply_lookup(branch_lookup, branch_lookup_2, node_labels[2])

    if simplify_topology in ('node', 'both'):
        # Fuse nodes that are connected to only 2 branches
        # - Identify nodes connected to only 2 branches
        nodes_to_fuse = node_rank(branches_by_nodes) == 2
        if np.any(nodes_to_fuse):
            # - Fuse nodes
            branches_by_nodes, branch_lookup2, nodes_mask, node_labels2 = \
                fuse_nodes(branches_by_nodes, nodes_to_fuse, (node_y, node_x))
            # - Apply the lookup tables on nodes and branches
            branch_lookup = apply_lookup(branch_lookup, branch_lookup2, node_labels[2])
            node_labels = tuple(np.concatenate((n, n1)) for n, n1 in zip(node_labels, node_labels2))
            node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]

        nodes_to_delete = node_rank(branches_by_nodes) == 0
        if np.any(nodes_to_delete):
            # - Delete nodes
            branches_by_nodes, branch_lookup2, nodes_mask = delete_nodes(branches_by_nodes, nodes_to_delete)
            # - Apply the lookup tables on nodes and branches
            branch_lookup = apply_lookup(branch_lookup, branch_lookup2, node_labels[2])
            node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]

    if return_label:
        if branch_lookup is not None:
            # Update branch labels
            labeled_branches = branch_lookup[labeled_branches]
        labeled_branches[node_labels[0].astype(np.int64), node_labels[1].astype(np.int64)] = node_labels[2]
        return branches_by_nodes, labeled_branches, (node_y, node_x)
    else:
        return branches_by_nodes


def compute_branches_by_nodes_matrix(skel: np.ndarray):
    bin_skel = skel > 0
    junctions_map = skel >= 3
    end_points_map = skel == 1
    nodes_map = junctions_map | end_points_map
    sqr3 = np.ones((3, 3), dtype=bool)

    # Label branches
    skel_no_nodes = bin_skel & ~scimage.binary_dilation(nodes_map, sqr3)
    labeled_branches, nb_branches = label(skel_no_nodes, return_num=True)
    labeled_branches = expand_labels(labeled_branches, 2) * (bin_skel & ~nodes_map)

    # Label nodes
    jy, jx = np.where(junctions_map)
    ey, ex = np.where(end_points_map)
    node_y = np.concatenate((jy, ey))
    node_x = np.concatenate((jx, ex))
    nb_junctions = len(jy)

    labels_nodes = np.zeros_like(labeled_branches)
    labels_nodes[node_y, node_x] = np.arange(1, len(node_y) + 1)

    # Identify branches connected to each junction
    ring_pattern = np.asarray([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=bool)
    nodes_ring = extract_unravelled_pattern(labeled_branches - labels_nodes,
                                            (node_y, node_x), ring_pattern, return_coordinates=False)
    branch_neighbors = np.maximum(nodes_ring, 0)
    nodes_neighbors = np.maximum(-nodes_ring, 0)

    # Build the matrix of connections from branches to nodes
    nb_nodes = len(node_y)
    branches_by_nodes = np.zeros((nb_branches + 1, nb_nodes), dtype=bool)
    branches_by_nodes[branch_neighbors.flatten(), np.repeat(np.arange(nb_nodes), ring_pattern.sum())] = 1
    branches_by_nodes = branches_by_nodes[1:, :]

    # Merge adjacent nodes to prevent invalid branches (branches connecting more than 2 nodes)
    pair_adjacent_nodes = np.where(nodes_neighbors)
    pair_adjacent_nodes = np.stack((pair_adjacent_nodes[0],
                                    nodes_neighbors[pair_adjacent_nodes[0], pair_adjacent_nodes[1]] - 1), axis=1)
    pair_adjacent_nodes.sort(axis=1)
    pair_adjacent_nodes = np.unique(pair_adjacent_nodes, axis=0)

    if len(pair_adjacent_nodes):
        nb_nodes = len(node_y)
        node_lookup = np.arange(nb_nodes)
        node_mask = np.ones(nb_nodes, dtype=bool)
        for cluster in solve_clusters(pair_adjacent_nodes):
            if len(cluster) > 1:
                # Convert the cluster index tso the new nodes index (to account for the previously merged nodes).
                cluster = np.asarray(list(cluster), dtype=np.int64)

                # Redirect branches to the first node of the cluster
                branches_by_nodes[:, cluster[0]] = np.any(branches_by_nodes[:, cluster], axis=1)

                # Mark the nodes to be merged into the first node of the cluster
                node_mask[cluster[1:]] = False
                node_lookup[cluster[1:]] = cluster[0]

        # Compute the node index shift due to nodes deletion
        node_shift = np.cumsum(node_mask) - 1
        node_lookup = node_shift[node_lookup]

        # Remove the merged nodess
        branches_by_nodes = branches_by_nodes[:, node_mask]
        node_y, node_x = apply_node_lookup_on_coordinates((node_y, node_x), node_lookup)

    invalid_branches = np.sum(branches_by_nodes, axis=1) != 2
    if np.any(invalid_branches):
        warnings.warn(f'{np.sum(invalid_branches)} branches are invalid (connecting more or less than 2 nodes).\n'
                      f'Those branches will be removed the graph. But this will probably cause invalid topology.\n'
                      f'You should report this issue to the developer.')
        branch_lookup = np.concatenate(([0], np.cumsum(~invalid_branches)))
        labeled_branches = branch_lookup[labeled_branches]
        branches_by_nodes = branches_by_nodes[~invalid_branches, :]

    return branches_by_nodes, labeled_branches, (node_y, node_x), nb_junctions


def branches_by_nodes_to_node_graph(branches_by_nodes, node_pos=None):
    branches = np.arange(branches_by_nodes.shape[0]) + 1
    branches_by_nodes = branches_by_nodes.astype(bool)
    node_adjacency = branches_by_nodes.T @ (branches_by_nodes * branches[:, None])
    graph = nx.from_numpy_array((node_adjacency > 0) & (~np.eye(branches_by_nodes.shape[1], dtype=bool)))
    if node_pos is not None:
        node_y, node_x = node_pos
        nx.set_node_attributes(graph, node_y, 'y')
        nx.set_node_attributes(graph, node_x, 'x')
    for edge in graph.edges():
        graph.edges[edge]['branch'] = node_adjacency[edge[0], edge[1]] - 1
    return graph
