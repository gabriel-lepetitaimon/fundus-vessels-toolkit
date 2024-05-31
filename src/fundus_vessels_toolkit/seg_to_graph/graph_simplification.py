import warnings
from typing import Literal, Mapping, Optional, TypeAlias, TypedDict

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..utils.geometric import distance_matrix
from ..utils.graph.branch_by_nodes import (
    branch_by_nodes_to_adjacency_list,
    compute_is_endpoints,
    delete_nodes,
    fuse_nodes,
    merge_equivalent_branches,
    merge_nodes_by_distance,
    merge_nodes_clusters,
    node_rank,
    reduce_clusters,
)
from ..utils.lookup_array import apply_lookup, apply_lookup_on_coordinates
from ..vgraph import Graph


class NodeMergeDistanceDict(TypedDict):
    junction: float
    termination: float
    node: float


class NodeSimplificationCallBack:
    def __call__(
        self,
        node_to_fuse: npt.NDArray[np.bool_],
        node_y: npt.NDArray[np.float_],
        node_x: npt.NDArray[np.float_],
        skeleton: npt.NDArray[np.bool_],
        branches_by_nodes: npt.NDArray[np.uint64],
    ) -> npt.NDArray[np.bool_]:
        pass


NodeMergeDistanceParam: TypeAlias = bool | float | NodeMergeDistanceDict
SimplifyTopology: TypeAlias = Literal["node", "branch", "both"] | None


def simplify_graph(
    vessel_graph: Graph,
    max_spurs_distance: float = 0,
    nodes_merge_distance: NodeMergeDistanceParam = True,
    merge_small_cycles: float = 0,
    simplify_topology: SimplifyTopology = "node",
    node_simplification_criteria: Optional[NodeSimplificationCallBack] = None,
) -> Graph:
    """
    Extract the naive vasculature graph from a vessel map.
    If return label is True, the label map of branches and nodes are also computed and returned.

    Small topological corrections are applied to the graph:
        - nodes too close to each other are merged (max distance=5√2/2 by default)
        - cycles with small perimeter are merged (max size=15% of the image width by default)
        - if simplify_topology is True, the graph is simplified by merging equivalent branches (branches connecting the same junctions)
            and removing nodes with degree 2.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        return_label:
            If True, return the label map of branches and nodes.

        max_spurs_distance:
            If larger than 0, spurs (terminal branches) with a length smaller than this value are removed (disabled by default).
        nodes_merge_distance:
            If larger than 0, nodes separated by less than this distance are merged (5√2/2 by default).

        merge_small_cycles:
            If larger than 0, cycles whose nodes are closer than this value from each other are merged (disabled by default).

            .. note::
                to limit computation time, only cycles with less than 5 nodes are considered.

        simplify_topology:
            - If ``'node'``, the graph is simplified by fusing nodes with degree 2 (connected to exactly 2 branches).
            If ``'branch'``, the graph is simplified by merging equivalent branches (branches connecting the same junctions).
            If ``'both'``, both simplifications are applied.

    Returns
    -------
        The adjacency map of the graph. (Similar to an adjacency list but instead of being pair of node indices,
            each row is a boolean vector with exactly 2 pixel set to True, corresponding to the 2 nodes connected by the branch.)
        Shape: (nBranch, nNode) where nBranch and nNode are the number of branches and nodes (junctions or terminations).

        If return_label is True, also return the label map of branches
            (where each branch of the skeleton is labeled by a unique integer corresponding to its index in the adjacency matrix)
        and the coordinates of the nodes as a tuple of (y, x) where y and x are vectors of length nNode.

    """  # noqa: E501
    branches_by_nodes = vessel_graph.branch_by_node
    labeled_branches = vessel_graph.branch_labels_map
    node_yx = vessel_graph.nodes_yx_coord
    node_y, node_x = node_yx.T

    is_endpoint = compute_is_endpoints(branches_by_nodes)

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
        single_branches = is_endpoint[spurs_junction]
        spurs_junction = spurs_junction[~single_branches]
        spurs_endpoint = spurs_endpoint[~single_branches]
        # - Compute the distance between those nodes for each terminal branch
        spurs_distance = np.linalg.norm(
            (node_y[spurs_junction] - node_y[spurs_endpoint], node_x[spurs_junction] - node_x[spurs_endpoint]), axis=0
        )
        # - Select every endpoint connected to a terminal branch with a distance smaller than max_spurs_distance
        node_to_delete = np.unique(
            np.concatenate(
                (
                    spurs_junction[spurs_distance < max_spurs_distance],
                    spurs_endpoint[spurs_distance < max_spurs_distance],
                )
            )
        )
        node_to_fuse = node_to_delete[~is_endpoint[node_to_delete]]
        node_to_delete = node_to_delete[is_endpoint[node_to_delete]]
        # - Delete those nodes
        branches_by_nodes, branch_lookup2, nodes_mask = delete_nodes(branches_by_nodes, node_to_delete)
        # - Apply the lookup tables on nodes and branches
        branch_lookup = apply_lookup(branch_lookup, branch_lookup2)
        node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]
        node_to_fuse = (np.cumsum(nodes_mask) - 1)[node_to_fuse]

        # - Remove useless nodes
        node_to_fuse = node_to_fuse[node_rank(branches_by_nodes[:, node_to_fuse]) == 2]
        if np.any(node_to_fuse):
            branches_by_nodes, branch_lookup2, nodes_mask, node_labels2 = fuse_nodes(
                branches_by_nodes, node_to_fuse, (node_y, node_x)
            )

            branch_lookup = apply_lookup(branch_lookup, branch_lookup2)
            node_labels = tuple(np.concatenate((n, n1)) for n, n1 in zip(node_labels, node_labels2, strict=True))
            node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]

        is_endpoint = compute_is_endpoints(branches_by_nodes)

    if nodes_merge_distance is True:
        nodes_merge_distance = 2.5 * np.sqrt(2)
    if isinstance(nodes_merge_distance, Mapping):
        junctions_merge_distance = nodes_merge_distance.get("junction", 0)
        terminations_merge_distance = nodes_merge_distance.get("termination", 0)
        nodes_merge_distance = nodes_merge_distance.get("node", 0)
    else:
        junctions_merge_distance = 0
        terminations_merge_distance = 0

    if nodes_merge_distance > 0 or junctions_merge_distance > 0 or terminations_merge_distance > 0:
        # Merge nodes clusters smaller than nodes_merge_distance
        distances = [
            (~is_endpoint, junctions_merge_distance, True),  # distance only for junctions
            (is_endpoint, terminations_merge_distance, True),  # distance only for terminations
            (None, nodes_merge_distance, True),
        ]  # distance for all nodes
        branches_by_nodes, branch_lookup2, nodes_coord = merge_nodes_by_distance(
            branches_by_nodes, (node_y, node_x), distances
        )
        # - Apply the lookup tables on nodes and branches
        branch_lookup = apply_lookup(branch_lookup, branch_lookup2, node_labels[2])
        node_y, node_x = nodes_coord
        is_endpoint = compute_is_endpoints(branches_by_nodes)

    if merge_small_cycles > 0:
        # Merge small cycles
        # - Identify cycles
        nodes_coord = np.asarray((node_y, node_x)).T
        nodes_adjacency_matrix = branches_by_nodes.T @ branches_by_nodes - np.eye(len(node_y))

        cycles = [
            _ for _ in nx.chordless_cycles(nx.from_numpy_array(nodes_adjacency_matrix), length_bound=4) if len(_) > 2
        ]
        # - Select chord whose maximum distance between two nodes is smaller than merge_small_cycles
        cycles_max_dist = [distance_matrix(nodes_coord[cycle]).max() for cycle in cycles]
        cycles = [
            cycle for cycle, max_dist in zip(cycles, cycles_max_dist, strict=True) if max_dist < merge_small_cycles
        ]

        cycles = reduce_clusters(cycles)
        # - Merge cycles
        branches_by_nodes, branch_lookup_2, nodes_mask = merge_nodes_clusters(
            branches_by_nodes, cycles, erase_branches=True
        )
        # - Apply the lookup tables on nodes and branches
        node_y, node_x = apply_lookup_on_coordinates((node_y, node_x), nodes_mask)
        branch_lookup = apply_lookup(branch_lookup, branch_lookup_2, node_labels[2])

    if merge_small_cycles > 0 or simplify_topology in ("branch", "both"):
        # Merge 2 branches cycles (branches that are connected to the same 2 nodes)
        # - If simplify_topology is neither 'branch' nor 'both', limit the merge distance to half merge_small_cycles
        max_nodes_distance = merge_small_cycles if simplify_topology not in ("branch", "both") else None
        # - Merge equivalent branches
        branches_by_nodes, branch_lookup_2 = merge_equivalent_branches(
            branches_by_nodes,
            max_nodes_distance=max_nodes_distance,
            nodes_coordinates=(node_y, node_x),
            remove_labels=True,
        )
        # - Apply the lookup tables on branches
        branch_lookup = apply_lookup(branch_lookup, branch_lookup_2, node_labels[2])

    if simplify_topology in ("node", "both"):
        # Fuse nodes that are connected to only 2 branches
        # - Identify nodes connected to only 2 branches
        nodes_to_fuse = node_rank(branches_by_nodes) == 2

        if node_simplification_criteria is not None:
            nodes_to_fuse = node_simplification_criteria(
                nodes_to_fuse, node_y, node_x, labeled_branches > 0, branches_by_nodes
            )

        if np.any(nodes_to_fuse):
            # - Fuse nodes
            branches_by_nodes, branch_lookup2, nodes_mask, node_labels2 = fuse_nodes(
                branches_by_nodes, nodes_to_fuse, (node_y, node_x)
            )
            # - Apply the lookup tables on nodes and branches
            branch_lookup = apply_lookup(branch_lookup, branch_lookup2, node_labels[2])
            node_labels = tuple(np.concatenate((n, n1)) for n, n1 in zip(node_labels, node_labels2, strict=True))
            node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]

        nodes_to_delete = node_rank(branches_by_nodes) == 0
        if np.any(nodes_to_delete):
            # - Delete nodes
            branches_by_nodes, branch_lookup2, nodes_mask = delete_nodes(branches_by_nodes, nodes_to_delete)
            # - Apply the lookup tables on nodes and branches
            branch_lookup = apply_lookup(branch_lookup, branch_lookup2, node_labels[2])
            node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]

    if branch_lookup is not None:
        # Update branch labels
        labeled_branches = branch_lookup[labeled_branches]
    labeled_branches[node_labels[0].astype(np.int64), node_labels[1].astype(np.int64)] = node_labels[2]

    return Graph(branches_by_nodes, labeled_branches, np.stack((node_y, node_x), axis=-1))
