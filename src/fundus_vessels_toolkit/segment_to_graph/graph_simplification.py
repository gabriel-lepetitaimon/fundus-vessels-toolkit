__all__ = [
    "cluster_nodes_by_distance",
    "merge_nodes_by_distance",
    "merge_small_cycles",
    "merge_equivalent_branches",
    "remove_spurs",
    "simplify_passing_nodes",
    "simplify_graph",
    "simplify_graph_legacy",
    "NodeSimplificationCallBack",
    "NodeMergeDistances",
    "NodeMergeDistanceParam",
    "SimplifyTopology",
]

from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, TypeAlias

import networkx as nx
import numpy as np
import numpy.typing as npt

from ..utils.cluster import cluster_by_distance, iterative_reduce_clusters, reduce_clusters
from ..utils.geometric import Point, distance_matrix
from ..utils.graph.branch_by_nodes import (
    branches_by_nodes_to_branch_list,
    compute_is_endpoints,
    delete_nodes,
    fuse_nodes,
    merge_nodes_clusters,
    node_rank,
)
from ..utils.graph.branch_by_nodes import merge_equivalent_branches as merge_equivalent_branches_legacy
from ..utils.graph.branch_by_nodes import merge_nodes_by_distance as merge_nodes_by_distance_legacy
from ..utils.graph.branch_by_nodes import reduce_clusters as reduce_clusters_legacy
from ..utils.lookup_array import apply_lookup, apply_lookup_on_coordinates
from ..vascular_data_objects import VBranchGeoData, VGraph


@dataclass
class NodeMergeDistances:
    junction: float
    tip: float
    node: float


class NodeSimplificationCallBack:
    def __call__(
        self,
        node_to_fuse: npt.NDArray[np.bool_],
        node_y: npt.NDArray[np.float64],
        node_x: npt.NDArray[np.float64],
        skeleton: npt.NDArray[np.bool_],
        branches_by_nodes: npt.NDArray[np.uint64],
    ) -> npt.NDArray[np.bool_]:
        pass


NodeMergeDistanceParam: TypeAlias = bool | float | NodeMergeDistances
SimplifyTopology: TypeAlias = Literal["node", "branch", "both"] | None


def simplify_graph(
    vessel_graph: VGraph,
    max_spurs_length: float = 0,
    min_orphan_branches_length: float = 0,
    nodes_merge_distance: NodeMergeDistanceParam = True,
    iterative_nodes_merge: bool = True,
    max_cycles_length: float = 0,
    simplify_topology: SimplifyTopology = "node",
    *,
    inplace=False,
) -> VGraph:
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

        max_spurs_length:
            If larger than 0, spurs (terminal branches) with a length smaller than this value are removed (disabled by default).

        min_orphan_branches_length:
            If larger than 0, orphan branches (connected to no other branches) shorter than this value are removed (disabled by default).

        nodes_merge_distance:
            If larger than 0, nodes separated by less than this distance are merged (5√2/2 by default).

        max_cycles_length:
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
        Shape: (nBranch, nNode) where nBranch and nNode are the number of branches and nodes (junctions or tips).

        If return_label is True, also return the label map of branches
            (where each branch of the skeleton is labeled by a unique integer corresponding to its index in the adjacency matrix)
        and the coordinates of the nodes as a tuple of (y, x) where y and x are vectors of length nNode.

    """  # noqa: E501
    if not inplace:
        vessel_graph = vessel_graph.copy()

    if max_spurs_length > 0:
        remove_spurs(vessel_graph, max_spurs_length, inplace=True)

    if nodes_merge_distance is True:
        nodes_merge_distance = 2.5 * np.sqrt(2)
    if isinstance(nodes_merge_distance, Mapping):
        junctions_merge_distance = nodes_merge_distance.get("junction", 0)
        tips_merge_distance = nodes_merge_distance.get("tip", 0)
        nodes_merge_distance = nodes_merge_distance.get("node", 0)
    elif isinstance(nodes_merge_distance, NodeMergeDistances):
        junctions_merge_distance = nodes_merge_distance.junction
        tips_merge_distance = nodes_merge_distance.tip
        nodes_merge_distance = nodes_merge_distance.node
    else:
        junctions_merge_distance = 0
        tips_merge_distance = 0

    if tips_merge_distance > 0:
        merge_nodes_by_distance(
            vessel_graph,
            max_distance=tips_merge_distance,
            nodes_type="endpoints",
            only_connected_nodes=False,
            inplace=True,
            iterative_clustering=iterative_nodes_merge,
        )

    if max_cycles_length > 0:
        merge_small_cycles(vessel_graph, max_cycles_length, inplace=True)

    if junctions_merge_distance > 0:
        merge_nodes_by_distance(
            vessel_graph,
            max_distance=junctions_merge_distance,
            nodes_type="junction",
            only_connected_nodes=True,
            inplace=True,
            iterative_clustering=iterative_nodes_merge,
        )

    if nodes_merge_distance > 0:
        merge_nodes_by_distance(
            vessel_graph,
            max_distance=nodes_merge_distance,
            nodes_type="all",
            inplace=True,
            iterative_clustering=iterative_nodes_merge,
        )

    if max_cycles_length > 0 or simplify_topology in ("branch", "both"):
        max_nodes_distance = max_cycles_length if simplify_topology not in ("branch", "both") else None
        merge_equivalent_branches(vessel_graph, max_nodes_distance, inplace=True)

    if min_orphan_branches_length > 0:
        remove_orphan_branches(vessel_graph, min_orphan_branches_length, inplace=True)

    if simplify_topology in ("node", "both"):
        simplify_passing_nodes(vessel_graph, inplace=True)

    return vessel_graph


def cluster_nodes_by_distance(
    vessel_graph: VGraph,
    max_distance: float,
    nodes_type: Literal["all", "junction", "endpoints"] = "all",
    only_connected_nodes: Optional[bool] = None,
    iterative_clustering: bool = False,
) -> List[set[int]]:
    """
    Cluster nodes of the vessel graph by distance.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        max_distance:
            The maximum distance between two nodes to be considered in the same cluster.

        nodes_type:
            The type of nodes to consider:`
            - ``'all'``: all nodes are considered;
            - ``'junction'``: only junctions and bifurcations are considered;
            - ``'endpoints'``: only endpoints are considered.

        only_connected_nodes:
            - If True, only nodes connected by a branch can be clustered.
            - If False, only nodes not connected by a branch can be clustered.
            - If None (by default), all nodes can be clustered.


    Returns
    -------
        A list of sets of nodes indices. Each set contains the indices of nodes that are closer than max_distance from each other.

    """  # noqa: E501

    if nodes_type == "junction":
        nodes_id = vessel_graph.non_endpoints_nodes()
    elif nodes_type == "endpoints":
        nodes_id = vessel_graph.endpoints_nodes()
    else:
        nodes_id = None

    if only_connected_nodes:
        # --- Cluster only connected nodes ---
        # ... Only edges of the graph are considered
        if nodes_id is None:
            branches = vessel_graph.branch_list
        else:
            branch_id = np.argwhere(np.isin(vessel_graph.branch_list, nodes_id).all(axis=1)).flatten()
            branches = vessel_graph.branch_list[branch_id]
        # Exclude branches with the same start and end nodes
        branches = branches[branches[:, 0] != branches[:, 1]]
        nodes_coord = vessel_graph.nodes_coord()
        # Compute the distance between the nodes of each branch
        branch_dist = np.linalg.norm(nodes_coord[branches[:, 0]] - nodes_coord[branches[:, 1]], axis=1)
        # Reduce the clusters
        if iterative_clustering:
            bmask = branch_dist < max_distance
            return iterative_reduce_clusters(branches[bmask], branch_dist[bmask], max_distance)
        return reduce_clusters(branches[branch_dist < max_distance])

    elif only_connected_nodes is None:
        # --- Cluster all nodes ---
        if nodes_id is None:
            return cluster_by_distance(vessel_graph.nodes_coord(), max_distance, iterative=iterative_clustering)
        else:
            nodes_coord = vessel_graph.nodes_coord()[nodes_id]
            clusters = cluster_by_distance(nodes_coord, max_distance, iterative=iterative_clustering)
            return [{nodes_id[_] for _ in cluster} for cluster in clusters]

    else:
        # --- Cluster only unconnected nodes ---
        # ... For each pair of independent subgraphs of the vessel graph, only their closest nodes can be clustered
        # Get the independent subgraphs of the vessel graph
        connected_nodes_id = vessel_graph.nodes_connected_graphs()
        # Filter the nodes type
        connected_nodes_id = [np.intersect1d(_, nodes_id, assume_unique=True) for _ in connected_nodes_id]
        # Compute the distance between all these nodes
        all_nodes_id = np.concatenate(connected_nodes_id)
        all_nodes_coord = vessel_graph.nodes_coord()[all_nodes_id]
        distance = np.linalg.norm(all_nodes_coord[:, None] - all_nodes_coord, axis=-1)
        # and create a short id for each subgraph
        connected_nodes_short_id = []
        n = 0
        for connected_nodes in connected_nodes_id:
            connected_nodes_short_id.append(np.arange(n, n + len(connected_nodes)))
            n += len(connected_nodes)
        # For each pair of independent subgraphs, find the closest nodes ...
        clusters = []
        for i, (id1, sid1) in enumerate(zip(connected_nodes_id, connected_nodes_short_id, strict=True)):
            for j, (id2, sid2) in enumerate(
                zip(connected_nodes_id[i + 1 :], connected_nodes_short_id[i + 1 :], strict=True)
            ):
                j += i + 1
                argmin = np.argmin(distance[sid1][:, sid2])
                closest1, closest2 = (argmin // len(sid2), argmin % len(sid2))
                # ... and add them to the clusters if they are closer than max_distance
                if distance[sid1[closest1], sid2[closest2]] < max_distance:
                    clusters.append({id1[closest1], id2[closest2]})
        return reduce_clusters(clusters)


def merge_nodes_by_distance(
    vessel_graph: VGraph,
    max_distance: float,
    nodes_type: Literal["all", "junction", "endpoints"] = "all",
    only_connected_nodes: Optional[bool] = None,
    *,
    inplace=False,
    iterative_clustering=False,
) -> VGraph:
    """
    Merge nodes of the vessel graph by distance.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        max_distance:
            The maximum distance between two nodes to be considered in the same cluster.

        nodes_type:
            The type of nodes to consider:`
            - ``'all'``: all nodes are considered;
            - ``'junction'``: only junctions and bifurcations are considered;
            - ``'endpoints'``: only endpoints are considered.

    Returns
    -------
        The modified graph with the nodes merged.

    """  # noqa: E501

    clusters = cluster_nodes_by_distance(
        vessel_graph=vessel_graph,
        max_distance=max_distance,
        nodes_type=nodes_type,
        only_connected_nodes=only_connected_nodes,
        iterative_clustering=iterative_clustering,
    )
    nodes_weight = None if nodes_type != "all" else (~vessel_graph.endpoints_nodes(as_mask=True)) * 1
    return vessel_graph.merge_nodes(clusters, nodes_weight=nodes_weight, inplace=inplace, assume_reduced=True)


def merge_small_cycles(vessel_graph: VGraph, max_cycle_size: float, inplace=False) -> VGraph:
    """
    Merge small cycles of the vessel graph.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        max_cycle_size:
            The maximum distance between two nodes to be considered in the same cluster.

    Returns
    -------
        The modified graph with the nodes merged.

    """  # noqa: E501

    nodes_coord = vessel_graph.nodes_coord()
    nx_graph = nx.from_numpy_array(vessel_graph.node_adjacency_matrix())
    cycles = [_ for _ in nx.chordless_cycles(nx_graph, length_bound=4) if len(_) > 2]
    cycles_max_dist = [distance_matrix(nodes_coord[cycle]).max() for cycle in cycles]
    cycles = [cycle for cycle, max_dist in zip(cycles, cycles_max_dist, strict=True) if max_dist < max_cycle_size]
    cycles = reduce_clusters(cycles)
    return vessel_graph.merge_nodes(cycles, inplace=inplace)


def merge_equivalent_branches(vessel_graph: VGraph, max_nodes_distance: float = None, inplace=False) -> VGraph:
    """
    Merge equivalent branches of the vessel graph.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        max_nodes_distance:
            The maximum distance between two nodes to be considered in the same cluster.

    Returns
    -------
        The modified graph with the nodes merged.

    """  # noqa: E501

    branches, branches_inverse, branches_count = np.unique(
        vessel_graph.branch_list, return_counts=True, return_inverse=True, axis=0
    )

    equi_branches_mask = branches_count > 1
    if max_nodes_distance is not None:
        equi_branches = branches[equi_branches_mask]
        nodes_coord = vessel_graph.nodes_coord()
        equi_branches_dist = np.linalg.norm(nodes_coord[equi_branches[:, 0]] - nodes_coord[equi_branches[:, 1]], axis=1)
        equi_branches_mask[equi_branches_mask] &= equi_branches_dist < max_nodes_distance

    branch_to_remove = []
    for duplicate_id in np.argwhere(equi_branches_mask).flatten():
        duplicate_branches_id = np.argwhere(branches_inverse.flatten() == duplicate_id).flatten()
        branch_to_remove.extend(duplicate_branches_id[1:])

    return vessel_graph.delete_branches(branch_to_remove, inplace=inplace)


def remove_spurs(vessel_graph: VGraph, max_spurs_length: float = 0, inplace=False) -> VGraph:
    """
    Remove spurs (terminal branches) whose chord length is shorter than max_spurs_distance.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        max_spurs_length:
            If larger than 0, spurs (terminal branches) with a length smaller than this value are removed (disabled by default).

    Returns
    -------
        The modified graph with the spurs removed.

    """  # noqa: E501

    terminal_branches = vessel_graph.endpoints_branches()
    terminal_branches_length = vessel_graph.branches_arc_length(terminal_branches)
    return vessel_graph.delete_branches(terminal_branches[terminal_branches_length < max_spurs_length], inplace=inplace)


def remove_orphan_branches(vessel_graph: VGraph, min_length: float | bool = 0, inplace=False) -> VGraph:
    """
    Remove single branches (branches connected to no other branches) whose length is shorter than min_length.

    Parameters
    ----------
    vessel_graph :
        The graph of the vasculature extracted from the vessel
    min_length : float, optional
        The minimum length of the single branches to keep, by default 0.
    inplace : bool, optional
        If True, modify the graph in place, by default False.

    Returns
    -------
        The modified graph with the single branches removed.
    """
    if not min_length or (np.isscalar(min_length) and min_length <= 0):
        return vessel_graph
    orphan_branches = vessel_graph.orphan_branches()
    if np.isscalar(min_length):
        single_branches_length = vessel_graph.branches_arc_length(orphan_branches)
        orphan_branches = orphan_branches[single_branches_length < min_length]
    return vessel_graph.delete_branches(orphan_branches, inplace=inplace)


def simplify_passing_nodes(
    vessel_graph: VGraph,
    *,
    not_fusable: Optional[npt.ArrayLike] = None,
    min_angle: float = 0,
    with_same_label=None,
    inplace=False,
) -> VGraph:
    """
    Merge nodes of the vessel graph that are connected to only 2 branches.

    Parameters
    ----------
        vessel_graph:
            The graph of the vasculature extracted from the vessel map.

        not_fusable:
            A list of nodes that should not be merged.

        min_angle:
            Under this minimum angle (in degrees) between the two branches connected to a passing node, the node is considered as a junction and is not removed.

            (Require the terminaison tangents field to be field in `VGeometricData`)

        with_same_label:
            If not None, the nodes are merged only if they have the same label.
            If a string, use ``vessel_graph.branches_attr[with_same_label]`` as the labels.

    Returns
    -------
        The modified graph with the nodes merged.

    """  # noqa: E501
    incident_branches = None

    # === Get all nodes with degree 2 ===
    nodes_to_fuse = np.argwhere(vessel_graph.nodes_degree() == 2).flatten()

    if not_fusable is not None:
        # === Filter out nodes that should not be merged ===
        nodes_to_fuse = np.setdiff1d(nodes_to_fuse, not_fusable)

    if min_angle > 0:
        # === Filter out nodes with a too small angle between their two incident branches ===
        geo_data = vessel_graph.geometric_data()
        d = np.stack(geo_data.tip_geodata_around_node(nodes_to_fuse, [VBranchGeoData.Fields.TIPS_TANGENTS]))
        incident_branches = d["branches"]
        t = d[VBranchGeoData.Fields.TIPS_TANGENTS]
        cos = np.sum(t[..., 0] * t[..., 1], axis=1)
        nodes_to_fuse = nodes_to_fuse[cos >= np.cos(np.deg2rad(min_angle))]

    if with_same_label is not None:
        # === Filter nodes which don't have the same label ===
        if isinstance(with_same_label, str):
            # Attempt to get the labels from the branches attributes
            with_same_label = vessel_graph.branches_attr[with_same_label]

        if incident_branches is None:
            # Get the incident branches if not already computed
            incident_branches = vessel_graph.incident_branches_individual(nodes_to_fuse)

        same_label = [with_same_label[b1] == with_same_label[b2] for b1, b2 in incident_branches]
        nodes_to_fuse = nodes_to_fuse[same_label]

    if len(nodes_to_fuse):
        return vessel_graph.fuse_nodes(
            nodes_to_fuse, inplace=inplace, quiet_invalid_node=True, incident_branches=incident_branches
        )
    return vessel_graph


def simplify_graph_legacy(
    vessel_graph: VGraph,
    max_spurs_distance: float = 0,
    nodes_merge_distance: NodeMergeDistanceParam = True,
    merge_small_cycles: float = 0,
    simplify_topology: SimplifyTopology = "node",
    node_simplification_criteria: Optional[NodeSimplificationCallBack] = None,
) -> VGraph:
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
        Shape: (nBranch, nNode) where nBranch and nNode are the number of branches and nodes (junctions or endpoints).

        If return_label is True, also return the label map of branches
            (where each branch of the skeleton is labeled by a unique integer corresponding to its index in the adjacency matrix)
        and the coordinates of the nodes as a tuple of (y, x) where y and x are vectors of length nNode.

    """  # noqa: E501
    branches_by_nodes = vessel_graph.branches_by_nodes()
    labeled_branches = vessel_graph.geometric_data().branches_label_map()
    node_yx = vessel_graph.nodes_coord().astype(np.int)
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
        spurs_junction, spurs_endpoint = branches_by_nodes_to_branch_list(branches_by_nodes[spurs_branches]).T
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
        # node_to_fuse = node_to_fuse[node_rank(branches_by_nodes[:, node_to_fuse]) == 2]
        # if np.any(node_to_fuse):
        #    branches_by_nodes, branch_lookup2, nodes_mask, node_labels2 = fuse_nodes(
        #        branches_by_nodes, node_to_fuse, (node_y, node_x)
        #    )

        #    branch_lookup = apply_lookup(branch_lookup, branch_lookup2)
        #    node_labels = tuple(np.concatenate((n, n1)) for n, n1 in zip(node_labels, node_labels2, strict=True))
        #    node_y, node_x = node_y[nodes_mask], node_x[nodes_mask]

        is_endpoint = compute_is_endpoints(branches_by_nodes)

    if nodes_merge_distance is True:
        nodes_merge_distance = 2.5 * np.sqrt(2)
    if isinstance(nodes_merge_distance, Mapping):
        junctions_merge_distance = nodes_merge_distance.get("junction", 0)
        tips_merge_distance = nodes_merge_distance.get("tip", 0)
        nodes_merge_distance = nodes_merge_distance.get("node", 0)
    else:
        junctions_merge_distance = 0
        tips_merge_distance = 0

    if nodes_merge_distance > 0 or junctions_merge_distance > 0 or tips_merge_distance > 0:
        # Merge nodes clusters smaller than nodes_merge_distance
        distances = [
            (~is_endpoint, junctions_merge_distance, True),  # distance only for junctions
            (is_endpoint, tips_merge_distance, True),  # distance only for tips
            (None, nodes_merge_distance, True),
        ]  # distance for all nodes
        branches_by_nodes, branch_lookup2, nodes_coord = merge_nodes_by_distance_legacy(
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

        cycles = reduce_clusters_legacy(cycles)
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
        branches_by_nodes, branch_lookup_2 = merge_equivalent_branches_legacy(
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

    geo_data = vessel_graph.geometric_data().copy()
    if branch_lookup is not None:
        # Update branch labels
        geo_data._reindex_branches(branch_lookup[1:] - 1)
        labeled_branches = branch_lookup[labeled_branches]

    labeled_branches[node_labels[0].astype(np.int64), node_labels[1].astype(np.int64)] = node_labels[2]
    geo_data.node_coord = {i: Point(y, x) for i, (y, x) in enumerate(zip(node_y, node_x))}

    return VGraph.from_branch_by_nodes(branches_by_nodes, geo_data)
