__all__ = [
    "cluster_nodes_by_distance",
    "merge_nodes_by_distance",
    "merge_small_cycles",
    "merge_equivalent_branches",
    "remove_spurs",
    "simplify_passing_nodes",
    "simplify_graph",
    "NodeSimplificationCallBack",
    "ReconnectEndpointsArg",
    "SimplifyTopology",
]

import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, TypeAlias

import networkx as nx
import numpy as np
import numpy.typing as npt

from fundus_vessels_toolkit.utils import if_none

from ..utils.cluster import cluster_by_distance, iterative_reduce_clusters, reduce_clusters
from ..utils.dataclass import UpdateableDataclass
from ..utils.geometric import Point, distance_matrix
from ..utils.lookup_array import create_removal_lookup
from ..vascular_data_objects import VBranchGeoData, VGraph


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


SimplifyTopology: TypeAlias = Literal["node", "branch", "both"] | None


@dataclass
class ReconnectEndpointsArg(UpdateableDataclass):
    max_distance: float = 30
    max_angle: float = 30
    intercept_snapping_distance: float = 20
    endpoints_to_endpoints_max_distance: Optional[float] = None
    tangent: VBranchGeoData.Key | npt.NDArray[np.float64] = VBranchGeoData.Fields.TIPS_TANGENT
    bspline: VBranchGeoData.Key = VBranchGeoData.Fields.BSPLINE


@dataclass
class GraphSimplifyArg(UpdateableDataclass):
    """Arguments for the graph simplification.
    See :func:`simplify_graph` for more details.
    """

    max_spurs_length: float = 0
    reconnect_endpoints: bool | ReconnectEndpointsArg = False
    max_cycles_length: float = 20
    junctions_merge_distance: float = 20
    min_orphan_branches_length: float = 30
    simplify_topology: SimplifyTopology | bool = "node"
    passing_node_min_angle: float = 110


def simplify_graph(
    vessel_graph: VGraph,
    arg: Optional[GraphSimplifyArg] = None,
    /,
    *,
    max_spurs_length: Optional[float] = None,
    reconnect_endpoints: Optional[bool | ReconnectEndpointsArg] = None,
    max_cycles_length: Optional[float] = None,
    junctions_merge_distance: Optional[float] = None,
    min_orphan_branches_length: Optional[float] = None,
    simplify_topology: Optional[SimplifyTopology] | None = None,
    passing_node_min_angle: Optional[float] = None,
    inplace: bool = False,
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
            If larger than 0, spurs (terminal branches) with a length smaller than this value are removed.

            Default is 0 (disabled).

        reconnect_endpoints:
            If ``True`` or :class:`ReconnectEndpointsArg`, the endpoints are reconnected to their closest branch in the direction of their tangent or to the closest facing endpoints.

            Default is ``False``.

        max_cycles_length:
            If larger than 0, cycles whose nodes are closer than this value from each other are merged (disabled by default).

            Default is 20.

            .. note::
                to limit computation time, only cycles with less than 5 nodes are considered.

        junctions_merge_distance:
            If larger than 0, nodes separated by less than this distance are merged.

            Default is 20.

        min_orphan_branches_length:
            If larger than 0, orphan branches (connected to no other branches) shorter than this value are removed.

            Default is 30.

        simplify_topology:
            - If ``'node'``, the graph is simplified by fusing nodes with degree 2 (connected to exactly 2 branches).
            - If ``'branch'``, the graph is simplified by merging equivalent branches (branches connecting the same junctions).
            - If ``'both'`` or ``True``, both simplifications are applied.

            Default is ``'node'``.

        passing_node_min_angle:
            Under this minimum angle (in degrees) between the two branches connected to a passing node, the node is considered as a junction and is not removed.

            Default is 110.

            If simplify_topology does not include ``'node'``, this parameter is ignored.

    Returns
    -------
        The adjacency map of the graph. (Similar to an adjacency list but instead of being pair of node indices,
            each row is a boolean vector with exactly 2 pixel set to True, corresponding to the 2 nodes connected by the branch.)
        Shape: (nBranch, nNode) where nBranch and nNode are the number of branches and nodes (junctions or tips).

        If return_label is True, also return the label map of branches
            (where each branch of the skeleton is labeled by a unique integer corresponding to its index in the adjacency matrix)
        and the coordinates of the nodes as a tuple of (y, x) where y and x are vectors of length nNode.

    """  # noqa: E501
    if arg is None:
        arg = GraphSimplifyArg()
    arg = arg.update(
        max_spurs_length=max_spurs_length,
        reconnect_endpoints=reconnect_endpoints,
        max_cycles_length=max_cycles_length,
        junctions_merge_distance=junctions_merge_distance,
        min_orphan_branches_length=min_orphan_branches_length,
        simplify_topology=simplify_topology,
        passing_node_min_angle=passing_node_min_angle,
    )

    if not inplace:
        vessel_graph = vessel_graph.copy()

    if arg.max_spurs_length > 0:
        remove_spurs(vessel_graph, arg.max_spurs_length, inplace=True)

    if arg.max_cycles_length > 0:
        merge_small_cycles(vessel_graph, arg.max_cycles_length, inplace=True)

    if arg.junctions_merge_distance > 0:
        merge_nodes_by_distance(
            vessel_graph,
            max_distance=arg.junctions_merge_distance,
            nodes_type="junction",
            only_connected_nodes=True,
            iterative_clustering=True,
            inplace=True,
        )

    if type(arg.reconnect_endpoints) is bool:
        if arg.reconnect_endpoints:
            vessel_graph = _reconnect_endpoints(vessel_graph, inplace=True)
    else:
        vessel_graph = _reconnect_endpoints(vessel_graph, arg.reconnect_endpoints, inplace=True)

    if arg.max_cycles_length > 0 or arg.simplify_topology in ("branch", "both") or arg.simplify_topology is True:
        max_nodes_distance = arg.max_cycles_length if arg.simplify_topology not in ("branch", "both") else None
        merge_equivalent_branches(vessel_graph, max_nodes_distance, inplace=True)

    if arg.simplify_topology in ("node", "both") or arg.simplify_topology is True:
        simplify_passing_nodes(vessel_graph, min_angle=arg.passing_node_min_angle, inplace=True)

    if arg.min_orphan_branches_length > 0:
        remove_orphan_branches(vessel_graph, arg.min_orphan_branches_length, inplace=True)

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
        nodes_id = vessel_graph.junctions_nodes()
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
    if len(cycles) == 0:
        return vessel_graph
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
    vgraph: VGraph,
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

    # === Get all nodes with degree 2 ===
    nodes_to_fuse = vgraph.passing_nodes(exclude_loop=True)

    if not_fusable is not None:
        # === Filter out nodes that should not be merged ===
        nodes_to_fuse = np.setdiff1d(nodes_to_fuse, not_fusable)

    if len(nodes_to_fuse) == 0:
        return vgraph

    incident_branches = None
    if min_angle > 0:
        # === Filter out nodes with a too small angle between their two incident branches ===
        geo_data = vgraph.geometric_data()
        incident_branches, idirs = vgraph.incident_branches_individual(nodes_to_fuse, return_branch_direction=True)
        incident_branches = np.stack(incident_branches)
        t = np.stack([geo_data.tips_tangent(b, d) for b, d in zip(incident_branches, idirs, strict=True)])
        cos = np.sum(t[:, 1, :] * t[:, 0, :], axis=1)
        fuseable_nodes = cos <= np.cos(np.deg2rad(min_angle))
        nodes_to_fuse = nodes_to_fuse[fuseable_nodes]
        incident_branches = incident_branches[fuseable_nodes]

    if with_same_label is not None:
        # === Filter nodes which don't have the same label ===
        if isinstance(with_same_label, str):
            # Attempt to get the labels from the branches attributes
            with_same_label = vgraph.branches_attr[with_same_label]
        with_same_label = np.asarray(with_same_label)

        if incident_branches is None:
            # Get the incident branches if not already computed
            incident_branches = np.stack(vgraph.incident_branches_individual(nodes_to_fuse))

        same_label = with_same_label[incident_branches[:, 0]] == with_same_label[incident_branches[:, 1]]
        nodes_to_fuse = nodes_to_fuse[same_label]
        incident_branches = incident_branches[same_label]

    if len(nodes_to_fuse):
        return vgraph.fuse_nodes(
            nodes_to_fuse, inplace=inplace, quiet_invalid_node=True, incident_branches=incident_branches
        )
    return vgraph


def find_facing_endpoints(
    vessel_graph: VGraph,
    max_distance: float = 100,
    max_angle: float = 30,
    filter: Optional[Literal["closest", "exclusive"]] = "exclusive",
    tangent: VBranchGeoData.Key = VBranchGeoData.Fields.TIPS_TANGENT,
):
    """
    Find pairs of endpoints that are facing each other.

    Parameters
    ----------
    vessel_graph: VGraph
        The vasculature graph.

    max_distance: float
        The maximum distance between the two endpoints.

    max_angle: float
        The maximum angle between the two endpoints tangents.

    exclusive: bool
        If True, the endpoints are considered facing each other only if they are the closest to each other.

    tangent: VBranchGeoData.Key | npt.NDArray[np.float64]
        The tangent field to use to find the facing endpoints.

    Returns
    -------
    facing_endpoints: npt.NDArray[np.int]
        An (E, 2) array where each row contains the indices of two endpoints that are facing each other.
    """  # noqa: E501
    geodata = vessel_graph.geometric_data()

    endp_branch, endp_mask = vessel_graph.endpoints_branches(return_endpoints_mask=True)
    b, endp_tip_id = np.where(endp_mask)
    endp_branch = endp_branch[b]
    endp = vessel_graph.branch_list[endp_branch, endp_tip_id]
    endp_first_tip = endp_tip_id == 0
    endp_pos = geodata.nodes_coord()[endp]

    dist = np.linalg.norm(endp_pos[:, None, :] - endp_pos[None, :, :], axis=2)
    facing = dist <= max_distance
    facing[np.diag_indices(len(endp))] = False

    if max_angle > 0:
        endp_tan = -geodata.tips_tangent(endp_branch, endp_first_tip)
        endp_to_endp_dir = endp_pos[None, :, :] - endp_pos[:, None, :]  # (origin, destination, yx)
        endp_to_endp_norm = np.linalg.norm(endp_to_endp_dir, axis=2)
        endp_to_endp_dir[endp_to_endp_norm != 0] /= endp_to_endp_norm[endp_to_endp_norm != 0, None]
        cos = np.sum(endp_to_endp_dir * endp_tan[:, None, :], axis=2)
        facing &= cos >= np.cos(np.deg2rad(max_angle))
        facing &= facing.T

    endpoint_pairs = np.argwhere(facing)
    if len(endpoint_pairs) == 0:
        return np.empty((0, 2), dtype=int)

    endpoint_pairs = endpoint_pairs[endpoint_pairs[:, 0] < endpoint_pairs[:, 1]]
    if filter is not None:
        pairs_dist = dist[tuple(endpoint_pairs.T)]
        endpoint_pairs = endpoint_pairs[np.argsort(pairs_dist)]
        unique_pairs = [endpoint_pairs[0]]
        if filter == "exclusive":
            for pair in endpoint_pairs[1:]:
                if not np.isin(pair, unique_pairs).any():
                    unique_pairs.append(pair)
        else:
            _, first_pos = np.unique(endpoint_pairs.flatten(), return_index=True)
            first_pos = np.unique(first_pos // 2)
            unique_pairs = endpoint_pairs[first_pos]
        endpoint_pairs = np.array(unique_pairs, dtype=int)
    return endp[endpoint_pairs]


def find_endpoints_branches_intercept(
    vessel_graph: VGraph,
    max_distance: float = 100,
    angle_tolerance: float = 10,
    intercept_snapping_distance: float = 30,
    *,
    ignore_endpoints: Optional[npt.NDArray[np.int_]] = None,
    omit_endpoints_to_endpoints: bool = False,
    tangent: VBranchGeoData.Key | npt.NDArray[np.float64] = VBranchGeoData.Fields.TIPS_TANGENT,
) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """
    Find candidates for reconnecting endpoints to their closest branch in the direction of their tangent.

    Parameters
    ----------
    vessel_graph: VGraph
        The vasculature graph.

    max_distance: float
        The maximum distance between an endpoint and the point of intercept on a branch.

    snap_intercept_distance: float
        The maximum distance between two point of intercept on the same branch. If two intercepts are closer than this distance, they are merged together.

    tangent: VBranchGeoData.Key | npt.NDArray[np.float64]
        The tangent field to use to find the intercepts.

    bspline: VBranchGeoData.Key
        The bspline field to use to find the intercepts.

    Returns
    -------
    new_edges: npt.NDArray[np.int]
        An (E, 2) array where each row contains the indices of two nodes that can be reconnected together.
        Those nodes are either nodes from the graph (at least on is an existing endpoint), but may also be new nodes defined by the following two arrays.

    new_nodes: npt.NDArray[np.int]
        An (n, 2) array where each row contains the index of a new node and the index of the branch it's situated on.

    new_nodes_yx: npt.NDArray[np.float64]
        An (n, 2) array containing the coordinates of the new nodes defined in ``new_nodes``.

    """  # noqa: E501
    import torch

    from ..utils.cpp_extensions.fvt_cpp import find_closest_branches as find_closest_branches_cpp

    endpoints = vessel_graph.endpoints_nodes()
    if ignore_endpoints is not None:
        endpoints = np.setdiff1d(endpoints, ignore_endpoints)

    endpoints_branches, idirs = vessel_graph.incident_branches_individual(endpoints, return_branch_direction=True)
    endpoints_branches = np.array(endpoints_branches).flatten()
    gdata = vessel_graph.geometric_data()
    nodes_yx = gdata.nodes_coord()
    n_nodes = len(nodes_yx)
    endpoints_yx = nodes_yx[endpoints]
    branch_list = vessel_graph.branch_list

    if not isinstance(tangent, np.ndarray):
        endpoints_t = -np.stack([gdata.tips_tangent(b, d) for b, d in zip(endpoints_branches, idirs, strict=True)])
    else:
        endpoints_t = tangent

    # === Intercept all branches with the endpoints tangents ===
    nearest_branch, intercept = find_closest_branches_cpp(
        torch.from_numpy(gdata.branches_label_map(connect_nodes=True)).int(),
        torch.from_numpy(endpoints_yx).int(),
        torch.from_numpy(endpoints_t).float(),
        max_distance,
        np.deg2rad(2 * angle_tolerance),
    )
    if nearest_branch.max() > len(branch_list):
        warnings.warn("Invalid nearest branch index.")
        nearest_branch[nearest_branch > len(branch_list)] = -1
    nearest_branch = nearest_branch.numpy()
    intercept = intercept.numpy()

    # === Ignore endpoints that have no intercept with any branch ===
    no_intercept = (nearest_branch < 0) | (nearest_branch == endpoints_branches)
    if np.all(no_intercept):
        return np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=np.float64)
    endpoints = endpoints[~no_intercept]
    intercept = intercept[~no_intercept]
    endpoints_yx = endpoints_yx[~no_intercept]
    nearest_branch = nearest_branch[~no_intercept]

    # === Redirect intercept points that are near branch tips to existing nodes ===
    dist_to_branch_tips = np.linalg.norm(intercept[:, None, :] - nodes_yx[branch_list[nearest_branch]], axis=2)
    closest_tip = np.argmin(dist_to_branch_tips, axis=1)
    dist_to_closest_tip = np.take_along_axis(dist_to_branch_tips, closest_tip[:, None], axis=1).squeeze(axis=1)
    closest_tip = branch_list[nearest_branch, closest_tip]
    redirect_to_tip = dist_to_closest_tip <= intercept_snapping_distance * 0.66

    endp_to_tips_edges = np.empty((0, 2), dtype=int)
    if np.any(redirect_to_tip):
        redirect_to_tip = np.argwhere(redirect_to_tip).flatten()
        closest_tip_is_endpoints = np.isin(closest_tip[redirect_to_tip], endpoints)
        redirect_to_node = redirect_to_tip[~closest_tip_is_endpoints]

        endp_to_tips_edges = np.stack((endpoints[redirect_to_node], closest_tip[redirect_to_node]), axis=1)
        redirected_endpoints = np.zeros(n_nodes, dtype=bool)
        redirected_endpoints[endp_to_tips_edges[:, 0]] = True

        if not omit_endpoints_to_endpoints:
            redirect_to_endp = redirect_to_tip[closest_tip_is_endpoints]
            new_endp_edges = np.stack((endpoints[redirect_to_endp], closest_tip[redirect_to_endp]), axis=1)
            new_endp_edges = np.unique(np.sort(new_endp_edges, axis=1), axis=0)
            endp_to_tips_edges = np.concatenate((endp_to_tips_edges, new_endp_edges), axis=0)
            redirected_endpoints[new_endp_edges.flatten()] = True

        redirected_endpoints = redirected_endpoints[endpoints]
        if np.all(redirected_endpoints):
            return endp_to_tips_edges, np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=np.float64)

        endpoints = endpoints[~redirected_endpoints]
        intercept = intercept[~redirected_endpoints]
        nearest_branch = nearest_branch[~redirected_endpoints]

    # === Merge intercept points of the same branches that are too close to each other ===
    #: The id of the new nodes
    #:         (They will be shifted by the number of existing nodes at the end of the function)
    new_intercept_nodes_id = np.arange(len(endpoints))

    endp_to_intercept_nodes_edges = np.array([endpoints, new_intercept_nodes_id], dtype=int).T
    new_intercept_nodes = np.array([new_intercept_nodes_id, nearest_branch], dtype=int).T

    branches, branches_inv, branches_count = np.unique(nearest_branch, return_inverse=True, return_counts=True)
    if intercept_snapping_distance > 0 and np.any(branches_count > 1):
        intercept_nodes_merge_lookup = np.arange(len(endpoints))
        intercept_to_remove = []
        for b, count in zip(branches, branches_count, strict=True):
            if count <= 1:
                continue
            intercept_ids = np.argwhere(branches_inv == b).flatten()
            clusters = cluster_by_distance(intercept[intercept_ids], intercept_snapping_distance, iterative=True)
            for cluster in clusters:
                if len(cluster) > 1:
                    cluster = np.sort(cluster)
                    cluster_ids = intercept_ids[cluster]
                    intercept_nodes_merge_lookup[cluster_ids[1:]] = cluster_ids[0]
                    intercept_to_remove.extend(cluster_ids[1:])
                    intercept[cluster_ids[0]] = np.mean(intercept[cluster_ids], axis=0)

        if len(intercept_to_remove):
            del_lookup = create_removal_lookup(intercept_to_remove, length=len(endpoints))
            intercept_nodes_merge_lookup = del_lookup[intercept_nodes_merge_lookup]

            intercept = np.delete(intercept, intercept_to_remove, axis=0)
            new_intercept_nodes = np.delete(new_intercept_nodes, intercept_to_remove, axis=0)
            new_intercept_nodes[:, 0] = intercept_nodes_merge_lookup[new_intercept_nodes[:, 0]]
            endp_to_intercept_nodes_edges[:, 1] = intercept_nodes_merge_lookup[endp_to_intercept_nodes_edges[:, 1]]

    endp_to_intercept_nodes_edges[:, 1] += n_nodes
    new_intercept_nodes[:, 0] += n_nodes

    return np.concatenate((endp_to_tips_edges, endp_to_intercept_nodes_edges), axis=0), new_intercept_nodes, intercept


def reconnect_endpoints(
    vessel_graph: VGraph,
    arg: Optional[ReconnectEndpointsArg] = None,
    /,
    *,
    max_distance: Optional[float] = None,
    endpoints_to_endpoints_max_distance: Optional[float] = None,
    intercept_snapping_distance: Optional[float] = None,
    max_angle: Optional[float] = None,
    tangent: Optional[VBranchGeoData.Key | npt.NDArray[np.float64]] = None,
    bspline: Optional[VBranchGeoData.Key] = None,
    inplace=False,
) -> VGraph:
    """
    Reconnect endpoints to their closest branch in the direction of their tangent.

    Parameters
    ----------
    vessel_graph : VGraph
        The vasculature graph.
    max_distance : float, optional
        The maximum distance between an endpoint and the point of intercept on a branch, by default 100
    intercept_snapping_distance : float, optional
        The maximum distance between two point of intercept on the same branch. If two intercepts are closer than this distance, they are merged together, by default 30
    tangent : VBranchGeoData.Key | npt.NDArray[np.float64], optional
        The tangent field to use to find the intercepts, by default VBranchGeoData.Fields.TIPS_TANGENT
    bspline : VBranchGeoData.Key, optional
        The bspline field to use to find the intercepts, by default VBranchGeoData.Fields.BSPLINE
    inplace : bool, optional
        If True, modify the graph in place, by default False
    Returns
    -------
    VGraph
        The modified graph with the endpoints reconnected.
    """  # noqa: E501

    if arg is None:
        arg = ReconnectEndpointsArg()
    arg.update(
        max_distance=max_distance,
        endpoints_to_endpoints_max_distance=endpoints_to_endpoints_max_distance,
        intercept_snapping_distance=intercept_snapping_distance,
        max_angle=max_angle,
        tangent=tangent,
        bspline=bspline,
    )

    new_endpoints_edges = find_facing_endpoints(
        vessel_graph,
        max_distance=if_none(arg.endpoints_to_endpoints_max_distance, arg.max_distance),
        max_angle=arg.max_angle,
        filter="exclusive",
        tangent=arg.tangent,
    )

    new_edges, new_nodes, new_nodes_yx = find_endpoints_branches_intercept(
        vessel_graph,
        max_distance=arg.max_distance,
        angle_tolerance=arg.max_angle,
        intercept_snapping_distance=arg.intercept_snapping_distance,
        tangent=arg.tangent,
        ignore_endpoints=new_endpoints_edges.flatten(),
    )

    vessel_graph = vessel_graph.copy() if not inplace else vessel_graph

    if len(new_nodes):
        branch, inv = np.unique(new_nodes[:, 1], return_inverse=True)
        lookup = np.arange(np.max(new_nodes[:, 0]) + 1)
        for i, b in enumerate(branch):
            nodes = new_nodes[inv == i, 0]
            split_yx = new_nodes_yx[inv == i]
            vessel_graph, new_nodes_id = vessel_graph.split_branch(b, split_yx, return_node_ids=True, inplace=True)
            lookup[nodes] = new_nodes_id
        new_edges[:, 1] = lookup[new_edges[:, 1]]

    if len(new_edges):
        vessel_graph.add_branches(new_edges, inplace=True)
    if len(new_endpoints_edges):
        vessel_graph.add_branches(new_endpoints_edges, inplace=True)

    return vessel_graph


_reconnect_endpoints = reconnect_endpoints
