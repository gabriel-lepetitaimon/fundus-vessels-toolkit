import itertools
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..utils.cluster import cluster_by_distance, reduce_clusters
from ..utils.geometric import Point
from ..utils.math import extract_splits, sigmoid
from ..vascular_data_objects import AVLabel, FundusData, VBranchGeoData, VGraph, VGraphNode, VTree
from .graph_simplification import simplify_passing_nodes


def assign_av_label(
    graph: VGraph,
    av_map: Optional[npt.NDArray[np.int_] | FundusData] = None,
    *,
    ratio_threshold: float = 4 / 5,
    split_av_branch=True,
    av_attr="av",
    propagate_labels=True,
    inplace=False,
):
    if not inplace:
        graph = graph.copy()

    if av_map is None:
        try:
            av_map = graph.geometric_data().fundus_data.av
        except AttributeError:
            raise ValueError("The AV map is not provided and cannot be found in the geometric data.") from None
    elif isinstance(av_map, FundusData):
        av_map = av_map.av

    gdata = graph.geometric_data()

    # === Assign the AV label to each branch based on the AV map ===
    graph.branches_attr[av_attr] = AVLabel.UNK
    branches_av_attr = graph.branches_attr[av_attr]
    for branch in graph.branches():
        branch_curve = branch.curve()
        # node_to_node_length = branch.node_to_node_length()
        if len(branch_curve) <= 2:
            continue

        # 0. Check the AV labels under each pixel of the skeleton and boundaries of the branch
        branch_skltn_av = av_map[branch_curve[:, 0], branch_curve[:, 1]]
        bound = branch.geodata(VBranchGeoData.Fields.BOUNDARIES, gdata).data
        branch_bound_av = av_map[bound[:, :, 0], bound[:, :, 1]]
        branch_av = np.concatenate([branch_skltn_av[:, None], branch_bound_av], axis=1)
        skel_is_art = np.any(branch_av == AVLabel.ART, axis=1)
        skel_is_vei = np.any(branch_av == AVLabel.VEI, axis=1)
        skel_is_both = np.any(branch_av == AVLabel.BOTH, axis=1)
        branch_av = np.full(len(branch_curve), AVLabel.UNK, dtype=int)
        branch_av[skel_is_art] = AVLabel.ART
        branch_av[skel_is_vei] = AVLabel.VEI
        branch_av[skel_is_both | (skel_is_art & skel_is_vei)] = AVLabel.BOTH

        # 1. Assign artery or vein label if its the label of at least ratio_threshold of the branch pixels
        #    (excluding background pixels)
        _, n_art, n_vei, n_both, n_unk = np.bincount(branch_av, minlength=5)[:5]
        n_threshold = (n_art + n_vei + n_both + n_unk) * ratio_threshold
        if n_art > n_threshold:
            branches_av_attr[branch.id] = AVLabel.ART
        elif n_vei > n_threshold:
            branches_av_attr[branch.id] = AVLabel.VEI

        # 2. Attempt to split the branch into artery and veins sections
        elif split_av_branch and len(branch_curve) > 30:
            av_splits = extract_splits(branch_av, medfilt_size=9)
            if len(av_splits) > 1:
                splits = [int(_[1]) for _ in list(av_splits.keys())[:-1]]
                _, new_ids = graph.split_branch(branch.id, split_curve_id=splits, inplace=True, return_branch_ids=True)
                for new_id, new_value in zip(new_ids, av_splits.values(), strict=True):
                    branches_av_attr[new_id] = new_value
            else:
                branches_av_attr[branch.id] = AVLabel.BOTH

        # 3. Otherwise, label the branch as both arteries and veins
        else:
            branches_av_attr[branch.id] = AVLabel.BOTH
    graph.branches_attr[av_attr] = branches_av_attr  # Why is this line necessary?

    # === Assign AV labels to nodes and propagate them through unknown passing nodes ===
    propagate_av_labels(graph=graph, av_attr=av_attr, only_label_nodes=not propagate_labels, inplace=True)

    return graph


def propagate_av_labels(
    graph: VGraph, av_attr="av", *, only_label_nodes=False, passing_node_min_angle: float = 110, inplace=False
):
    if not inplace:
        graph = graph.copy()

    gdata = graph.geometric_data()
    graph.nodes_attr[av_attr] = AVLabel.UNK
    nodes_av_attr = graph.nodes_attr[av_attr]
    branches_av_attr = graph.branches_attr[av_attr]

    propagated = True
    while propagated:
        propagated = False
        for node in graph.nodes(nodes_av_attr == AVLabel.UNK):
            # List labels of the incident branches of the node
            n = node.degree
            ibranches_av_count = np.bincount(branches_av_attr[node.branches_ids], minlength=5)[1:5]
            n_art, n_vei, n_both, n_unk = ibranches_av_count

            # 1. If all branches are arteries (resp. veins) color the node as artery (resp. vein)
            if n_art == n:
                nodes_av_attr[node.id] = AVLabel.ART
            elif n_vei == n:
                nodes_av_attr[node.id] = AVLabel.VEI
            # 2. If some branches are arteries and some are veins or if any is both, color the node as both
            elif n_both > 0 or (n_art > 0 and n_vei > 0):
                nodes_av_attr[node.id] = AVLabel.BOTH

            # 3. If the node connect exactly two branches and one is unknown,
            #     propagate the label of the known branch to the node and the other branch
            elif not only_label_nodes and n == 2 and n_unk == 1:
                if passing_node_min_angle > 0:
                    # Check if the branches are forming an small angle
                    tips_tangents = node.tips_tangent(gdata)
                    if np.dot(tips_tangents[0], tips_tangents[1]) >= np.cos(np.deg2rad(passing_node_min_angle)):
                        continue
                if n_art > 0:
                    nodes_av_attr[node.id] = AVLabel.ART
                    branches_av_attr[node.branches_ids] = AVLabel.ART
                elif n_vei > 0:
                    nodes_av_attr[node.id] = AVLabel.VEI
                    branches_av_attr[node.branches_ids] = AVLabel.VEI
                else:
                    nodes_av_attr[node.id] = AVLabel.BOTH
                    branches_av_attr[node.branches_ids] = AVLabel.BOTH
            # 4. Otherwise, keep the node as unknown
            else:
                continue

            # In the case 1, 2, and 3 the label of the node has been propagated
            propagated = True

        # If the propagation is disabled, return the after labelling the nodes
        if only_label_nodes:
            return graph

    # === Relabels branches connected to two nodes labelled BOTH, as BOTH ===
    for branch in graph.branches():
        if all(nodes_av_attr[list(branch.nodes_id)] == AVLabel.BOTH) and branch.node_to_node_length() < 30:
            branches_av_attr[branch.id] = AVLabel.BOTH

    # === Propagate AV labels to clusters of unknown branches ===
    # Find clusters of unknown branches
    unk_branches = graph.as_branches_ids(graph.branches_attr[av_attr] == AVLabel.UNK)
    unk_clusters = []
    solo_unk = []
    incoming_av = {}
    for branch in graph.branches(unk_branches):
        solo = True
        for adj in branch.adj_branches_ids():
            if adj in unk_branches:
                if adj > branch.id:
                    unk_clusters.append([branch.id, adj])
                solo = False
            else:
                incoming_av.setdefault(branch.id, []).append(graph.branches_attr[av_attr][adj])
        if solo:
            solo_unk.append([branch.id])
    unk_clusters = reduce_clusters(unk_clusters)

    # Attempt to label them based on the label of their incident branches and nodes
    for cluster in unk_clusters + solo_unk:
        # Fetch labels of the exterior nodes of the cluster
        cluster_nodes = np.unique(graph.branch_list[cluster])
        ext_nodes = cluster_nodes[np.isin(cluster_nodes, np.delete(graph.branch_list, cluster).flatten())]
        ext_nodes_av = nodes_av_attr[ext_nodes]
        n_art, n_vei, n_both, n_unk = np.bincount(ext_nodes_av, minlength=5)[1:5]
        n = len(ext_nodes_av)

        # Fetch labels of the incident branches of the cluster
        cluster_av = sum((incoming_av.get(b, []) for b in cluster), [])
        b_art, b_vei, b_both, b_unk = np.bincount(cluster_av, minlength=5)[1:5]
        b = len(cluster_av)

        cluster_label = AVLabel.UNK
        # 1. If all exterior nodes or incident  branches are arteries (resp. veins):
        #       => color the cluster as artery (resp. vein)
        if n_art == n or b_art == b:
            cluster_label = AVLabel.ART
        elif n_vei == n or b_vei == b:
            cluster_label = AVLabel.VEI
        # 2. If all exterior nodes are both: color the cluster as both
        elif n_both == n:
            cluster_label = AVLabel.BOTH

        # Assign the label to all branches and nodes of the cluster
        branches_av_attr[cluster] = cluster_label
        nodes_av_attr[cluster_nodes] = cluster_label

    return graph


def simplify_av_graph(
    graph: VGraph,
    av_attr="av",
    *,
    node_merge_distance: float = 15,
    unknown_node_merge_distance: float = 25,
    orphan_branch_min_length: float = 60,
    passing_node_min_angle: float = 110,
    propagate_labels=True,
    inplace=False,
):
    if not inplace:
        graph = graph.copy()

    # === Remove passing nodes of same type (pre-clustering) ===
    simplify_passing_nodes(graph, min_angle=passing_node_min_angle, with_same_label=av_attr, inplace=True)

    # === Remove small orphan branches ===
    graph.delete_branches(
        [b.id for b in graph.branches(filter="orphan") if b.node_to_node_length() < orphan_branch_min_length],
        inplace=True,
    )

    geodata = graph.geometric_data()
    nodes_av_attr = graph.nodes_attr[av_attr]

    # === Merge all small branches that are connected to two nodes of the same type ===
    nodes_clusters = []
    unknown_nodes_clusters = []
    max_merge_distance = max(node_merge_distance, unknown_node_merge_distance)
    for branch in graph.branches(filter="non-endpoint"):
        if branch.node_to_node_length() < max_merge_distance:
            n1, n2 = nodes_av_attr[list(branch.nodes_id)]
            # For this step, we consider branches with both type as unknown
            n1 = AVLabel.UNK if n1 == AVLabel.BOTH else n1
            n2 = AVLabel.UNK if n2 == AVLabel.BOTH else n2
            # If the nodes are of the same type, we add the branch to the corresponding cluster
            if n1 == n2:
                if n1 == AVLabel.UNK:
                    unknown_nodes_clusters.append(branch.nodes_id)
                else:
                    nodes_clusters.append(branch.nodes_id)
    if len(nodes_clusters):
        nodes_clusters = cluster_by_distance(geodata.nodes_coord(), node_merge_distance, nodes_clusters, iterative=True)
    if len(unknown_nodes_clusters):
        unknown_nodes_clusters = cluster_by_distance(
            geodata.nodes_coord(), unknown_node_merge_distance, unknown_nodes_clusters, iterative=True
        )
    if len(nodes_clusters) or len(unknown_nodes_clusters):
        graph.merge_nodes(nodes_clusters + unknown_nodes_clusters, inplace=True, assume_reduced=True)

    twin_branches = []
    for twins in graph.twin_branches():
        b0 = graph.branch(twins[0])
        if b0.node_to_node_length() < unknown_node_merge_distance:
            b0.attr[av_attr] = AVLabel.UNK
            nodes_av_attr[b0.nodes_id] = AVLabel.UNK
            twin_branches.extend(twins[1:])
    graph.delete_branches(twin_branches, inplace=True)

    # === Remove passing nodes of same type (post-clustering) ===
    simplify_passing_nodes(graph, min_angle=passing_node_min_angle, with_same_label=av_attr, inplace=True)

    # === Relabel unknown branches ===
    if propagate_labels:
        propagate_av_labels(graph, av_attr=av_attr, inplace=True)

    # === Remove geometry of branches with both type ===
    geodata.clear_branches_gdata(graph.as_branches_ids(graph.branches_attr[av_attr] == AVLabel.BOTH))

    return graph


def naive_av_split(graph: VGraph, *, av_attr: str = "av") -> Tuple[VGraph, VGraph]:
    n_attr = graph.nodes_attr
    a_graph = graph.delete_nodes(n_attr.index[n_attr[av_attr] == AVLabel.VEI].to_numpy(), inplace=False)
    v_graph = graph.delete_nodes(n_attr.index[n_attr[av_attr] == AVLabel.ART].to_numpy(), inplace=False)

    b_attr = a_graph.branches_attr
    a_graph.delete_branches(b_attr.index[b_attr[av_attr] == AVLabel.VEI].to_numpy(), inplace=True)
    b_attr = v_graph.branches_attr
    v_graph.delete_branches(b_attr.index[b_attr[av_attr] == AVLabel.ART].to_numpy(), inplace=True)

    return a_graph, v_graph


def naive_vgraph_to_vtree(
    graph: VGraph, root_pos: Point, reorder_nodes: bool = False, reorder_branches: bool = False
) -> VTree:
    # === Prepare graph ===
    graph = graph.copy()
    loop_branches = graph.self_loop_branches()
    if len(loop_branches) > 0:
        warnings.warn("The graph contains self loop branches. They will be ignored.", stacklevel=1)
        graph.delete_branches(loop_branches, inplace=True)

    nodes_coord = graph.nodes_coord()
    branch_list = graph.branch_list

    # === Prepare result variables ===
    branch_tree = -np.ones((graph.branches_count,), dtype=int)
    branch_dirs = np.zeros(len(graph.branch_list), dtype=bool)
    visited_branches = np.zeros(graph.branches_count, dtype=bool)

    # === Utilities method ===
    def list_adjacent_branches(node: int) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
        branches = np.argwhere(np.any(branch_list == node, axis=1)).flatten()
        return np.stack([branches, np.where(branch_list[branches, 0] == node, 1, 0)]).T

    ID, DIR = 0, 1

    def list_direct_successors(branch: int) -> Tuple[np.ndarray, np.ndarray]:
        head_node = branch_list[branch, 1 if branch_dirs[branch] else 0]
        branches_id_dirs = list_adjacent_branches(head_node)
        branches_id_dirs = branches_id_dirs[branches_id_dirs[:, ID] != branch]
        return branches_id_dirs

    stack = []

    def affiliate(branch: int, successors: np.ndarray):
        succ_ids, succ_dirs = successors.T
        branch_tree[succ_ids] = branch
        branch_dirs[succ_ids] = succ_dirs
        visited_branches[succ_ids] = True
        stack.extend(succ_ids)

    # === Find the root node of each sub tree ===
    roots = {}
    for nodes in graph.nodes_connected_graphs():
        nodes_dist = np.linalg.norm(nodes_coord[nodes] - root_pos, axis=1)
        min_node_id = np.argmin(nodes_dist)
        roots[nodes[min_node_id]] = nodes_dist[min_node_id]

    for root in sorted(roots, key=roots.get):
        root_branches_dirs = list_adjacent_branches(root)
        for root_branch, root_dir in root_branches_dirs:
            if not visited_branches[root_branch]:
                stack.append(root_branch)
                visited_branches[root_branch] = True
                branch_dirs[root_branch] = root_dir

    # === Walk the branches list of each sub tree ===
    delayed_stack: Dict[int, List[int]] = {}  # {node: [branch, ...]}
    while stack or delayed_stack:
        if stack:
            branch = stack.pop(0)

            # 1. List the children of the first branch on the stack
            successors = list_direct_successors(branch)
            successors_ids = successors[:, ID]

            # 2. If the branch has more than 2 successors, its evaluation is delayed until
            #    all other branch are visited, to resolve potential cycles.
            if len(successors_ids) > 2:
                head_node = branch_list[branch, 1 if branch_dirs[branch] else 0]
                if head_node not in delayed_stack:
                    delayed_stack[head_node] = [branch]
                else:
                    delayed_stack[head_node].append(branch)
                continue

            # 3. Otherwise, check if any of the children has already been visited
            if np.any(visited_branches[successors_ids]):
                successors = successors[~visited_branches[successors_ids]]

            # 4. Remember the hierarchy of the branches and add the children to the stack
            affiliate(branch, successors)

        else:
            # 1'. If the stack is empty, evaluate a delayed nodes
            node, ancestors = (k := next(iter(delayed_stack)), delayed_stack.pop(k))

            if len(ancestors) == 1:
                # 2'. If the node has only one ancestor, process it as a normal branch
                branch = ancestors[0]
                successors = list_direct_successors(branch)
                affiliate(branch, successors)
                continue
            ancestors = np.array(ancestors, dtype=int)

            # 3'. Otherwise, list all incident branches of the node and remove the ancestors
            successors_id_dirs = list_adjacent_branches(node)
            successors_id_dirs = successors_id_dirs[~np.isin(successors_id_dirs[:, ID], ancestors)]
            successors = successors_id_dirs[:, ID]
            succ_dirs = successors_id_dirs[:, DIR]
            acst_dirs = branch_dirs[ancestors]

            # 4'. For each successor, determine the best ancestor base on branch direction
            adjacent_branches = np.concatenate([ancestors, successors])
            adjacent_dirs = np.concatenate([~acst_dirs, succ_dirs])
            adjacent_nodes = branch_list[adjacent_branches][np.arange(len(adjacent_branches)), adjacent_dirs]
            tangents = graph.geometric_data().tip_data(
                VBranchGeoData.Fields.TIPS_TANGENT, adjacent_branches, first_tip=adjacent_dirs
            )
            for i, t in enumerate(tangents):  # If the tangent is not available, use the nodes coordinates
                if np.isnan(t).any() or np.sum(t) == 0:
                    tangents[i] = Point.from_array(nodes_coord[adjacent_nodes[i]] - nodes_coord[node]).normalized()
            acst_tangents = -tangents[: len(ancestors)]
            succ_tangents = tangents[len(ancestors) :]
            cos_angles = np.sum(succ_tangents[:, None, :] * acst_tangents[None, :, :], axis=-1)
            best_ancestor = np.argmax(cos_angles, axis=1)
            for succ, succ_dir, acst in zip(successors, succ_dirs, ancestors[best_ancestor], strict=True):
                branch_tree[succ] = acst
                branch_dirs[succ] = succ_dir
                stack.append(succ)
                visited_branches[succ] = True

    assert np.all(visited_branches), "Some branches were not added to the tree."

    # === Build vtree ===
    vtree = VTree.from_graph(graph, branch_tree, branch_dirs, copy=False)

    if reorder_nodes:
        new_order = np.array([n.id for n in vtree.walk_nodes(depth_first=False)], dtype=int)
        _, idx = np.unique(new_order, return_index=True)  # Take the first occurrence of each node
        vtree.reindex_nodes(new_order[np.sort(idx)], inverse_lookup=True)

    if reorder_branches:
        new_order = [b.id for b in vtree.walk_branches(depth_first=False)]
        vtree.reindex_branches(new_order, inverse_lookup=True)

    return vtree


def build_line_digraph(
    graph: VGraph,
    fundus_data: FundusData,
    img_half_width: int = 512,
    *,
    split_twins=True,
    av_attr: str = "av",
    passing_as_root_max_angle=30,
) -> Tuple[VGraph, npt.NDArray[np.int_], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """Build a directed line graph storing the probabilities that a branch is the parent of another branch.

    Parameters
    ----------
    graph : VGraph
        _description_

    Returns
    -------
    Tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_], npt.NDArray[np.float64]]
        Three arrays describing the directed line graph.

        The first array is its edge list of shape (n_edges, 2) where each row store the index of the parent branch, the index of the child branches and the index of the node connecting them.

        The second array is is 0 or 1 array of shape (n_edges, 2) storing for each edge, the index of the nodes through which the branches are connected.

        The third array of shape (n_edges,) stores, for each directed edge, the probabilities of the parent branch being the parent of the child branch.
    """  # noqa: E501
    from .geometry_parsing import derive_tips_geometry_from_curve_geometry
    from .graph_simplification import find_endpoints_branches_intercept, find_facing_endpoints

    edge_list = []
    edge_first_tip = []
    edge_probs = []

    macula_yx = fundus_data.macula_center if fundus_data.has_macula else fundus_data.infered_macula_center()
    od_yx = fundus_data.od_center
    od_mac_dist = od_yx.distance(macula_yx)
    od_diameter = fundus_data.od_diameter

    if split_twins:
        # === Split self-loop branches in 3 and twin branches in two to differentiate their tips ===
        for b in graph.self_loop_branches():
            graph.split_branch(b, split_curve_id=[0.3, 0.6], inplace=True)
        for twins in graph.twin_branches():
            for b in twins:
                graph.split_branch(b, split_curve_id=0.5, inplace=True)

    # === Discover virtual branch to reconnect end nodes to adjacent branches or to other end nodes ===
    virtual_endp_edges = find_facing_endpoints(graph, max_distance=100, max_angle=30, filter="closest")
    virtual_edges, new_nodes, new_nodes_yx = find_endpoints_branches_intercept(
        graph, max_distance=100, intercept_snapping_distance=20, ignore_endpoints=virtual_endp_edges.flatten()
    )

    # Insert any new nodes required by the virtual edges between end points and branches
    if len(new_nodes):
        branch, inv = np.unique(new_nodes[:, 1], return_inverse=True)
        lookup = np.arange(np.max(new_nodes[:, 0]) + 1)
        for e, b in enumerate(branch):
            nodes = new_nodes[inv == e, 0]
            split_yx = new_nodes_yx[inv == e]
            _, new_nodes_id = graph.split_branch(b, split_yx, return_node_ids=True, inplace=True)
            lookup[nodes] = new_nodes_id
        virtual_edges[:, 1] = lookup[virtual_edges[:, 1]]

    # === Refresh the tip geometric data ===
    graph = derive_tips_geometry_from_curve_geometry(graph, inplace=True)
    geodata = graph.geometric_data()

    # === Duplicate all branch with BOTH or UNKNOWN AV label ===
    branches_av = graph.branches_attr[av_attr]
    both_branches = branches_av[branches_av.isin([AVLabel.BOTH, AVLabel.UNK])].index.to_numpy()
    _, new_branches = graph.duplicate_branches(both_branches, return_branch_id=True, inplace=True)
    graph.branches_attr[av_attr].iloc[both_branches] = AVLabel.ART
    graph.branches_attr[av_attr].iloc[new_branches] = AVLabel.VEI

    graph.flip_branches_direction(graph.branch_list[:, 0] > graph.branch_list[:, 1])

    nodes_yx = geodata.nodes_coord()

    out_pole_yx = od_yx.numpy()[None, :]
    in_pole_yx = ((macula_yx - od_yx) * 1.5 + od_yx).numpy()[None, :]

    def expected_tangent_sim(yx, t):
        u_out = yx - out_pole_yx
        norm_u_out = np.linalg.norm(u_out, axis=1)
        out_influence = (1 / norm_u_out**2)[:, None]
        u_out /= norm_u_out[:, None]

        u_in = in_pole_yx - yx
        norm_u_in = np.linalg.norm(u_in, axis=1)
        in_influence = (1 / norm_u_in**2)[:, None]
        u_in /= norm_u_in[:, None]

        expected_t = (u_out * out_influence + u_in * in_influence) / (out_influence + in_influence)
        return (t * expected_t).sum(axis=1)

    # === Compute the probability associated with the orientation of each branch ===
    branches_dir_p = np.zeros((graph.branches_count, 2), dtype=float)
    branches_calibres = geodata.branch_data(VBranchGeoData.Fields.CALIBRES)
    branches_tangents = geodata.branch_data(VBranchGeoData.Fields.TANGENTS)
    branches_curves = geodata.branch_curve()
    for branch in graph.branches():
        branch_id = branch.id

        # 1. ... based on its calibre variation
        p_calibre = 0
        calibre = branches_calibres[branch_id]
        if calibre is not None and len(calibre.data) > 40:
            n = len(calibre.data)
            calibre = np.nanmean(calibre.data[: n // 2]) - np.nanmean(calibre.data[n // 2 :])
            p_calibre = sigmoid(calibre / 3, antisymmetric=True)

        # 2. ... based on its tangents
        p_tan = 0
        if branches_tangents[branch_id] is not None:
            p_tan = expected_tangent_sim(branches_curves[branch_id], branches_tangents[branch_id].data).mean()

        p_dir = p_tan * 0.3 + p_calibre * 1
        branches_dir_p[branch_id] = p_dir, -p_dir

    # === Utility function to compute the edge probabilities  ===
    def root_p(node_yx, tip_tangent):
        # === Compute the probability that a branch is a root branch ===
        # 1. ... based on the distance to the optic disc
        od_dist = od_yx.distance(node_yx)
        p_dist = (od_dist_offset := 0) - sigmoid(od_dist / od_mac_dist, antisymmetric=True)

        # 2. ... based on the angle between the tangent of the tip and the expected tangent
        # Expected tangent: away from the disc, toward the macula (similar to magnetic field lines)
        p_tan = float(expected_tangent_sim(node_yx.numpy(), tip_tangent))

        return p_tan + p_dist

    def branch_to_branch_p(tip_tangents, mean_calibres, av_labels, od_dist, offset=0) -> float:
        # === Compute the probability that a branch is the parent of another branch ===
        # 1. ... based on the angle between the tangents of the tips of the branches
        p_tan = tip_tangents[0].dot(-tip_tangents[1])

        # 2. ... based on the similarity of the AV labels of the branches
        if np.any(av_labels == AVLabel.UNK):
            p_av = 0
        else:
            p_av = 1 if av_labels[0] == av_labels[1] else -1
        p_av *= 1 - sigmoid(od_dist / (2 * od_mac_dist))

        # 3. ... based on the difference of the calibres of the branches
        if np.any(np.isnan(mean_calibres)) or np.any(mean_calibres == 0):
            p_calibre = 0
        else:
            p_calibre = sigmoid((mean_calibres[0] - mean_calibres[1]) / 3, antisymmetric=True) / 2

        common_p = p_tan + p_av + offset
        return common_p + p_calibre, common_p - p_calibre

    def fetch_tangents_calibre_av_labels(branch_ids, first_tip):
        tip_tangents = geodata.tips_tangent(branch_ids, first_tip)
        calibres = geodata.branch_data(VBranchGeoData.Fields.CALIBRES, branch_ids)
        calibres = np.array([np.nanmean(c.data) if c is not None and len(c.data) else np.nan for c in calibres])
        av_labels = np.array(graph.branches_attr[av_attr][branch_ids], dtype=int)
        return tip_tangents, calibres, av_labels

    # === Utility function to add edge to the directed line graph ===
    def add_edge_pair(branch12, tip_id12, p12, p21):
        edge_list.append(branch12)
        edge_first_tip.append(tip_id12)
        edge_probs.append(p12)

        edge_list.append(branch12[::-1])
        edge_first_tip.append(tip_id12[::-1])
        edge_probs.append(p21)

    def add_root_edge(node: VGraphNode, tips_tangent=None):
        node_yx = Point.from_array(nodes_yx[node.id])
        tangents = node.tips_tangent(geodata) if tips_tangent is None else tips_tangent
        tips = node.branches_first_node
        for t, b, first in zip(tangents, node.branches(), tips, strict=True):
            edge_list.append([-1, b.id])
            edge_first_tip.append([0, 0 if first else 1])
            edge_probs.append(root_p(node_yx, t))

    # === Find closest nodes to the optic disc for each connected component ===
    root_nodes = np.zeros(graph.nodes_count, dtype=bool)
    for nodes in graph.nodes_connected_graphs():
        root_nodes[nodes[np.argmin(od_yx.distance(nodes_yx[nodes]))]] = True

    # === Create the directed line graph from the graph ===
    # 1. ORPHAN BRANCHES
    # for branch in graph.branches(filter="orphan"):
    #     n0_yx, n1_yx = nodes_yx[branch.nodes_id]
    #     tips_tangents = geodata.tips_tangent(branch.id)
    #     p0 = root_p(Point.from_array(n0_yx), tips_tangents[0])
    #     p1 = root_p(Point.from_array(n1_yx), tips_tangents[1])
    #     edge_list.append([-1, branch.id])
    #     if p0 > p1:
    #         edge_first_tip.append([False, True])
    #         edge_probs.append(p0)
    #     else:
    #         edge_first_tip.append([False, False])
    #         edge_probs.append(p1)

    for node in graph.nodes():
        node_degree = node.degree
        od_dist = od_yx.distance(nodes_yx[node.id])
        if root_nodes[node.id] or node_degree == 1 or od_dist < od_diameter * 0.75:
            # 2. ROOT NODES
            # if not orphan_branches[node.branches_ids[0]]:  # Orphan branches were processed at the beginning
            add_root_edge(node)

        if node_degree > 1:
            # 3. PASSING / BRANCHING / JUNCTION NODES
            od_dist = od_yx.distance(nodes_yx[node.id])

            # For each pairs of branches, compute the probability of the first being the parent of the second
            branch_ids = np.array(node.branches_ids)
            first_tip = np.array(node.branches_first_node)
            tip_tangents, mean_calibres, branch_av = fetch_tangents_calibre_av_labels(branch_ids, first_tip)

            for b1, b2 in itertools.combinations(range(len(branch_ids)), 2):
                b12 = [b1, b2]
                p12, p21 = branch_to_branch_p(tip_tangents[b12], mean_calibres[b12], branch_av[b12], od_dist)
                tip_id12 = np.where(first_tip[b12], 0, 1)
                add_edge_pair(branch_ids[b12], tip_id12, p12, p21)

    # 4. VIRTUAL EDGES: CONNECTING ENDPOINTS TOGETHER
    b, t = graph.incident_branches_individual(virtual_endp_edges.flatten(), return_branch_direction=True)
    endp_branch = []
    endp_first_tips = []
    vendp_edges = []
    for e in range(0, len(virtual_endp_edges) * 2, 2):
        # Endpoints may have been duplicated because of a BOTH AV label. In this case, all combinations are considered.
        for i in range(len(b[e])):
            for j in range(len(b[e + 1])):
                endp_branch.extend([b[e][i], b[e + 1][j]])
                endp_first_tips.extend([t[e][i], t[e + 1][j]])
                vendp_edges.append(virtual_endp_edges[e // 2])
    if len(endp_branch) != 0:
        endp_branch = np.array(endp_branch, dtype=int)
        endp_first_tips = np.array(endp_first_tips, dtype=bool)
        virtual_endp_edges = np.array(vendp_edges, dtype=int)

        tips_tangents, branches_calibres, av_labels = fetch_tangents_calibre_av_labels(endp_branch, endp_first_tips)
        endp_branch = endp_branch.reshape(-1, 2)
        endp_first_tips = endp_first_tips.reshape(-1, 2)
        tips_tangents = tips_tangents.reshape(-1, 2, 2)
        branches_calibres = branches_calibres.reshape(-1, 2)
        av_labels = av_labels.reshape(-1, 2)

        for e in range(len(endp_branch)):
            od_dist = np.mean(od_yx.distance(nodes_yx[virtual_endp_edges[e]]))
            dist_between_endp = np.linalg.norm(np.diff(nodes_yx[virtual_endp_edges[e]], axis=0))
            p_offset = 1 - dist_between_endp / 40
            p12, p21 = branch_to_branch_p(
                tips_tangents[e], branches_calibres[e], av_labels[e], od_dist, offset=p_offset
            )
            tip_id12 = np.where(endp_first_tips[e], 0, 1)
            add_edge_pair(endp_branch[e], tip_id12, p12, p21)

    # 5. VIRTUAL EDGES: CONNECTING ENDPOINTS TO BRANCHES
    for e in range(virtual_edges.shape[0]):
        endpoint_node = graph.node(virtual_edges[e, 0])
        endp_branch_ids = endpoint_node.branches_ids
        endp_first_tips = endpoint_node.branches_first_node
        endp_tips_tangent, endp_mean_calibres, endp_branch_av = fetch_tangents_calibre_av_labels(
            endp_branch_ids, endp_first_tips
        )

        node = graph.node(virtual_edges[e, 1])
        node_branch_ids = node.branches_ids
        node_first_tips = node.branches_first_node
        node_tips_tangent, node_mean_calibres, node_branch_av = fetch_tangents_calibre_av_labels(
            node_branch_ids, node_first_tips
        )

        od_dist = np.mean(od_yx.distance(nodes_yx[node.id]))
        p_offset = -np.linalg.norm(np.diff(nodes_yx[[endpoint_node.id, node.id]], axis=0)) / 40

        for i in range(len(endp_branch_ids)):
            for j in range(len(node_branch_ids)):
                p12, p21 = branch_to_branch_p(
                    tip_tangents=np.array([endp_tips_tangent[i], node_tips_tangent[j]]),
                    mean_calibres=np.array([endp_mean_calibres[i], node_mean_calibres[j]]),
                    av_labels=np.array([endp_branch_av[i], node_branch_av[j]]),
                    od_dist=od_dist,
                    offset=p_offset,
                )
                branch_id12 = endp_branch_ids[i], node_branch_ids[j]
                first_tip12 = endp_first_tips[i], node_first_tips[j]
                tip_id12 = np.where(first_tip12, 0, 1)
                add_edge_pair(branch_id12, tip_id12, p12, p21)

    return graph, np.array(edge_list), np.array(edge_first_tip), np.array(edge_probs), branches_dir_p


def resolve_digraph_to_vtree(
    vgraph: VGraph,
    line_list: npt.NDArray[np.int_],
    line_tips: npt.NDArray[np.int_],
    line_probability: npt.NDArray[np.float64],
    branches_dir_p: npt.NDArray[np.float64],
    tricot: bool = True,
    debug_info: Optional[Dict[str, Any]] = None,
) -> VTree:
    import networkx as nx
    from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

    if not tricot:
        orphan_branch = vgraph.orphan_branches(as_mask=True)

        digraph = nx.DiGraph()
        for line, p, tips_id in zip(line_list, line_probability, line_tips, strict=True):
            if orphan_branch[line[1]] and (already_added := digraph.edges.get(line, None)) is not None:
                if already_added["p"] >= p:
                    continue
            digraph.add_edge(*line, p=p, tips=tips_id)

        optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)
        branch_tree = {}
        branch_dirs = {}
        for b_from, b_to, info in optimal_tree.edges(data=True):
            tips_id = info["tips"]
            n1 = vgraph.branch_list[b_from, tips_id[0]]
            n2 = vgraph.branch_list[b_to, tips_id[1]]
            if b_from == -1 or n1 == n2:
                branch_tree[b_to] = b_from
                branch_dirs[b_to] = tips_id[1] == 0
            else:
                vgraph, b_id = vgraph.add_branches([[n1, n2]], return_branch_id=True, inplace=True)
                b_id = b_id[0]
                branch_tree[b_id] = b_from
                branch_dirs[b_id] = True
                branch_tree[b_to] = b_id
                branch_dirs[b_to] = tips_id[1] == 0

        try:
            branch_tree = np.array([branch_tree[b] for b in range(vgraph.branches_count)], dtype=int)
            branch_dirs = np.array([branch_dirs[b] for b in range(vgraph.branches_count)], dtype=bool)
        except KeyError:
            raise ValueError("Some branches were not added to the tree.") from None
    else:
        from ..utils.cpp_extensions.fvt_cpp import maximum_weighted_independent_set

        # NEW ATTEMPT TO DIFFERENTIATE ROOT TIPS AND PREVENT BRANCHES FROM BEING BOTH PRIMARY AND SECONDARY
        # REVERSE TRICOT!!!!

        print("REVERSE TRICOT")

        digraph = nx.DiGraph()
        for id, (line, p, tips_id) in enumerate(zip(line_list, line_probability, line_tips, strict=True)):
            b_from, b_to = line + 1
            b_from_tip_id, b_to_tip_id = tips_id
            if b_from_tip_id == 0:
                b_from = -b_from  # Source: First node of the source branch => the source branch is reversed

            if b_to_tip_id == 1:
                b_to = -b_to  # Target: Second node of the destination branch => the dest branch is reversed

            dir_p = branches_dir_p[line[0], 0 if b_from_tip_id == 1 else 1]
            dir_p += branches_dir_p[line[1], 0 if b_to_tip_id == 0 else 1]
            digraph.add_edge(b_from, b_to, p=p + dir_p, id=id)

        # Solve the double optimal tree (using both direction for each branch)
        optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)

        if True:
            # Regenerate digraph using only the edges of the optimal tree
            kept_edges = np.zeros(len(line_list), dtype=bool)
            digraph = nx.DiGraph()
            for b0, b1, info in optimal_tree.edges(data=True):
                digraph.add_edge(abs(b0) - 1, abs(b1) - 1, p=info["p"], tips=[1 if b0 > 0 else 0, 0 if b1 > 0 else 1])
                kept_edges[info["id"]] = True

            optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)
            branch_tree = {}
            branch_dirs = {}
            for b_from, b_to, info in optimal_tree.edges(data=True):
                tips_id = info["tips"]
                n1 = vgraph.branch_list[b_from, tips_id[0]]
                n2 = vgraph.branch_list[b_to, tips_id[1]]
                if b_from == -1 or n1 == n2:
                    branch_tree[b_to] = b_from
                    branch_dirs[b_to] = tips_id[1] == 0
                else:
                    vgraph, b_id = vgraph.add_branches([[n1, n2]], return_branch_id=True, inplace=True)
                    b_id = b_id[0]
                    branch_tree[b_id] = b_from
                    branch_dirs[b_id] = True
                    branch_tree[b_to] = b_id
                    branch_dirs[b_to] = tips_id[1] == 0

            try:
                branch_tree = np.array([branch_tree[b] for b in range(vgraph.branches_count)], dtype=int)
                branch_dirs = np.array([branch_dirs[b] for b in range(vgraph.branches_count)], dtype=bool)
            except KeyError:
                raise ValueError("Some branches were not added to the tree.") from None

            if debug_info is not None:
                debug_info["kept_edges"] = kept_edges
        else:
            double_branch_tree = np.zeros((vgraph.branches_count, 2), int) - 2
            double_branch_tree_p = np.zeros((vgraph.branches_count, 2), float)
            for B_from, B_to, info in optimal_tree.edges(data=True):
                double_branch_tree[abs(B_to) - 1, 1 if B_to < 0 else 0] = B_from
                double_branch_tree_p[abs(B_to) - 1, 1 if B_to < 0 else 0] = info["p"]

            # Identify the subtrees
            root_branches = np.argwhere(double_branch_tree == 0)
            subtrees = []
            subtrees_dirs = []
            subtrees_p = []

            for root in root_branches:
                p = double_branch_tree_p[tuple(root)]
                subtree = [np.array([root[0]], dtype=int)]
                subtree_dirs = np.zeros((vgraph.branches_count, 2), bool)
                subtree_dirs[tuple(root)] = True

                root = root[0] + 1 if root[1] == 0 else -(root[0] + 1)
                active_branches = np.array([root], dtype=int)

                while active_branches.size > 0:
                    direct_branch = np.argwhere(np.isin(double_branch_tree[:, 0], active_branches)).flatten()
                    if direct_branch.size != 0:
                        assert not subtree_dirs[direct_branch].any(), "The optimal arborescence contains cycles."
                        subtree_dirs[direct_branch, 0] = True
                        p += double_branch_tree_p[direct_branch, 0].sum()
                        subtree.append(direct_branch)

                    inverse_branch = np.argwhere(np.isin(double_branch_tree[:, 1], active_branches)).flatten()
                    if inverse_branch.size != 0:
                        assert not subtree_dirs[inverse_branch].any(), "The optimal arborescence contains cycles."
                        subtree_dirs[inverse_branch, 1] = True
                        p += double_branch_tree_p[inverse_branch, 1].sum()
                        subtree.append(inverse_branch)

                    active_branches = np.concatenate([direct_branch + 1, -(inverse_branch + 1)])

                subtrees.append(np.concatenate(subtree))
                subtrees_dirs.append(subtree_dirs)
                subtrees_p.append(p)

            # Solve for the maximal weighted independent set of subtrees
            # (the set may not contains two trees sharing a branch, we search the forest providing the highest probability)
            trees_mutually_exclusive = []
            for s1, s2 in itertools.combinations(range(len(subtrees_dirs)), 2):
                if (subtrees_dirs[s1].any(axis=1) & subtrees_dirs[s2].any(axis=1)).any():
                    trees_mutually_exclusive.append((s1, s2))

            subtrees_p = np.array(subtrees_p, dtype=float)  # + np.array([s.sum() for s in subtrees_dirs]) * 10
            if subtrees_p.min() < 0:
                subtrees_p -= subtrees_p.min() - 1
            optimal_forest = maximum_weighted_independent_set(trees_mutually_exclusive, subtrees_p)

            # Reconstruct the tree from the optimal forest
            branch_tree = np.zeros(vgraph.branches_count, int) - 1
            branch_dirs = np.zeros(vgraph.branches_count, bool)
            for s in optimal_forest:
                subtree_branches = subtrees_dirs[s]
                b, dirs = np.where(subtree_branches)
                branch_tree[b] = abs(double_branch_tree[b, dirs]) - 1
                branch_dirs[b] = dirs == 0

            if debug_info is not None:
                debug_info["subtrees"] = subtrees_dirs
                debug_info["subtrees_p"] = subtrees_p
                debug_info["optimal_forest"] = optimal_forest

    vtree = VTree.from_graph(vgraph, branch_tree, branch_dirs, copy=False)
    return vtree


def inspect_digraph_solving(
    fundus_data: FundusData,
    graph: VGraph,
    *,
    av_attr: str = "av",
    passing_as_root_max_angle=30,
):
    import pandas as pd
    import panel as pn
    from bokeh.models.widgets.tables import BooleanFormatter, NumberFormatter
    from jppype import Mosaic

    pn.extension("tabulator")

    graph = graph.copy()

    # === Compute and solve the digraph ===
    graph, line_list, line_tips, line_probability, branches_dir_p = build_line_digraph(
        graph,
        fundus_data,
        av_attr=av_attr,
        passing_as_root_max_angle=passing_as_root_max_angle,
    )
    B = graph.branches_count
    debug_info = {}
    tree = resolve_digraph_to_vtree(
        graph.copy(), line_list, line_tips, line_probability, branches_dir_p, debug_info=debug_info
    )
    tree.branches_attr["subtree"] = tree.subtrees_branch_labels()

    # === Create the graph views ===
    GRAPH_AV_COLORS = {
        AVLabel.BKG: "grey",
        AVLabel.ART: "red",
        AVLabel.VEI: "blue",
        AVLabel.BOTH: "purple",
        AVLabel.UNK: "green",
    }

    def GENERIC_CMAP(x):
        from jppype.utils.color import colormap_by_name

        colormap = colormap_by_name()
        colormap = ["red", "blue", "purple", "green", "orange", "cyan", "pink", "yellow", "teal", "lime"]
        return colormap[x % len(colormap)]

    m = Mosaic(2)
    fundus_data.draw(view=m, labels_opacity=0.3)

    m[0]["graph"] = graph.jppype_layer(bspline=True, edge_labels=True, node_labels=True)
    m[0]["tangents"] = graph.geometric_data().jppype_branches_tips_tangents(scaling=2)
    m[0]["graph"].edges_cmap = graph.branches_attr["av"].map(GRAPH_AV_COLORS).to_dict()

    m[1]["graph"] = tree.jppype_layer(bspline=True, edge_labels=True)
    m[1]["tangents"] = tree.geometric_data().jppype_branches_tips_tangents(
        scaling=4, show_only="junctions", invert_direction="tree"
    )
    m[1]["graph"].edges_cmap = tree.branches_attr["subtree"].map(GENERIC_CMAP).to_dict()
    m[1]["graph"].edges_labels = tree.branches_attr["subtree"].to_dict()

    nodes_color = pd.Series("grey", index=tree.nodes_attr.index)
    nodes_color[tree.root_nodes_ids()] = "black"
    m[1]["graph"].nodes_cmap = nodes_color.to_dict()

    m.show()

    # === Create the graph views ===
    branch_anc = tree.branch_tree[line_list[:, 1]]
    b0, b1 = line_list.T
    tip0, tip1 = line_tips.T

    # line_subtrees = np.full(b0.shape, "", dtype=object)
    # for i, s in enumerate(debug_info["subtrees"]):
    #     line_subtrees[s[b0, tip0] & s[b1, tip1]] = str(i)

    lines_data = pd.DataFrame(
        {
            "b0": b0,
            "b1": b1,
            "n0": tree.branch_list[b0, tip0],
            "n1": tree.branch_list[b1, tip1],
            "tree1": tree.branches_attr["subtree"][b1].to_numpy(),
            "p": line_probability,
            "dir0_p": branches_dir_p[b0, np.where(tip0 == 1, 0, 1)],
            "dir1_p": branches_dir_p[b1, np.where(tip1 == 0, 0, 1)],
            "optimal": ((branch_anc == b0) | ((branch_anc > B) & (tree.branch_tree[branch_anc] == b0))),
            # "subtree": line_subtrees,
            "kept": debug_info.get("kept_edges", np.zeros(b0.shape, dtype=bool)),
        }
    )
    root_data = lines_data[lines_data["b0"] == -1]
    root_data = root_data.drop(columns=["b0", "n0"])
    non_root_data = lines_data[lines_data["b0"] != -1]

    def highlight_optimal(row):
        if row["optimal"]:
            return ["background-color: #fff8d7"] * len(row)
        elif row["kept"]:  # row["subtree"] != "":
            return ["background-color: #f8d7ff"] * len(row)
        else:
            return [""] * len(row)

    tab_opt = dict(
        show_index=False,
        disabled=True,
        layout="fit_data",
        height=400,
        align="center",
        pagination=None,
        header_filters=True,
        formatters={
            "p": NumberFormatter(format="0.000"),
            "dir0_p": NumberFormatter(format="0.000"),
            "dir1_p": NumberFormatter(format="0.000"),
            "optimal": BooleanFormatter(icon="check-circle"),
            "kept": BooleanFormatter(icon="check-circle"),
        },
    )
    root_table = pn.widgets.Tabulator(
        root_data,
        editors={k: None for k in root_data.columns},
        text_align={k: "center" for k in root_data.columns},
        **tab_opt,
    )
    root_table.style.apply(highlight_optimal, axis=1)
    non_root_table = pn.widgets.Tabulator(
        non_root_data,
        editors={k: None for k in non_root_data.columns},
        text_align={k: "center" for k in non_root_data.columns},
        **tab_opt,
    )
    non_root_table.style.apply(highlight_optimal, axis=1)
    non_root_table2 = pn.widgets.Tabulator(
        non_root_data,
        editors={k: None for k in non_root_data.columns},
        text_align={k: "center" for k in non_root_data.columns},
        **tab_opt,
    )
    non_root_table2.style.apply(highlight_optimal, axis=1)

    def focus_on_root(event):
        coord = tree.nodes_coord()[root_data["n1"].iloc[event.row]]
        m[0].goto(coord[::-1], scale=3)

    def focus_on_non_root(event):
        coord = tree.nodes_coord()[non_root_data["n1"].iloc[event.row]]
        m[0].goto(coord[::-1], scale=3)

    root_table.on_click(focus_on_root)
    non_root_table.on_click(focus_on_non_root)
    non_root_table2.on_click(focus_on_non_root)

    return (
        pn.Row(root_table, pn.Spacer(width=10), non_root_table, pn.Spacer(width=10), non_root_table2, height=400),
        graph,
        tree,
        lines_data,
    )
