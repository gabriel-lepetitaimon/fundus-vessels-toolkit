import itertools
from logging import warning
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import skimage.segmentation as sk_seg

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
        return assign_av_label(
            graph,
            av_map=av_map,
            ratio_threshold=ratio_threshold,
            split_av_branch=split_av_branch,
            av_attr=av_attr,
            propagate_labels=propagate_labels,
            inplace=inplace,
        )

    if av_map is None:
        try:
            av_map = graph.geometric_data().fundus_data.av
        except AttributeError:
            raise ValueError("The AV map is not provided and cannot be found in the geometric data.") from None
    elif isinstance(av_map, FundusData):
        av_map = av_map.av

    branches_seg_map = sk_seg.expand_labels(graph.branches_label_map(), 10) * (av_map > 0)
    graph.branches_attr[av_attr] = AVLabel.UNK

    # === Label branches as arteries, veins or unknown (both) ===
    for branch in graph.branches():
        branch_curve = branch.curve()
        node_to_node_length = branch.node_to_node_length()
        if len(branch_curve) <= max(3, node_to_node_length * 0.35):
            continue

        branch_seg = branches_seg_map == branch.id + 1

        # 1. Assign artery or vein label if its the label of at least ratio_threshold of the branch pixels
        #    (excluding background pixels)
        _, n_art, n_vei, n_both, n_unk = np.bincount(av_map[branch_seg], minlength=5)[:5]
        n_threshold = (n_art + n_vei + n_both + n_unk) * ratio_threshold
        if n_art > n_threshold:
            branch.attr[av_attr] = AVLabel.ART
        elif n_vei > n_threshold:
            branch.attr[av_attr] = AVLabel.VEI

        # 2. Attempt to split the branch into artery and veins sections
        elif split_av_branch and len(branch_curve) > 30:
            av_splits = extract_splits(av_map[branch_curve[:, 0], branch_curve[:, 1]], medfilt_size=9)
            if len(av_splits) > 1:
                splits = [int(_[1]) for _ in list(av_splits.keys())[:-1]]
                _, new_ids = graph.split_branch(branch.id, split_curve_id=splits, inplace=True, return_branch_ids=True)
                for new_id, new_value in zip(new_ids, av_splits.values(), strict=True):
                    graph.branches_attr[av_attr, new_id] = new_value
            else:
                branch.attr[av_attr] = AVLabel.BOTH

        # 3. Otherwise, label the branch as both arteries and veins
        else:
            branch.attr[av_attr] = AVLabel.BOTH

    propagated = True
    graph.nodes_attr[av_attr] = AVLabel.UNK
    while propagated:
        # === Propagate the labels from branches to nodes ===
        propagated = False
        for node in graph.nodes():
            if node.attr[av_attr] != AVLabel.UNK:
                continue
            # List labels of the incident branches of the node
            n = node.degree
            ibranches_av_count = np.bincount([ibranch.attr[av_attr] for ibranch in node.branches()], minlength=5)[1:5]
            n_art, n_vei, n_both, n_unk = ibranches_av_count

            # 1. If all branches are arteries (resp. veins) color the node as artery (resp. vein)
            if n_art == n:
                node.attr[av_attr] = AVLabel.ART
            elif n_vei == n:
                node.attr[av_attr] = AVLabel.VEI
            # 2. If some branches are arteries and some are veins or if any is both, color the node as both
            elif n_both > 0 or (n_art > 0 and n_vei > 0):
                node.attr[av_attr] = AVLabel.BOTH

            # 3. If the node connect exactly two branches and one is unknown, propagate the label of the other branch
            elif n == 2 and n_unk == 1:
                if n_art > 0:
                    node.attr[av_attr] = AVLabel.ART
                elif n_vei > 0:
                    node.attr[av_attr] = AVLabel.VEI
                else:
                    node.attr[av_attr] = AVLabel.BOTH
            # 4. Otherwise, keep the node as unknown
            else:
                continue

            # In the case 1, 2, and 3 the label of the node has been propagated
            propagated = True

        # === If necessary, propagate labels back from nodes to branches ===
        if not propagate_labels or not propagated or AVLabel.UNK not in graph.branches_attr[av_attr]:
            break

        propagated = False
        for branch in graph.branches():
            if branch.attr[av_attr] != AVLabel.UNK:
                continue

            branch_nodes_av = list(np.unique([node.attr[av_attr] for node in branch.nodes()]))

            # If both nodes of the branch are artery (resp. veins or both), propagate the label to the branch
            if branch_nodes_av == [AVLabel.ART]:
                branch.attr[av_attr] = AVLabel.ART
            elif branch_nodes_av == [AVLabel.VEI]:
                branch.attr[av_attr] = AVLabel.VEI
            elif branch_nodes_av in ([AVLabel.ART, AVLabel.VEI], [AVLabel.BOTH]):
                branch.attr[av_attr] = AVLabel.BOTH
            else:
                continue

            propagated = True

    return graph


def simplify_av_graph(
    graph: VGraph,
    av_attr="av",
    inplace=False,
):
    if not inplace:
        graph = graph.copy()
        return simplify_av_graph(graph, av_attr, inplace=True)

    geodata = graph.geometric_data()

    # === Remove geometry of branches with unknown type ===
    geodata.clear_branches_gdata(
        [
            b.id
            for b in graph.branches()
            if (b.attr[av_attr] == AVLabel.UNK and b.arc_length() < 30) or b.attr[av_attr] == AVLabel.BOTH
        ]
    )

    # === Remove passing nodes of same type ===
    simplify_passing_nodes(graph, min_angle=100, with_same_label=av_attr, inplace=True)

    # === Merge all small branches that are connected to two nodes of unknown type ===
    branch_to_fuse = []
    for branch in graph.branches():
        if (
            branch.arc_length() > 30
            and branch.attr["av"] == AVLabel.UNK
            and all(node.attr["av"] == AVLabel.UNK for node in branch.nodes())
        ):
            branch_to_fuse.append(branch)

    graph = graph.merge_nodes([branch.nodes_id for branch in branch_to_fuse], inplace=inplace)
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
        warning.warn("The graph contains self loop branches. They will be ignored.")
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
    od_yx: Point,
    macula_yx: Point | None = None,
    img_half_width: int = 512,
    *,
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

        The second array is a boolean array of shape (n_edges, 2) storing for each branch of the edge list, if it's connected through its first (True) or second (False) node.

        The third array of shape (n_edges,) stores, for each directed edge, the probabilities of the parent branch being the parent of the child branch.
    """  # noqa: E501
    from .graph_simplification import find_endpoints_branches_intercept, find_facing_endpoints

    edge_list = []
    edge_first_tip = []
    edge_probs = []

    geodata = graph.geometric_data()
    if macula_yx is not None:
        od_mac_dist = od_yx.distance(macula_yx)
    else:
        od_mac_dist = img_half_width
        macula_yx = od_yx + Point(0, img_half_width) if od_yx.x < img_half_width else od_yx - Point(0, img_half_width)

    # === Discover virtual branch to reconnect end nodes to adjacent branches or to other end nodes ===
    virtual_endp_edges = find_facing_endpoints(graph, max_distance=100, max_angle=30, filter="closest")
    virtual_edges, new_nodes, new_nodes_yx = find_endpoints_branches_intercept(
        graph, max_distance=100, intercept_snapping_distance=20, omit_endpoints_to_endpoints=True
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

    # === Duplicate all branch with BOTH AV label ===
    branches_av = graph.branches_attr[av_attr]
    both_branches = branches_av[branches_av == AVLabel.BOTH].index.to_numpy()
    _, new_branches = graph.duplicate_branches(both_branches, return_branch_id=True, inplace=True)
    graph.branches_attr[av_attr].iloc[both_branches] = AVLabel.ART
    graph.branches_attr[av_attr].iloc[new_branches] = AVLabel.VEI

    nodes_yx = geodata.nodes_coord()

    # === Utility function to connect all branches of a node to the tree root ===
    def add_root_edge(node: VGraphNode, tips_tangent=None):
        node_yx = Point.from_array(nodes_yx[node.id])
        od_dist = od_yx.distance(node_yx)
        p_dist = 1 - od_dist / od_mac_dist
        tangents = -node.tips_tangent(geodata) if tips_tangent is None else -tips_tangent
        for t, b in zip(tangents, node.branches(), strict=True):
            # Expected tangent: perpendicular to the macula ...
            expected_t = (macula_yx - node_yx).normalized().rot90()
            if expected_t.dot(node_yx - od_yx) < 0:
                expected_t = -expected_t  # ... and pointing outward from the optic disc
            p_tan = t.dot(expected_t)

            edge_list.append([-1, b.id])
            edge_first_tip.append([False, node.branches_first_node[0]])
            edge_probs.append(p_tan + p_dist)

    # === Utility function to compute the probabilities of a branch being the parent of another  ===
    def branch_to_branch_p(tip_tangents, mean_calibres, av_labels, od_dist, offset=0) -> float:
        # === Compute the probability that a branch is the parent of another branch ===
        # 1. ... based on the angle between the tangents of the tips of the branches
        p_tan = tip_tangents[0].dot(-tip_tangents[1])

        # 2. ... based on the similarity of the AV labels of the branches
        if np.any(av_labels == AVLabel.UNK):
            p_av = 0
        else:
            p_av = 1 if av_labels[0] == av_labels[1] else -1

        # 3. ... based on the difference of the calibres of the branches
        if np.any(np.isnan(mean_calibres)) or np.any(mean_calibres == 0):
            p_calibre = 0
        else:
            p_calibre = (mean_calibres[0] - mean_calibres[1]) / np.max(mean_calibres)

        common_p = p_tan + p_av * (1 - sigmoid(od_dist / 2 * od_mac_dist)) + offset
        return common_p + p_calibre, common_p - p_calibre

    def fetch_tangents_calibre_av_labels(branch_ids, first_tip):
        tip_tangents = geodata.tips_tangent(branch_ids, first_tip)
        calibres = geodata.branch_data(VBranchGeoData.Fields.CALIBRES, branch_ids)
        calibres = np.array([np.nanmean(c.data) if c is not None and len(c.data) else np.nan for c in calibres])
        av_labels = np.array(graph.branches_attr[av_attr][branch_ids], dtype=int)
        return tip_tangents, calibres, av_labels

    def add_edge_pair(branch12, first_tip12, p12, p21):
        edge_list.append(branch12)
        edge_first_tip.append(first_tip12)
        edge_probs.append(p12)

        edge_list.append(branch12[::-1])
        edge_first_tip.append(first_tip12[::-1])
        edge_probs.append(p21)

    # === Create the directed line graph from the graph ===
    for node in graph.nodes():
        node_degree = node.degree
        if node_degree == 1:
            # === LEAF NODES ===
            add_root_edge(node)

        else:
            # === PASSING / BRANCHING / JUNCTION NODES ===
            if node_degree == 2:
                # If passing node but with branches forming an small angle
                tip_tangents = node.tips_tangent(geodata)
                if np.dot(tip_tangents[0], tip_tangents[1]) >= np.cos(np.deg2rad(passing_as_root_max_angle)):
                    add_root_edge(node, tip_tangents)

            od_dist = od_yx.distance(nodes_yx[node.id])

            # For each pairs of branches, compute the probability of the first being the parent of the second
            branch_ids = np.array(node.branches_ids)
            first_tip = np.array(node.branches_first_node)
            tip_tangents, mean_calibres, branch_av = fetch_tangents_calibre_av_labels(branch_ids, first_tip)

            for b1, b2 in itertools.combinations(range(len(branch_ids)), 2):
                b12 = [b1, b2]
                p12, p21 = branch_to_branch_p(tip_tangents[b12], mean_calibres[b12], branch_av[b12], od_dist)
                add_edge_pair(branch_ids[b12], first_tip[b12], p12, p21)

    # === Add virtual edges connecting endpoints ===
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
    endp_branch = np.array(endp_branch, dtype=int)
    endp_first_tips = np.array(endp_first_tips, dtype=bool)
    virtual_endp_edges = np.array(vendp_edges, dtype=int)

    tips_tangents, calibres, av_labels = fetch_tangents_calibre_av_labels(endp_branch, endp_first_tips)
    endp_branch = endp_branch.reshape(-1, 2)
    endp_first_tips = endp_first_tips.reshape(-1, 2)
    tips_tangents = tips_tangents.reshape(-1, 2, 2)
    calibres = calibres.reshape(-1, 2)
    av_labels = av_labels.reshape(-1, 2)

    for e in range(len(endp_branch)):
        od_dist = np.mean(od_yx.distance(nodes_yx[virtual_endp_edges[e]]))
        dist_between_endp = np.linalg.norm(np.diff(nodes_yx[virtual_endp_edges[e]]))
        p_offset = 1 - dist_between_endp / 40
        p12, p21 = branch_to_branch_p(tips_tangents[e], calibres[e], av_labels[e], od_dist, offset=p_offset)
        add_edge_pair(endp_branch[e], endp_first_tips[e], p12, p21)

    # === Add virtual edges connecting branches to end points ===
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
        p_offset = 1 - np.linalg.norm(np.diff(nodes_yx[[endpoint_node.id, node.id]], axis=0)) / 40

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
                add_edge_pair(branch_id12, first_tip12, p12, p21)

    return graph, np.array(edge_list), np.array(edge_first_tip), np.array(edge_probs)


def resolve_digraph_to_vtree(
    vgraph: VGraph,
    line_list: npt.NDArray[np.int_],
    line_tips: npt.NDArray[np.bool_],
    line_probability: npt.NDArray[np.float64],
) -> VTree:
    import networkx as nx
    from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

    digraph = nx.DiGraph()
    for line, p, tips in zip(line_list, line_probability, line_tips, strict=True):
        digraph.add_edge(*line, p=p, tip=tips[1])

    optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)
    set_branches = np.zeros(vgraph.branches_count, dtype=bool)
    branch_tree = np.empty(vgraph.branches_count, int)
    branch_dirs = np.empty(vgraph.branches_count, bool)
    for b1, b2, info in optimal_tree.edges(data=True):
        branch_tree[b2] = b1
        branch_dirs[b2] = info["tip"]
        set_branches[b2] = True

    assert np.all(set_branches), "Some branches were not added to the tree."

    vtree = VTree.from_graph(vgraph, branch_tree, branch_dirs, copy=False)
    # reorder_nodes
    new_order = np.array([n.id for n in vtree.walk_nodes(depth_first=False)], dtype=int)
    _, idx = np.unique(new_order, return_index=True)  # Take the first occurrence of each node
    vtree.reindex_nodes(new_order[np.sort(idx)], inverse_lookup=True)

    # reorder_branches
    # new_order = [b.id for b in vtree.walk_branches(depth_first=False)]
    # vtree.reindex_branches(new_order, inverse_lookup=True)

    return vtree
