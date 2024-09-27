from logging import warning
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import skimage.segmentation as sk_seg

from ..utils.geometric import Point
from ..utils.math import extract_splits
from ..vascular_data_objects import AVLabel, FundusData, VBranchGeoData, VGraph, VTree
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
                graph.branches_attr[av_attr, new_ids] = list(av_splits.values())
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
            and all(node.attr["av"] == AVLabel.UNK for node in branch.nodes)
        ):
            branch_to_fuse.append(branch)

    graph = graph.merge_nodes([branch.nodes_id for branch in branch_to_fuse], inplace=inplace)
    return graph


def av_split(graph: VGraph, *, av_attr: str = "av") -> Tuple[VGraph, VGraph]:
    n_attr = graph.nodes_attr
    a_graph = graph.delete_nodes(n_attr.index[n_attr[av_attr] == AVLabel.VEI].to_numpy(), inplace=False)
    v_graph = graph.delete_nodes(n_attr.index[n_attr[av_attr] == AVLabel.ART].to_numpy(), inplace=False)

    a_graph.delete_branches([b.id for b in a_graph.branches() if b.attr[av_attr] == AVLabel.VEI], inplace=True)
    # a_graph.delete_branches(
    #    [b.id for b in a_graph.branches(only_terminal=True) if b.attr["av"] == AVLabel.UNK], inplace=True
    # )

    v_graph.delete_branches([b.id for b in v_graph.branches() if b.attr["av"] == AVLabel.ART], inplace=True)
    # v_graph.delete_branches(
    #    [b.id for b in v_graph.branches(only_terminal=True) if b.attr["av"] == AVLabel.UNK], inplace=True
    # )

    # if simplify:

    #     def fuse_passing_nodes(graph: VGraph, artery: bool):
    #         own_label = 1 if artery else 2
    #         nodes_to_fuse = []
    #         for n in graph.nodes():
    #             if n.degree == 2 and all(b.attr["av"] in (0, own_label) for b in n.branches):

    #                 nodes_to_fuse.append(n.id)
    #                 for b in n.branches:
    #                     b.attr["av"] = own_label
    #         graph.fuse_nodes(nodes_to_fuse, inplace=True)

    #     fuse_passing_nodes(a_graph, artery=True)
    #     fuse_passing_nodes(v_graph, artery=False)
    return a_graph, v_graph


def vgraph_to_vtree(
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
                print("Cycle detected")
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

            # 3'. Otherwise, list all incident branches of the node and remove the ancestors
            successors_id_dirs = list_adjacent_branches(node)
            successors_id_dirs = successors_id_dirs[~np.isin(successors_id_dirs[:, ID], ancestors)]
            successors = successors_id_dirs[:, ID]
            succ_dirs = successors_id_dirs[:, DIR]
            acst_dirs = branch_dirs[ancestors]

            # 4'. For each successor, determine the best ancestor base on branch direction
            adjacent_branches = np.concatenate([ancestors, successors])
            adjacent_dirs = np.concatenate([~acst_dirs, succ_dirs])
            adjacent_nodes = branch_list[adjacent_branches][:, 1 - adjacent_dirs]
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
            for succ, acst in zip(successors, ancestors[best_ancestor], strict=True):
                branch_tree[succ] = acst
                branch_dirs[succ] = succ_dirs[succ]
                stack.append(succ)
                visited_branches[succ] = True

    assert np.all(visited_branches), "Some branches were not added to the tree."

    # === Reorder the branches ===

    vtree = VTree.from_graph(graph, branch_tree, branch_dirs, copy=False)

    if reorder_nodes:
        new_order = np.array([n.id for n in vtree.walk_nodes(depth_first=False)], dtype=int)
        _, idx = np.unique(new_order, return_index=True)  # Take the first occurrence of each node
        vtree.reindex_nodes(new_order[np.sort(idx)], inverse_lookup=True)

    if reorder_branches:
        new_order = [b.id for b in vtree.walk_branches(depth_first=False)]
        vtree.reindex_branches(new_order, inverse_lookup=True)

    return vtree


def clean_vtree(vtree: VTree, *, av_attr: str = "av") -> VTree:
    # === Remove terminal with unknown type ===
    while to_delete := [b.id for b in vtree.branches() if not b.has_successors and b.attr[av_attr] == AVLabel.UNK]:
        vtree.delete_branches(to_delete, inplace=True)

    # === Remove passing nodes ===
    indegrees = vtree.node_indegree()
    outdegrees = vtree.node_outdegree()
    passing_nodes = np.argwhere((indegrees == 1) & (outdegrees == 1)).flatten()
    vtree.fuse_nodes(passing_nodes, quiet_invalid_node=False, inplace=True)

    return vtree
