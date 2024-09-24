from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import skimage.segmentation as sk_seg

from ..utils.geometric import Point
from ..utils.math import extract_splits
from ..vascular_data_objects import AVLabel, FundusData, VGraph, VTree
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
                graph.branches_attr[av_attr].iloc[new_ids] = list(av_splits.values())
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
    for branch in graph.branches():
        if branch.attr[av_attr] == AVLabel.UNK:
            geodata.clear_branches_gdata(branch.i)

    # === Remove passing nodes of same type ===
    simplify_passing_nodes(graph, min_angle=100, with_same_label=av_attr, inplace=True)

    # === Merge all small branches that are connected to two nodes of unknown type ===
    branch_to_fuse = []
    for branch in graph.branches():
        if (
            branch.chord_length() > 30
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


def vgraph_to_vtree(graph: VGraph, root_pos: Point, reorder_nodes: bool = False) -> VTree:
    nodes_coord = graph.nodes_coord()
    branch_list = graph.branch_list

    final_roots = []
    final_branch_list = np.empty_like(branch_list)
    final_n_branches = 0
    final_branch_reindex = -np.ones((len(graph.branch_list),), dtype=int)
    final_branch_flip = np.zeros(len(graph.branch_list), dtype=bool)

    branches_tree = -np.ones((graph.branches_count,), dtype=int)

    visited_nodes = np.zeros(graph.nodes_count, dtype=bool)

    def list_children(node: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find the children of a node in the graph and the branches to reach them.

        Returns
        -------
        Self loop branches:
            The indexes of the branches that loop back to the node as a 1d array.

        Children nodes:
            A 2d array of shape (N, 3) where N is the number of children. Each row contains the index of the branch
            to reach the child, the index of the child node and a boolean indicating if the branch is inverted.
        """
        branches_n0 = branch_list[:, 0] == node
        branches_n1 = branch_list[:, 1] == node
        loop_branches = np.argwhere(branches_n0 & branches_n1).flatten()
        ibranches = np.argwhere(np.logical_xor(branches_n0, branches_n1)).flatten()
        invert_ibranches = branches_n1[ibranches]
        children_nodes = branch_list[ibranches, 1 - invert_ibranches]
        return loop_branches, np.stack([ibranches, children_nodes, invert_ibranches], axis=1)

    # === Find the root node of each sub tree ===
    roots = {}
    for nodes in graph.nodes_connected_graphs():
        nodes_dist = np.linalg.norm(nodes_coord[nodes] - root_pos, axis=1)
        min_node_id = np.argmin(nodes_dist)
        roots[nodes[min_node_id]] = nodes_dist[min_node_id]

    final_roots = sorted(roots, key=roots.get)

    # === Walk the branches list of each sub tree ===
    for root in final_roots:
        stack = [root]
        delayed_stack: Dict[int, List[int]] = {}  # {node: [parent_node, ...]}
        while stack or delayed_stack:
            if stack:
                node = stack.pop()

                # 1. List the children of the first node on the stack
                loop_branches, children = list_children(node)

                # 2. Add the self loop branches to the final tree
                for loop_branch in loop_branches:
                    final_branch_list[final_n_branches] = node, node
                    final_branch_reindex[loop_branch] = final_n_branches
                    final_n_branches += 1

                # 3. If the node has more than 3 incident branches, its evaluation is delayed until
                #    all other nodes are visited, to resolve potential cycles.
                if len(children) > 3:
                    if node not in delayed_stack:
                        delayed_stack[node] = [node]
                    else:
                        delayed_stack[node].append(node)
                    continue

                # 4. Otherwise, remove the already visited children
                already_visited = visited_nodes[children[:, 1]]
                if np.sum(already_visited) > 1:
                    print("Cycle detected")
                children = children[~already_visited]
            else:
                # 1'. If the stack is empty, evaluate a delayed nodes
                node, roots = delayed_stack.popitem()
                _, children = list_children(node)
                already_visited = visited_nodes[children[:, 1]]
                if np.sum(already_visited) > len(roots):
                    print("Cycle detected")
                children = children[~already_visited]

                # TODO: Decide which node is the parent of the delayed node AND store it...
                # This require a modification of VTree to explicitly store the parent of each node.

            # 5. Add the children to the final tree
            for child_branch, child_node, inverted in children:
                final_branch_list[final_n_branches] = node, child_node
                final_branch_reindex[child_branch] = final_n_branches
                final_branch_flip[child_branch] = inverted
                final_n_branches += 1

                stack.append(child_node)
            visited_nodes[node] = True

    assert final_n_branches == graph.branches_count, "Some branches were not added to the tree."
    assert np.all(visited_nodes), "Some nodes were not visited during the tree traversal."

    # === Reorder the branches ===
    final_branch_list = np.array(final_branch_list)
    geometric_data = graph.geometric_data().copy()
    geometric_data._flip_branches_direction(np.argwhere(final_branch_flip).flatten())
    geometric_data._reindex_branches(final_branch_reindex)

    vtree = VTree(
        final_branch_list,
        branches_tree,
        None,
        geometric_data=geometric_data,
        nodes_attr=graph.nodes_attr,
        branches_attr=graph.branches_attr.set_index(final_branch_reindex),
        nodes_count=graph.nodes_count,
    )

    if reorder_nodes:
        new_order = []

        for root in vtree.root_nodes_id:
            new_order.append(root)
            new_order.extend(vtree.walk(root, depth_first=False))
        vtree.reindex_nodes(new_order, inverse_lookup=True)

    return vtree
