import itertools
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..utils.cluster import cluster_by_distance, reduce_clusters
from ..utils.geometric import Point
from ..utils.lookup_array import create_removal_lookup
from ..utils.math import extract_splits, quantized_higher, sigmoid
from ..vascular_data_objects import AVLabel, FundusData, VBranchGeoData, VGraph, VGraphNode, VTree
from .graph_simplification import simplify_passing_nodes


def assign_av_label(
    graph: VGraph,
    av_map: Optional[npt.NDArray[np.int_] | FundusData] = None,
    *,
    av_medfilt_size: int = 9,
    split_av_branch=True,
    split_high_curvature=0,
    split_av_threshold=2 / 3,
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
    # === Split branches with high curvature ===
    if split_high_curvature and gdata.has_branch_data(VBranchGeoData.Fields.CURVATURES):
        curvatures = gdata.branch_data(VBranchGeoData.Fields.CURVATURES)
        splits = []
        for b in range(graph.branch_count):
            if curvatures[b] is None or len(curvatures[b].data) < 10:
                continue
            curv = abs(curvatures[b].data)
            b_splits = quantized_higher(curv, split_high_curvature, medfilt_size=5)
            if len(b_splits) > 0:
                splits.append((b, b_splits))

        if len(splits) > 0:
            for b, b_splits in splits:
                graph.split_branch(b, split_curve_id=b_splits, inplace=True)

    # === Assign the AV label to each branch based on the AV map ===
    graph.branch_attr[av_attr] = AVLabel.UNK
    branches_av_attr = graph.branch_attr[av_attr]
    for branch in graph.branches():
        branch_curve = branch.curve()
        if len(branch_curve) <= 2:
            continue

        # 0. Check the AV labels under each pixel of the skeleton and boundaries of the branch
        bound = branch.geodata(VBranchGeoData.Fields.BOUNDARIES, gdata).data
        valid_bound = gdata.domain.contains(bound).all(axis=1)
        if not np.all(valid_bound):
            # warnings.warn(f"Branch {branch.id} has invalid boundary points. They will be ignored.", stacklevel=1)
            branch_curve = branch_curve[valid_bound]
            if len(branch_curve) < 2:
                continue
            bound = bound[valid_bound]

        branch_skltn_av = av_map[branch_curve[:, 0], branch_curve[:, 1]]
        branch_bound_av = av_map[bound[:, :, 0], bound[:, :, 1]]
        branch_av = np.concatenate([branch_skltn_av[:, None], branch_bound_av], axis=1)
        skel_is_art = np.any(branch_av == AVLabel.ART, axis=1)
        skel_is_vei = np.any(branch_av == AVLabel.VEI, axis=1)
        skel_is_both = np.any(branch_av == AVLabel.BOTH, axis=1)
        branch_av = np.full(len(branch_curve), AVLabel.UNK, dtype=int)
        branch_av[skel_is_art] = AVLabel.ART
        branch_av[skel_is_vei] = AVLabel.VEI
        branch_av[skel_is_both | (skel_is_art & skel_is_vei)] = AVLabel.BOTH

        if split_av_branch and len(branch_curve) > 30:
            # 1. Assign artery or vein label if its the label of at least ratio_threshold of the branch pixels
            #    (excluding background pixels)
            _, n_art, n_vei, n_both, n_unk = np.bincount(branch_av, minlength=5)[:5]
            n_threshold = (n_art + n_vei + n_both + n_unk) * split_av_threshold
            if n_art > n_threshold:
                branches_av_attr[branch.id] = AVLabel.ART
            elif n_vei > n_threshold:
                branches_av_attr[branch.id] = AVLabel.VEI

            # 2. Attempt to split the branch into artery and veins sections
            else:
                av_splits = extract_splits(branch_av, medfilt_size=av_medfilt_size)
                if len(av_splits) > 1:
                    splits = [int(_[1]) for _ in list(av_splits.keys())[:-1]]
                    _, new_ids = graph.split_branch(
                        branch.id, split_curve_id=splits, inplace=True, return_branch_ids=True
                    )
                    for new_id, new_value in zip(new_ids, av_splits.values(), strict=True):
                        branches_av_attr[new_id] = new_value
                else:
                    branches_av_attr[branch.id] = next(iter(av_splits.values()))
        else:
            _, n_art, n_vei, n_both, n_unk = np.bincount(branch_av, minlength=5)[:5]
            branches_av_attr[branch.id] = [AVLabel.ART, AVLabel.VEI, AVLabel.BOTH][np.argmax([n_art, n_vei, n_both])]

    graph.branch_attr[av_attr] = branches_av_attr  # Why is this line necessary?

    # === Assign AV labels to nodes and propagate them through unknown passing nodes ===
    propagate_av_labels(graph=graph, av_attr=av_attr, only_label_nodes=not propagate_labels, inplace=True)

    return graph


def propagate_av_labels(
    graph: VGraph, av_attr="av", *, only_label_nodes=False, passing_node_min_angle: float = 110, inplace=False
):
    if not inplace:
        graph = graph.copy()

    gdata = graph.geometric_data()
    graph.node_attr[av_attr] = AVLabel.UNK
    nodes_av_attr = graph.node_attr[av_attr]
    branches_av_attr = graph.branch_attr[av_attr]

    propagated = True
    while propagated:
        propagated = False
        for node in graph.nodes(nodes_av_attr == AVLabel.UNK):
            # List labels of the incident branches of the node
            n = node.degree
            ibranches_av_count = np.bincount(branches_av_attr[node.adjacent_branch_ids], minlength=5)[1:5]
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
                    branches_av_attr[node.adjacent_branch_ids] = AVLabel.ART
                elif n_vei > 0:
                    nodes_av_attr[node.id] = AVLabel.VEI
                    branches_av_attr[node.adjacent_branch_ids] = AVLabel.VEI
                else:
                    nodes_av_attr[node.id] = AVLabel.BOTH
                    branches_av_attr[node.adjacent_branch_ids] = AVLabel.BOTH
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
        if all(nodes_av_attr[list(branch.node_ids)] == AVLabel.BOTH) and branch.node_to_node_length() < 30:
            branches_av_attr[branch.id] = AVLabel.BOTH

    # === Propagate AV labels to clusters of unknown branches ===
    # Find clusters of unknown branches
    unk_branches = graph.as_branch_ids(graph.branch_attr[av_attr] == AVLabel.UNK)
    unk_clusters = []
    solo_unk = []
    incoming_av = {}
    for branch in graph.branches(unk_branches):
        solo = True
        for adj in branch.adjacent_branch_ids():
            if adj in unk_branches:
                if adj > branch.id:
                    unk_clusters.append([branch.id, adj])
                solo = False
            else:
                incoming_av.setdefault(branch.id, []).append(graph.branch_attr[av_attr][adj])
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
    orphan_branch_min_length: float = 20,
    passing_node_min_angle: float = 110,
    propagate_labels=True,
    inplace=False,
):
    if not inplace:
        graph = graph.copy()

    # === Fuse passing nodes of same type (pre-clustering) ===
    # simplify_passing_nodes(graph, min_angle=passing_node_min_angle, with_same_label=av_attr, inplace=True)

    # === Remove small orphan branches ===
    graph.delete_branch(
        [b.id for b in graph.branches(filter="orphan") if b.node_to_node_length() < orphan_branch_min_length],
        inplace=True,
    )

    geodata = graph.geometric_data()
    nodes_av_attr = graph.node_attr[av_attr]

    # === Merge nodes of the same type connected by a small branch ===
    nodes_clusters = []
    unknown_nodes_clusters = []
    max_merge_distance = max(node_merge_distance, unknown_node_merge_distance)
    for branch in graph.branches(filter="non-endpoint"):
        if branch.node_to_node_length() < max_merge_distance:
            n1, n2 = nodes_av_attr[list(branch.node_ids)]  # type: ignore
            # For this step, we consider branches with both type as unknown
            n1 = AVLabel.UNK if n1 == AVLabel.BOTH else n1
            n2 = AVLabel.UNK if n2 == AVLabel.BOTH else n2
            # If the nodes are of the same type, we add the branch to the corresponding cluster
            if n1 == n2:
                if n1 == AVLabel.UNK:
                    unknown_nodes_clusters.append(branch.node_ids)
                else:
                    nodes_clusters.append(branch.node_ids)

    if len(nodes_clusters):
        nodes_clusters = cluster_by_distance(geodata.node_coord(), node_merge_distance, nodes_clusters, iterative=True)
    if len(unknown_nodes_clusters):
        unknown_nodes_clusters = cluster_by_distance(
            geodata.node_coord(), unknown_node_merge_distance, unknown_nodes_clusters, iterative=True
        )
    if len(nodes_clusters) or len(unknown_nodes_clusters):
        graph.merge_nodes(nodes_clusters + unknown_nodes_clusters, inplace=True, assume_reduced=True)

    # === Keep only one branch for any group of small or undefined twin branches ===
    # TODO: Merge cycles?
    twin_branches = []
    for twins in graph.twin_branches():
        b0 = graph.branch(twins[0])
        if b0.node_to_node_length() < unknown_node_merge_distance or not geodata.has_branch_curve(twins).all():
            # Keep the first branch and remove the others
            twin_branches.extend(twins[1:])
            # Label the first branch as unknown and clear its geometry data
            b0.attr[av_attr] = AVLabel.UNK
            nodes_av_attr[b0.node_ids] = AVLabel.UNK
            geodata.clear_branch_gdata([b0.id])

    graph.delete_branch(twin_branches, inplace=True)

    # === Remove passing nodes of same type (post-clustering) ===
    simplify_passing_nodes(graph, min_angle=passing_node_min_angle, with_same_label=av_attr, inplace=True)

    # === Relabel unknown branches ===
    if propagate_labels:
        propagate_av_labels(graph, av_attr=av_attr, inplace=True)

    # === Remove geometry of branches with both type ===
    geodata.clear_branch_gdata(graph.as_branch_ids(graph.branch_attr[av_attr] == AVLabel.BOTH))

    return graph


def split_av_graph(
    graph: VGraph, *, av_attr: str = "av", simplify: bool = True, center_junction_nodes: bool = True
) -> Tuple[VGraph, VGraph]:
    b_attr = graph.branch_attr
    a_graph = graph.delete_branch(b_attr[av_attr] == AVLabel.VEI, inplace=False)
    v_graph = graph.delete_branch(b_attr[av_attr] == AVLabel.ART, inplace=False)

    if simplify:
        from .graph_simplification import simplify_passing_nodes

        def remove_unknown_leaf_branches(graph):
            unkown_branches = graph.as_branch_ids(graph.branch_attr[av_attr] == AVLabel.UNK)
            while (unkown_terminal_branches := np.intersect1d(unkown_branches, graph.leaf_branches_ids())).size:
                graph.delete_branch(unkown_terminal_branches, inplace=True)

        remove_unknown_leaf_branches(a_graph)
        remove_unknown_leaf_branches(v_graph)

        simplify_passing_nodes(a_graph, with_same_label=av_attr, inplace=True)
        simplify_passing_nodes(v_graph, with_same_label=av_attr, inplace=True)

    if center_junction_nodes:
        from .geometry_parsing import center_junction_nodes

        center_junction_nodes(a_graph, inplace=True)
        center_junction_nodes(v_graph, inplace=True)

    return a_graph, v_graph


def split_av_graph_by_subtree(
    tree: VTree,
    *,
    av_attr: str = "av",
    detect_major_error: bool = True,
    simplify: bool = True,
    center_junction_nodes: bool = True,
    inplace: bool = False,
) -> Tuple[VGraph, VGraph]:
    if not inplace:
        tree = tree.copy()

    tree.branch_attr.fillna({av_attr: AVLabel.UNK}, inplace=True)
    art_branches = tree.as_branch_ids(tree.branch_attr[av_attr] == AVLabel.ART)
    vei_branches = tree.as_branch_ids(tree.branch_attr[av_attr] == AVLabel.VEI)
    geodata = tree.geometric_data()

    total_calibres = [
        c.data[np.isfinite(c.data)].sum() if c is not None else 0
        for c in geodata.branch_data(VBranchGeoData.Fields.CALIBRES)
    ]
    total_calibres = np.array(total_calibres)

    def subtree_av_weight(subtree):
        # a_weight = np.sum(geodata.branch_arc_length(np.intersect1d(subtree, art_branches), fast_approximation=True))
        # v_weight = np.sum(geodata.branch_arc_length(np.intersect1d(subtree, vei_branches), fast_approximation=True))
        a_weight = total_calibres[np.intersect1d(subtree, art_branches)].sum()
        v_weight = total_calibres[np.intersect1d(subtree, vei_branches)].sum()
        return a_weight, v_weight

    # === Assign one artery or vein label to each subtree ===
    subtrees = tree.subtrees()
    subtrees_av = [False for _ in range(len(subtrees))]  # True: artery, False: vein
    for i, subtree in enumerate(list(subtrees)):
        if detect_major_error and len(
            (crossings := tree.crossing_nodes_ids(subtree, return_branch_ids=True, only_traversing=False))[0]
        ):
            processed_branch = set()
            for _, branches in zip(*crossings, strict=True):
                art_b = [b for b in branches.keys() if b in art_branches and b not in processed_branch]
                vei_b = [b for b in branches.keys() if b in vei_branches and b not in processed_branch]
                if art_b and vei_b:
                    # Backtrack each branch until the label change or the root of the subtree is found
                    partial_root = []
                    for incoming_branch, art in [(_, True) for _ in art_b] + [(_, False) for _ in vei_b]:
                        b_label = AVLabel.ART if art else AVLabel.VEI
                        b = tree.branch(incoming_branch)
                        while (b_anc := b.ancestor()) is not None and b_anc.attr[av_attr] == b_label:
                            b = b_anc
                        partial_root.append(b.id)

                    # The subtree with the farther root is kept, the subtrees with opposite labels will be detached
                    farther_root_i = np.argmin(tree.branch_distance_to_root(partial_root))
                    n = len(art_b)
                    art_root = farther_root_i < n
                    outliers_b = partial_root[n:] if art_root else partial_root[:n]

                    for b in tree.branches(outliers_b):
                        outlier_subtree = np.concatenate([[b.id], tree.branch_successors(b.id, max_depth=None)])
                        a_w, v_w = subtree_av_weight(outlier_subtree)
                        if a_w < v_w if art_root else a_w > v_w:
                            # Detach subtree from the main tree
                            tree._branch_tree[b.id] = -1
                            subtree = np.setdiff1d(subtree, outlier_subtree)
                            subtrees[i] = subtree
                            subtrees.append(outlier_subtree)
                            processed_branch |= set(outlier_subtree)
                            subtrees_av.append(not art_root)

        a_weight, v_weight = subtree_av_weight(subtree)
        if a_weight >= v_weight:
            subtrees_av[i] = True
        else:
            subtrees_av[i] = False

    # === Split the tree into two trees based on the AV label of the subtrees ===
    subtrees_av = np.array(subtrees_av, dtype=bool)
    if any(subtrees_av):
        v_branches = np.concatenate([subtrees for i, subtrees in enumerate(subtrees) if not subtrees_av[i]])
        a_tree = tree.delete_branch(v_branches, inplace=False)
    else:
        a_tree = VTree.empty_like(tree)
    if any(~subtrees_av):
        a_branches = np.concatenate([subtrees for i, subtrees in enumerate(subtrees) if subtrees_av[i]])
        v_tree = tree.delete_branch(a_branches, inplace=False)
    else:
        v_tree = VTree.empty_like(tree)

    # === Simplify the trees ===
    if simplify:
        from .graph_simplification import simplify_passing_nodes
        # from .tree_simplification import disconnect_crossing_nodes

        def remove_unknown_leaf_branches(graph):
            unkown_branches = graph.as_branch_ids(graph.branch_attr[av_attr] == AVLabel.UNK)
            while (unkown_terminal_branches := np.intersect1d(unkown_branches, graph.leaf_branches_ids())).size:
                del_lookup = create_removal_lookup(unkown_terminal_branches, length=graph.branch_count)
                graph.delete_branch(unkown_terminal_branches, inplace=True)
                unkown_branches = np.setdiff1d(del_lookup[unkown_branches], [-1])

        remove_unknown_leaf_branches(a_tree)
        remove_unknown_leaf_branches(v_tree)

        simplify_passing_nodes(a_tree, min_angle=110, inplace=True)
        simplify_passing_nodes(v_tree, min_angle=110, inplace=True)

        # disconnect_crossing_nodes(a_tree, inplace=True)
        # disconnect_crossing_nodes(v_tree, inplace=True)

    # === Center junction nodes ===
    if center_junction_nodes:
        from .geometry_parsing import center_junction_nodes

        center_junction_nodes(a_tree, inplace=True)
        center_junction_nodes(v_tree, inplace=True)

    return a_tree, v_tree


def relabel_av_by_subtree(tree: VTree, *, av_attr: str = "av", inplace: bool = False) -> VTree:
    """Relabel the branches of a tree based on the most common label of its subtree.

    Parameters
    ----------
    tree : VTree
        Tree to relabel.

    av_attr : str, optional
        Name of the attribute storing the AV labels.

    inplace : bool, optional
        If True, the tree is modified in place.

    Returns
    -------
    VTree
        The relabeled tree.
    """
    if not inplace:
        tree = tree.copy()

    tree.branch_attr[av_attr].fillna(AVLabel.UNK, inplace=True)
    art_branches = tree.as_branch_ids(tree.branch_attr[av_attr] == AVLabel.ART)
    vei_branches = tree.as_branch_ids(tree.branch_attr[av_attr] == AVLabel.VEI)
    geodata = tree.geometric_data()

    for root in tree.root_branches_ids():
        subtree = np.concatenate([[root], tree.branch_successors(root, max_depth=None)])
        subtree_a = np.intersect1d(subtree, art_branches)
        subtree_v = np.intersect1d(subtree, vei_branches)
        a_weight = np.sum(geodata.branch_arc_length(subtree_a, fast_approximation=True))
        v_weight = np.sum(geodata.branch_arc_length(subtree_v, fast_approximation=True))

        if a_weight >= v_weight:
            tree.branch_attr.loc[subtree_v, av_attr] = AVLabel.ART
        else:
            tree.branch_attr.loc[subtree_a, av_attr] = AVLabel.VEI

    return tree


def naive_vgraph_to_vtree(
    graph: VGraph, root_pos: Point, reorder_nodes: bool = False, reorder_branches: bool = False
) -> VTree:
    # === Prepare graph ===
    graph = graph.copy()
    loop_branches = graph.self_loop_branches()
    if len(loop_branches) > 0:
        warnings.warn("The graph contains self loop branches. They will be ignored.", stacklevel=1)
        graph.delete_branch(loop_branches, inplace=True)

    nodes_coord = graph.node_coord()
    branch_list = graph.branch_list

    # === Prepare result variables ===
    branch_tree = -np.ones((graph.branch_count,), dtype=int)
    branch_dirs = np.zeros(len(graph.branch_list), dtype=bool)
    visited_branches = np.zeros(graph.branch_count, dtype=bool)

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
    for nodes in graph.node_connected_components():
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
        new_order = np.array([n.id for n in vtree.walk_nodes(traversal="dfs")], dtype=int)
        _, idx = np.unique(new_order, return_index=True)  # Take the first occurrence of each node
        vtree.reindex_nodes(new_order[np.sort(idx)], inverse_lookup=True, inplace=True)

    if reorder_branches:
        new_order = [b.id for b in vtree.walk_branches(traversal="dfs")]
        vtree.reindex_branches(new_order, inverse_lookup=True)

    return vtree


@dataclass
class LineDigraphOpt:
    #: Weighting of the tips distance penalty in the probability of branch to branch connection
    b2b_tips_dist_penalty_w: float = 1 / 80

    #: Minimum tips distance under which no penalty is applied to the probability of branch to branch connection
    b2b_tips_dist_penalty_min: float = 5

    #: Weighting of the calibre difference in the probability of branch to branch connection
    #: ()
    b2b_p_calibre_w: float = 0.3

    #: Weighting of the tangents angle in the probability of branch to branch connection
    b2b_p_tangent_w: float = 0.8

    #: Offset added to the root candidate probability
    root_p_offset: float = 1.5

    #: Any nodes closer than this distance to the center of the optic disc will automatically be considered
    #: as a root candidates
    #: (The distance unit is OD diameter)
    root_candidates_od_distance: float = 1

    #: Any nodes closer than this distance to the border of the image will automatically be considered
    #: as a root candidates
    #: (The distance is in pixel)
    root_candidates_border_dist: float = 10

    #: The distance threshold to consider two branches as connected.
    # reconnect_dist_penalty_coef: float = 80


def build_line_digraph(
    graph: VGraph,
    fundus_data: FundusData,
    *,
    opt: Optional[LineDigraphOpt] = None,
    split_twins=False,
    av_attr: str = "av",
    inplace: bool = False,
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

    if not inplace:
        graph = graph.copy()
    if opt is None:
        opt = LineDigraphOpt()

    edge_list = []
    edge_first_tip = []
    edge_probs = []

    macula_yx = fundus_data.infered_macula_center()
    od_yx = fundus_data.od_center
    assert od_yx is not None, "The optic disc center is not defined."

    od_mac_dist = od_yx.distance(macula_yx) if macula_yx is not None else fundus_data.shape[0] / 2
    od_diameter = fundus_data.od_diameter

    center_yx = Point(*fundus_data.shape) / 2
    fundus_radius = fundus_data.shape[0] / 2

    if split_twins:
        # === Split self-loop branches in 3 and twin branches in two to differentiate their tips ===
        for b in graph.self_loop_branches():
            graph.split_branch(b, split_curve_id=[0.3, 0.6], inplace=True)
        for twins in graph.twin_branches():
            for b in twins:
                graph.split_branch(b, split_curve_id=0.5, inplace=True)

    # === Discover virtual branch to reconnect end nodes to adjacent branches or to other end nodes ===
    virtual_endp_edges = find_facing_endpoints(graph, max_distance=100, max_angle=30)
    virtual_edges, new_nodes, new_nodes_yx = find_endpoints_branches_intercept(
        graph, max_distance=100, intercept_snapping_distance=30, omit_endpoints_to_endpoints=True
    )

    # Insert any new nodes required by the virtual edges between end points and branches
    if len(new_nodes):
        branch, inv = np.unique(new_nodes[:, 1], return_inverse=True)
        lookup = np.arange(np.max(new_nodes[:, 0]) + 1)
        for e, b in enumerate(branch):
            nodes = new_nodes[inv == e, 0]
            split_yx = new_nodes_yx[inv == e]
            split_curve_id = graph.geometric_data().branch_closest_index(split_yx, b)
            nodes_order = np.argsort(split_curve_id)
            split_yx, split_curve_id = split_yx[nodes_order], split_curve_id[nodes_order]
            _, new_nodes_id = graph.split_branch(b, split_curve_id, split_yx, return_node_ids=True, inplace=True)
            lookup[nodes[nodes_order]] = new_nodes_id
        virtual_edges[:, 1] = lookup[virtual_edges[:, 1]]

    # === Refresh the tip geometric data ===
    graph = derive_tips_geometry_from_curve_geometry(graph, inplace=True)
    geodata = graph.geometric_data()

    # === Duplicate all branch with BOTH or UNKNOWN AV label ===
    av_col_id = graph.branch_attr.columns.get_loc(av_attr)
    branches_av = graph.branch_attr[av_attr]
    both_branches = branches_av[branches_av.isin([AVLabel.BOTH, AVLabel.UNK])].index.to_numpy()
    _, new_branches = graph.duplicate_branch(both_branches, return_branch_id=True, inplace=True)
    both_branches = np.concatenate([both_branches, new_branches])
    graph.branch_attr.iloc[both_branches, av_col_id] = AVLabel.ART  # type: ignore
    graph.branch_attr.iloc[new_branches, av_col_id] = AVLabel.VEI  # type: ignore

    # graph.flip_branch_direction(graph.branch_list[:, 0] > graph.branch_list[:, 1]) # For debugging

    nodes_yx = geodata.node_coord()

    out_pole_yx = od_yx.numpy()[None, :]
    if macula_yx is not None:
        in_pole_yx = ((macula_yx - od_yx) * 2 + od_yx).numpy()[None, :]
        uncertain_yx = macula_yx.numpy()[None, :]
        uncertain_v = (macula_yx - od_yx).normalized().numpy()[None, :]
        UNCERTAIN_COS = 0.9659  # cos(15°)

        def expected_tangent_sim(yx, t):
            u_out = yx - out_pole_yx
            norm_u_out = np.linalg.norm(u_out, axis=1) + 1e-3
            out_influence = (1 / norm_u_out**2)[:, None]
            u_out /= norm_u_out[:, None]

            u_in = in_pole_yx - yx
            norm_u_in = np.linalg.norm(u_in, axis=1) + 1e-3
            in_influence = (1 / norm_u_in**2)[:, None]
            u_in /= norm_u_in[:, None]

            u_uncertain = yx - uncertain_yx
            u_uncertain /= np.linalg.norm(u_uncertain, axis=1)[:, None] + 1e-3
            uncertain_p = np.clip(1 - (u_uncertain * uncertain_v).sum(axis=1), 0, 1 - UNCERTAIN_COS) / (
                1 - UNCERTAIN_COS
            )

            expected_t = (u_out * out_influence + u_in * in_influence) / (out_influence + in_influence)
            return (t * expected_t).sum(axis=1) * uncertain_p
    else:

        def expected_tangent_sim(yx, t):
            expected_t = yx - out_pole_yx
            expected_t /= np.linalg.norm(expected_t, axis=1)[:, None] + 1e-3
            return (t * expected_t).sum(axis=1)

    # === Compute the probability associated with the orientation of each branch ===
    branches_dir_p = np.zeros((graph.branch_count, 2), dtype=float)
    branches_calibres: List[VBranchGeoData.Curve] = geodata.branch_data(VBranchGeoData.Fields.CALIBRES)  # type: ignore
    branches_tangents: List[VBranchGeoData.Tangents] = geodata.branch_data(VBranchGeoData.Fields.TANGENTS)  # type: ignore
    branches_curves = geodata.branch_curve()
    for branch in graph.branches():
        branch_id = branch.id
        curve = branches_curves[branch_id]
        if curve is None or len(curve) < 3:
            continue

        # 1. ... based on its calibre variation
        p_calibre = 0
        calibre = branches_calibres[branch_id]
        if calibre is not None and len(calibre.data) > 40:
            calibre = calibre.data[calibre.data < 30]  # TODO: Remove this line when the calibre data is fixed
            n = len(calibre)
            calibre = np.nanmedian(calibre[: n // 2]) - np.nanmedian(calibre[n // 2 :])
            p_calibre = sigmoid(calibre / 3, antisymmetric=True)

        # 2. ... based on its tangents
        p_tan = 0
        if branches_tangents[branch_id] is not None:
            assert len(branches_tangents[branch_id].data), f"Branch {branch_id} has no tangent data."
            p_tan = expected_tangent_sim(branches_curves[branch_id], branches_tangents[branch_id].data).mean()

        p_dir = p_tan * 0.3 + p_calibre * 0.7
        p_dir *= sigmoid(len(curve) / 50, antisymmetric=True)
        branches_dir_p[branch_id] = p_dir, -p_dir

    # === Utility function to compute the edge probabilities  ===
    def root_p(node_yx, tip_tangent):
        # === Compute the probability that a branch is a root branch ===
        # 1. ... based on the distance to the optic disc and the border of the image
        od_dist = od_yx.distance(node_yx)
        border_dist = fundus_radius - center_yx.distance(node_yx)

        p_dist = opt.root_p_offset
        p_dist -= sigmoid(od_dist / od_mac_dist, antisymmetric=True)
        p_dist -= np.clip((border_dist - opt.root_candidates_border_dist) / opt.root_candidates_border_dist, 0.1, 1)
        p_dist = np.clip(p_dist, -1, 1)

        # 2. ... based on the angle between the tangent of the tip and the expected tangent
        # Expected tangent: away from the disc, toward the macula (similar to magnetic field lines)
        # p_tan = float(expected_tangent_sim(node_yx.numpy(), tip_tangent))
        return +p_dist  # + p_tan # Remove p_tan, direction is checked in the branch

    def branch_to_branch_p(tip_coords, tip_tangents, mean_calibres, av_labels, od_dist, offset=0) -> float:
        # === Compute the probability that a branch is the parent of another branch ===
        # 1. ... based on the angle between the tangents of the tips of the branches and the distance between the tips
        node_to_node_t = tip_coords[1] - tip_coords[0]
        node_to_node_dist = np.linalg.norm(node_to_node_t)
        if not np.any(np.isnan(mean_calibres)) and np.any(node_to_node_t != 0):
            node_to_node_t = node_to_node_t / node_to_node_dist
            p_tan = (
                np.cos(
                    np.clip(np.arccos(tip_tangents[0].dot(-node_to_node_t)), 0, np.pi / 2)
                    + np.clip(np.arccos(tip_tangents[1].dot(node_to_node_t)), 0, np.pi / 2)
                )
                + tip_tangents[0].dot(-tip_tangents[1])
            ) / 2
            p_dist = -np.maximum(node_to_node_dist - opt.b2b_tips_dist_penalty_min, 0) * opt.b2b_tips_dist_penalty_w
        else:
            p_tan = tip_tangents[0].dot(-tip_tangents[1])
            p_dist = -node_to_node_dist * opt.b2b_tips_dist_penalty_w

        # 2. ... based on the similarity of the AV labels of the branches
        if np.any(av_labels == AVLabel.UNK):
            p_av = 0
        else:
            p_av = 1 if av_labels[0] == av_labels[1] else -1
        p_av *= np.clip(1 - sigmoid(od_dist / od_mac_dist, antisymmetric=True), 0, 0.5)
        # p_av *= np.maximum(1- (od_dist / 2*od_mac_dist)**2, 0)

        # 3. ... based on the difference of the calibres of the branches
        if np.any(np.isnan(mean_calibres)) or np.any(mean_calibres == 0):
            p_calibre = 0
        else:
            p_calibre = sigmoid((mean_calibres[0] - mean_calibres[1]) / 3, antisymmetric=True)
            p_calibre *= opt.b2b_p_calibre_w
        common_p = p_tan * opt.b2b_p_tangent_w + p_av + offset + p_dist
        return common_p + p_calibre, common_p - p_calibre

    def fetch_tangents_calibre_av_labels(branch_ids, first_tip):
        tip_coords = geodata.tip_coord(branch_ids, first_tip)
        tip_tangents = geodata.tip_tangent(branch_ids, first_tip)
        calibres = geodata.branch_data(VBranchGeoData.Fields.CALIBRES, branch_ids)
        calibres = np.array([np.nanmean(c.data) if c is not None and len(c.data) else np.nan for c in calibres])
        av_labels = np.array(graph.branch_attr[av_attr][branch_ids], dtype=int)
        return tip_coords, tip_tangents, calibres, av_labels

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
        tips = node.adjacent_branches_first_node
        for t, b, first in zip(tangents, node.adjacent_branches(), tips, strict=True):
            edge_list.append([-1, b.id])
            edge_first_tip.append([0, 0 if first else 1])
            edge_probs.append(root_p(node_yx, t))

    # === Find closest nodes to the optic disc for each connected component ===
    nodes_od_dist = od_yx.distance(nodes_yx)
    nodes_border_dist = fundus_radius - center_yx.distance(nodes_yx)

    root_candidates = np.zeros(graph.node_count, dtype=bool)
    for nodes in graph.node_connected_components():
        root_candidates[nodes[np.argmin(od_yx.distance(nodes_yx[nodes]))]] = True
    for node in graph.nodes():
        if (
            node.degree == 1
            or nodes_od_dist[node.id] < od_diameter * opt.root_candidates_od_distance
            or nodes_border_dist[node.id] < 2 * opt.root_candidates_border_dist
        ):
            root_candidates[node.id] = True

    # === Create the directed line graph from the graph ===
    for node in graph.nodes():
        node_degree = node.degree

        if root_candidates[node.id]:
            # 1. ROOT NODES
            add_root_edge(node)

        if node_degree > 1:
            # 2. PASSING / BRANCHING / JUNCTION NODES
            od_dist = nodes_od_dist[node.id]

            # For each pairs of branches, compute the probability of the first being the parent of the second
            branch_ids = np.array(node.adjacent_branch_ids)
            first_tip = np.array(node.adjacent_branches_first_node)
            tip_coords, tip_tangents, mean_calibres, branch_av = fetch_tangents_calibre_av_labels(branch_ids, first_tip)

            for b1, b2 in itertools.combinations(range(len(branch_ids)), 2):
                b12 = [b1, b2]
                p12, p21 = branch_to_branch_p(
                    tip_coords[b12], tip_tangents[b12], mean_calibres[b12], branch_av[b12], od_dist
                )
                tip_id12 = np.where(first_tip[b12], 0, 1)
                add_edge_pair(branch_ids[b12], tip_id12, p12, p21)

    # 3. VIRTUAL EDGES: CONNECTING ENDPOINTS TOGETHER
    b, t = graph.adjacent_branches_per_node(virtual_endp_edges.flatten(), return_branch_direction=True)
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

        tip_coords, tips_tangents, branches_calibres, av_labels = fetch_tangents_calibre_av_labels(
            endp_branch, endp_first_tips
        )
        endp_branch = endp_branch.reshape(-1, 2)
        endp_first_tips = endp_first_tips.reshape(-1, 2)
        tip_coords = tip_coords.reshape(-1, 2, 2)
        tips_tangents = tips_tangents.reshape(-1, 2, 2)
        branches_calibres = branches_calibres.reshape(-1, 2)
        av_labels = av_labels.reshape(-1, 2)

        for e in range(len(endp_branch)):
            od_dist = np.mean(od_yx.distance(nodes_yx[virtual_endp_edges[e]]))
            # dist_between_endp = np.linalg.norm(np.diff(nodes_yx[virtual_endp_edges[e]], axis=0))
            p_offset = -1e-3  # -dist_between_endp / opt.reconnect_dist_penalty_coef - 1e-3
            p12, p21 = branch_to_branch_p(
                tip_coords[e], tips_tangents[e], branches_calibres[e], av_labels[e], od_dist, offset=p_offset
            )
            tip_id12 = np.where(endp_first_tips[e], 0, 1)
            add_edge_pair(endp_branch[e], tip_id12, p12, p21)

    # 4. VIRTUAL EDGES: CONNECTING ENDPOINTS TO BRANCHES
    for e in range(virtual_edges.shape[0]):
        endpoint_node = graph.node(virtual_edges[e, 0])
        endp_branch_ids = endpoint_node.adjacent_branch_ids
        endp_first_tips = endpoint_node.adjacent_branches_first_node
        endp_tip_coords, endp_tips_tangent, endp_mean_calibres, endp_branch_av = fetch_tangents_calibre_av_labels(
            endp_branch_ids, endp_first_tips
        )

        node = graph.node(virtual_edges[e, 1])
        node_branch_ids = node.adjacent_branch_ids
        node_first_tips = node.adjacent_branches_first_node
        node_tip_coords, node_tips_tangent, node_mean_calibres, node_branch_av = fetch_tangents_calibre_av_labels(
            node_branch_ids, node_first_tips
        )

        od_dist = np.mean(od_yx.distance(nodes_yx[node.id]))
        # dist_between_endp = np.linalg.norm(np.diff(nodes_yx[[endpoint_node.id, node.id]], axis=0))
        # p_offset = -dist_between_endp / opt.reconnect_dist_penalty_coef

        for i in range(len(endp_branch_ids)):
            for j in range(len(node_branch_ids)):
                p12, p21 = branch_to_branch_p(
                    tip_coords=np.array([endp_tip_coords[i], node_tip_coords[j]]),
                    tip_tangents=np.array([endp_tips_tangent[i], node_tips_tangent[j]]),
                    mean_calibres=np.array([endp_mean_calibres[i], node_mean_calibres[j]]),
                    av_labels=np.array([endp_branch_av[i], node_branch_av[j]]),
                    od_dist=od_dist,
                    # offset=p_offset,
                )
                branch_id12 = endp_branch_ids[i], node_branch_ids[j]
                first_tip12 = endp_first_tips[i], node_first_tips[j]
                tip_id12 = np.where(first_tip12, 0, 1)
                add_edge_pair(branch_id12, tip_id12, p12, p21)

    edge_through_node = []
    N = len(edge_list)

    # 5. VIRTUAL EDGES: CONNECTING ENDPOINTS TO ENDPOINTS THROUGH BRANCHES
    intercept_nodes, intercept_count = np.unique(virtual_edges[:, 1], return_counts=True)
    for through_node, count in zip(intercept_nodes, intercept_count, strict=True):
        if count <= 1:
            continue
        od_dist = np.mean(od_yx.distance(nodes_yx[through_node]))
        connected_endp = virtual_edges[virtual_edges[:, 1] == through_node, 0]
        for endp1, endp2 in itertools.combinations(connected_endp, 2):
            endp1_node = graph.node(endp1)
            endp2_node = graph.node(endp2)
            endp1_branch_ids = endp1_node.adjacent_branch_ids
            endp1_first_tips = endp1_node.adjacent_branches_first_node
            endp1_tip_coords, endp1_tips_tangent, endp1_mean_calibres, endp1_branch_av = (
                fetch_tangents_calibre_av_labels(endp1_branch_ids, endp1_first_tips)
            )

            endp2_branch_ids = endp2_node.adjacent_branch_ids
            endp2_first_tips = endp2_node.adjacent_branches_first_node
            endp2_tip_coords, endp2_tips_tangent, endp2_mean_calibres, endp2_branch_av = (
                fetch_tangents_calibre_av_labels(endp2_branch_ids, endp2_first_tips)
            )

            # dist_between_endp = np.linalg.norm(np.diff(nodes_yx[[endp1, endp2]], axis=0))
            # p_offset = -dist_between_endp / opt.reconnect_dist_penalty_coef
            for i in range(len(endp1_branch_ids)):
                for j in range(len(endp2_branch_ids)):
                    p12, p21 = branch_to_branch_p(
                        tip_coords=np.array([endp1_tip_coords[i], endp2_tip_coords[j]]),
                        tip_tangents=np.array([endp1_tips_tangent[i], endp2_tips_tangent[j]]),
                        mean_calibres=np.array([endp1_mean_calibres[i], endp2_mean_calibres[j]]),
                        av_labels=np.array([endp1_branch_av[i], endp2_branch_av[j]]),
                        od_dist=od_dist,
                        # offset=p_offset,
                    )
                    branch_id12 = endp1_branch_ids[i], endp2_branch_ids[j]
                    first_tip12 = endp1_first_tips[i], endp2_first_tips[j]
                    tip_id12 = np.where(first_tip12, 0, 1)
                    add_edge_pair(branch_id12, tip_id12, p12, p21)
                    edge_through_node += [through_node] * 2
    edge_through_node = np.concatenate([-np.ones(N), edge_through_node])

    # Relabel the branches that were duplicated
    graph.branch_attr.fillna({av_attr: AVLabel.UNK}, inplace=True)
    graph.branch_attr.loc[both_branches, av_attr] = AVLabel.UNK

    return graph, np.array(edge_list), np.array(edge_first_tip), np.array(edge_probs), edge_through_node, branches_dir_p


def resolve_digraph_to_vtree(
    vgraph: VGraph,
    line_list: npt.NDArray[np.int_],
    line_tips: npt.NDArray[np.int_],
    line_probability: npt.NDArray[np.float64],
    line_through_node: npt.NDArray[np.int_],
    branches_dir_p: npt.NDArray[np.float64],
    *,
    av_attr: str = "av",
    pre_filter_line: bool = True,
    debug_info: Optional[Dict[str, Any]] = None,
) -> VTree:
    import networkx as nx
    from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

    orphan_branch = vgraph.orphan_branches(as_mask=True)

    if pre_filter_line:
        digraph = nx.DiGraph()
        line_total_p = np.zeros(len(line_list), dtype=float)
        for id, (line, p, tips_id, through_node) in enumerate(
            zip(line_list, line_probability, line_tips, line_through_node, strict=True)
        ):
            b_from, b_to = line + 1
            b_from_tip_id, b_to_tip_id = tips_id
            if b_from_tip_id == 0:
                b_from = -b_from  # Source: First node of the source branch => the source branch is reversed

            if b_to_tip_id == 1:
                b_to = -b_to  # Target: Second node of the destination branch => the dest branch is reversed

            if line[0] != -1:
                p += branches_dir_p[line[0], 0 if b_from_tip_id == 1 else 1]
                p += branches_dir_p[line[1], 0 if b_to_tip_id == 0 else 1]
            else:
                p += 2 * branches_dir_p[line[1], 0 if b_to_tip_id == 0 else 1]

            if (already_added := digraph.edges.get((b_from, b_to), None)) is not None and already_added["p"] >= p:
                continue
            digraph.add_edge(b_from, b_to, p=p, id=id, through=through_node)
            line_total_p[id] = p

        # Solve the double optimal tree (using both direction for each branch)
        try:
            optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)
        except nx.NetworkXException as e:
            warnings.warn(f"Error while solving the optimal tree: {e}")
            optimal_tree = digraph

        kept_edges = np.zeros(len(line_list), dtype=bool)

        digraph = nx.DiGraph()
        for B0, B1, info in optimal_tree.edges(data=True):
            b0, b1 = abs(B0) - 1, abs(B1) - 1
            if (already_added := digraph.edges.get((b0, b1), None)) is not None and already_added["p"] >= info["p"]:
                continue
            digraph.add_edge(
                b0,
                b1,
                p=info["p"],
                id=info["id"],
                through=info["through"],
                tips=[1 if B0 > 0 else 0, 0 if B1 > 0 else 1],
            )
            kept_edges[info["id"]] = True

        if debug_info is not None:
            debug_info["kept_edges"] = kept_edges
            debug_info["P"] = line_total_p
    else:
        digraph = nx.DiGraph()
        for id, (line, p, tips_id) in enumerate(zip(line_list, line_probability, line_tips, strict=True)):
            if orphan_branch[line[1]] and (already_added := digraph.edges.get(line, None)) is not None:
                if already_added["p"] >= p:
                    continue
            digraph.add_edge(*line, p=p, id=id, tips=tips_id)

    optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)

    N_ini_branch = vgraph.branch_count
    branch_tree = {}
    branch_dirs = {}
    branch_head = {}
    for b_from, b_to in nx.edge_bfs(optimal_tree, -1):
        info = optimal_tree[b_from][b_to]
        tips_id = info["tips"]
        n1 = vgraph.branch_list[b_from, tips_id[0]]
        n2 = vgraph.branch_list[b_to, tips_id[1]]

        # Check for rebound, if detected redirect the branch to its valid ancestor
        if b_from != -1:
            while branch_head[b_from] != n1:
                b_from = branch_tree[b_from]
                if b_from == -1:
                    break
                tips_id[0] = 1 if branch_dirs[b_from] else 0
                n1 = vgraph.branch_list[b_from, tips_id[0]]

                if b_from >= N_ini_branch and vgraph.branch_list[b_from, 1 - tips_id[0]] == n2:
                    # If the redirection leads to a virtual branch whose tail is n2, take its parent again
                    b_from = branch_tree[b_from]
                    tips_id[0] = 1 if branch_dirs[b_from] else 0
                    n1 = vgraph.branch_list[b_from, tips_id[0]]
                    break

        if b_from == -1 or n1 == n2:
            branch_tree[b_to] = b_from
        else:
            if (through_node := info.get("through", -1)) != -1:
                _, [b_id0, b_id1] = vgraph.add_branch(
                    [[n1, through_node], [through_node, n2]], return_branch_id=True, inplace=True
                )
                branch_tree[b_id0] = b_from
                branch_tree[b_id1] = b_id0
                branch_dirs[b_id0] = True
                branch_head[b_id0] = through_node
                branch_dirs[b_id1] = True
                branch_head[b_id1] = n2
                branch_tree[b_to] = b_id1
                vgraph.branch_attr.loc[[b_id0, b_id1], av_attr] = vgraph.branch_attr.loc[b_from, av_attr]
            else:
                vgraph, b_id = vgraph.add_branch([[n1, n2]], return_branch_id=True, inplace=True)
                b_id = b_id[0]
                branch_tree[b_id] = b_from
                branch_dirs[b_id] = True
                branch_head[b_id] = n2
                branch_tree[b_to] = b_id
                vgraph.branch_attr.loc[b_id, av_attr] = vgraph.branch_attr.loc[b_from, av_attr]

        branch_dirs[b_to] = tips_id[1] == 0
        branch_head[b_to] = vgraph.branch_list[b_to, 1 - tips_id[1]]

    try:
        branch_tree = np.array([branch_tree[b] for b in range(vgraph.branch_count)], dtype=int)
        branch_dirs = np.array([branch_dirs[b] for b in range(vgraph.branch_count)], dtype=bool)
    except KeyError:
        raise ValueError("Some branches were not added to the tree.") from None

    vtree = VTree.from_graph(vgraph, branch_tree, branch_dirs, copy=False)

    # Fix branches that are both primary and secondary for a given node
    secondary_branches = np.argwhere(vtree.branch_tree != -1).flatten()
    primary_branches = vtree.branch_tree[secondary_branches]
    # invalid_prim_sec = vtree.branch_tail(primary_branches) == vtree.branch_tail(secondary_branches)
    invalid_prim_sec = vtree.branch_tail(secondary_branches) != vtree.branch_head(primary_branches)
    invalid_prim = primary_branches[invalid_prim_sec]

    if len(invalid_prim) > 0:
        raise ValueError("Rebound still present in the tree.")
        # Redirect their direct successors to their direct ancestor
        invalid_sec = secondary_branches[invalid_prim_sec]
        vtree.branch_tree[invalid_sec] = vtree.branch_tree[invalid_prim]

    return vtree


def inspect_digraph_solving(
    fundus_data: FundusData,
    graph: VGraph,
    *,
    av_attr: str = "av",
):
    import pandas as pd
    import panel as pn
    from bokeh.models.widgets.tables import BooleanFormatter, NumberFormatter
    from jppype import Mosaic

    pn.extension("tabulator")

    graph = graph.copy()

    # === Compute and solve the digraph ===
    graph, line_list, line_tips, line_probability, line_through, branches_dir_p = build_line_digraph(
        graph,
        fundus_data,
        av_attr=av_attr,
    )
    B = graph.branch_count
    debug_info = {}
    tree = resolve_digraph_to_vtree(
        graph.copy(), line_list, line_tips, line_probability, line_through, branches_dir_p, debug_info=debug_info
    )
    tree.branch_attr["subtree"] = tree.subtrees_branch_labels()

    a_tree, v_tree = split_av_graph_by_subtree(tree, av_attr=av_attr)

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

    m = Mosaic(3)
    fundus_data.draw(view=m, labels_opacity=0.3)

    m[0]["graph"] = graph.jppype_layer(bspline=True, edge_labels=True, node_labels=True)
    m[0]["tangents"] = graph.geometric_data().jppype_branch_tip_tangent(scaling=2)
    m[0]["graph"].edges_cmap = graph.branch_attr["av"].map(GRAPH_AV_COLORS).to_dict()

    m[1]["graph"] = tree.jppype_layer(bspline=True, edge_labels=True, node_labels=True)
    m[1]["tangents"] = tree.geometric_data().jppype_branch_tip_tangent(
        scaling=4, show_only="junctions", invert_direction="tree"
    )
    m[1]["graph"].edges_cmap = tree.branch_attr["subtree"].map(GENERIC_CMAP).to_dict()
    m[1]["graph"].edges_labels = tree.branch_attr["subtree"].to_dict()

    nodes_color = pd.Series("grey", index=tree.node_attr.index)
    nodes_color[tree.root_nodes_ids()] = "black"
    m[1]["graph"].nodes_cmap = nodes_color.to_dict()

    m[2]["a_tree"] = a_tree.jppype_layer(bspline=True, edge_labels=True, node_labels=True)
    m[2]["a_tree"].edges_cmap = GRAPH_AV_COLORS[AVLabel.ART]
    nodes_color = pd.Series(GRAPH_AV_COLORS[AVLabel.ART], index=a_tree.node_attr.index)
    nodes_color[a_tree.root_nodes_ids()] = "#7a1a1a"
    nodes_color[a_tree.leaf_nodes_ids()] = "#da7676"
    m[2]["a_tree"].nodes_cmap = nodes_color.to_dict()

    m[2]["v_tree"] = v_tree.jppype_layer(bspline=True, edge_labels=True, node_labels=True)
    m[2]["v_tree"].edges_cmap = GRAPH_AV_COLORS[AVLabel.VEI]
    nodes_color = pd.Series(GRAPH_AV_COLORS[AVLabel.VEI], index=v_tree.node_attr.index)
    nodes_color[v_tree.root_nodes_ids()] = "#1a1a7a"
    nodes_color[v_tree.leaf_nodes_ids()] = "#7676da"
    m[2]["v_tree"].nodes_cmap = nodes_color.to_dict()

    m.show()

    # === Create the graph views ===
    branch_anc = tree.branch_tree[line_list[:, 1]]
    b0, b1 = line_list.T
    tip0, tip1 = line_tips.T

    lines_data = dict(
        b0=b0,
        b1=b1,
        n0=tree.branch_list[b0, tip0],
        n1=tree.branch_list[b1, tip1],
        tree1=tree.branch_attr["subtree"][b1].to_numpy(),
        p=line_probability,
        dir0_p=branches_dir_p[b0, np.where(tip0 == 1, 0, 1)],
        dir1_p=branches_dir_p[b1, np.where(tip1 == 0, 0, 1)],
    )
    if "P" in debug_info:
        lines_data["P"] = debug_info["P"]
    lines_data |= dict(
        optimal=((branch_anc == b0) | ((branch_anc > B) & (tree.branch_tree[branch_anc] == b0))),
        kept=debug_info.get("kept_edges", np.zeros(b0.shape, dtype=bool)),
    )
    lines_data = pd.DataFrame(lines_data)
    root_data = lines_data[lines_data["b0"] == -1]
    root_data = root_data.drop(columns=["b0", "n0", "dir0_p"])
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
            "P": NumberFormatter(format="0.000"),
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
        coord = tree.node_coord()[root_data["n1"].iloc[event.row]]
        m[0].goto(coord[::-1], scale=3)

    def focus_on_non_root(event):
        coord = tree.node_coord()[non_root_data["n1"].iloc[event.row]]
        m[0].goto(coord[::-1], scale=3)

    root_table.on_click(focus_on_root)
    non_root_table.on_click(focus_on_non_root)
    non_root_table2.on_click(focus_on_non_root)

    return (
        pn.Row(root_table, pn.Spacer(width=10), non_root_table, pn.Spacer(width=10), non_root_table2, height=400),
        graph,
        (a_tree, v_tree),
        lines_data,
    )
