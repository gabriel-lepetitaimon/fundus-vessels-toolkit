import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from fundus_toolkits import AVLabel, FundusData
from fundus_toolkits.utils.geometric import Point
from fundus_toolkits.utils.typing import Bool1DArray, Indices

from ..pipelines.seg_to_graph import SegToGraph
from ..utils.cluster import cluster_by_distance, reduce_clusters
from ..utils.math import extract_splits, quantized_higher
from ..vascular_data_objects import BranchIndicesLike, VBranchGeoData, VGraph, VTree
from .graph_simplification import simplify_passing_nodes


def assign_av_label(
    graph: VGraph,
    av_map: Optional[npt.NDArray[np.uint8] | FundusData] = None,
    *,
    av_medfilt_size: int = 9,
    split_av_branch=True,
    split_high_curvature=0,
    split_av_threshold=4 / 5,
    av_attr="av",
    discard_joint_branch_geometry=True,
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

    geodata = graph.geometric_data()
    # === Split branches with high curvature ===
    if split_high_curvature and geodata.has_branch_data(VBranchGeoData.Fields.CURVATURES):
        curvatures = geodata.branch_data(VBranchGeoData.Fields.CURVATURES)
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
        if not np.issubdtype(branch_curve.dtype, np.integer):
            branch_curve = np.round(branch_curve).astype(int)
        # 0. Check the AV labels under each pixel of the skeleton and boundaries of the branch
        bound = branch.geodata(VBranchGeoData.Fields.BOUNDARIES, geodata).data
        valid_bound = geodata.domain.contains(bound).all(axis=1)
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
            main_av_label = [AVLabel.ART, AVLabel.VEI, AVLabel.BOTH][np.argmax([n_art, n_vei, n_both])]
            branches_av_attr[branch.id] = main_av_label

    graph.branch_attr[av_attr] = branches_av_attr  # Why is this line necessary?

    # === Assign AV labels to nodes and propagate them through unknown passing nodes ===
    propagate_av_labels(graph=graph, av_attr=av_attr, only_label_nodes=not propagate_labels, inplace=True)

    # === Remove or update geometry of branches with both type ===
    if discard_joint_branch_geometry:
        # geodata.clear_branch_gdata(graph.as_branch_ids(graph.branch_attr[av_attr] == AVLabel.BOTH))
        ...
    else:
        segToGraph = SegToGraph(max_spurs_length=5, clean_branches_tips=5)
        branch_to_delete = []
        for branch in graph.branches(graph.branch_attr[av_attr] == AVLabel.BOTH):
            if branch.curve().shape[0] < 5:
                geodata.clear_branch_gdata(branch.id)
            else:
                # Draw the adjacent branches mask
                branch_mask, bbox = branch.rasterize(geodata=geodata, return_bbox=True, expand=2)
                av_bbox = av_map[bbox.slice()]
                both_mask = np.isin(av_bbox, (AVLabel.BOTH, AVLabel.UNK))
                a_mask = av_bbox == AVLabel.ART
                v_mask = av_bbox == AVLabel.VEI
                if (
                    both_mask[branch_mask & (av_bbox != AVLabel.BKG)].mean() > 0.8
                    or a_mask[branch_mask].mean() < 0.1
                    or v_mask[branch_mask].mean() < 0.1
                ):
                    continue  # The AV map does not provide enough information to split the branch

                # Isolate the artery mask and the vein mask
                a_mask = (a_mask | both_mask) & branch_mask
                v_mask = (v_mask | both_mask) & branch_mask

                # Parse both mask individually
                def parse_graph(mask, branch):
                    # Parse topology
                    g = segToGraph(mask, simplify=False, parse_geometry=True)
                    g.geometric_data()._domain = bbox + geodata.domain.top_left

                    # Filter branch whose tangents are not aligned with the branch main directions
                    both_curve = branch.curve()
                    both_t = branch.geodata(VBranchGeoData.Fields.TANGENTS, geodata).data
                    invalid_branches = []
                    for b in g.branches():
                        if (b_curve := b.curve()).shape[0] == 0:
                            invalid_branches.append(b.id)
                            continue
                        D = np.linalg.norm(both_curve[None, :, :] - b_curve[:, None, :], axis=2)
                        closest_points = np.argmin(D, axis=1)
                        b_t = b.geodata(VBranchGeoData.Fields.TANGENTS, g.geometric_data()).data
                        cos_sim = np.einsum("ij,ij->i", both_t[closest_points], b_t).mean()
                        if -0.7 < cos_sim < 0.7:
                            invalid_branches.append(b.id)
                    g.delete_branch(invalid_branches, inplace=True)
                    return g

                a_graph = parse_graph(a_mask, branch)
                b_graph = parse_graph(v_mask, branch)

                # If no branch were found leave the branch as both
                if a_graph.is_empty() and b_graph.is_empty():
                    continue
                # If a single artery or vein branch was found re-assign the branch label
                elif a_graph.is_empty() and b_graph.branch_count == 1:
                    branch.attr[av_attr] = AVLabel.VEI
                    continue
                elif a_graph.branch_count == 1 and b_graph.is_empty():
                    branch.attr[av_attr] = AVLabel.ART
                    continue

                # Otherwise incorporate the new branches in the graph
                def add_branches(g, av_label, branch):
                    if g.branch_count == 0:
                        return False

                    # Insert branches in the graph
                    new_branches = np.arange(graph.branch_count, graph.branch_count + g.branch_count)
                    new_nodes = np.arange(graph.node_count, graph.node_count + g.node_count)
                    graph.append(g, inplace=True)
                    graph.node_attr.loc[new_nodes, av_attr] = av_label
                    graph.branch_attr.loc[new_branches, av_attr] = av_label

                    # Merge the closest node with the previous branch start and end
                    tip_nodes = branch._node_ids
                    tip_yx = geodata.node_coord(tip_nodes)
                    nodes_yx = geodata.node_coord(new_nodes)
                    D = np.linalg.norm(tip_yx[:, None, :] - nodes_yx[None, :, :], axis=2)
                    closest = np.argmin(D[:, : g.node_count], axis=1)
                    MAX_MERGE_DIST = 25
                    if D[0, closest[0]] > MAX_MERGE_DIST:
                        closest[0] = -1
                    if D[1, closest[1]] > MAX_MERGE_DIST:
                        closest[1] = -1
                    if closest[0] == closest[1]:
                        if D[0, closest[0]] < D[1, closest[1]]:
                            closest[1] = -1
                        else:
                            closest[0] = -1

                    nodes_weight = np.zeros(graph.node_count, dtype=float)
                    nodes_weight[tip_nodes] = 1
                    clusters = []
                    if closest[0] >= 0:
                        clusters += [[tip_nodes[0], new_nodes[closest[0]]]]
                    if closest[1] >= 0:
                        clusters += [[tip_nodes[1], new_nodes[closest[1]]]]
                    if len(clusters) == 0:
                        return False
                    graph.merge_nodes(clusters, inplace=True, assume_reduced=True, nodes_weight=nodes_weight)
                    return True

                add_branches(a_graph, AVLabel.ART, branch)
                add_branches(b_graph, AVLabel.VEI, branch)

                branch_to_delete.append(branch.id)

        if branch_to_delete:
            graph.delete_branch(branch_to_delete, inplace=True)

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

    # === Merge nodes of the same type connected by a small branch ===
    nodes_clusters = []
    unknown_nodes_clusters = []
    max_merge_distance = max(node_merge_distance, unknown_node_merge_distance)
    for branch in graph.branches(filter="non-endpoint"):
        if branch.node_to_node_length() < max_merge_distance:
            n1, n2 = graph.node_attr.loc[list(branch.node_ids), av_attr]  # type: ignore
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
    twin_branches = []
    for twins in graph.twin_branches():
        b0 = graph.branch(twins[0])
        if b0.node_to_node_length() < unknown_node_merge_distance or not geodata.has_branch_curve(twins).all():
            # Keep the first branch and remove the others
            twin_branches.extend(twins[1:])
            # Label the first branch as unknown and clear its geometry data
            b0.attr[av_attr] = AVLabel.UNK
            graph.node_attr.loc[list(b0.node_ids), av_attr] = AVLabel.UNK
            geodata.clear_branch_gdata([b0.id])

    graph.delete_branch(twin_branches, inplace=True)

    # === Remove passing nodes of same type ===
    graph.node_connected_components()
    simplify_passing_nodes(graph, min_angle=passing_node_min_angle, with_same_branch_attr=av_attr, inplace=True)

    # === Delete self-loop undefined branches ===
    self_loop = graph.self_loop_branches()
    if len(self_loop) > 0:
        self_loop = self_loop[geodata.has_branch_curve(self_loop)]
        graph.delete_branch(self_loop, inplace=True)

    # === Relabel unknown branches ===
    if propagate_labels:
        propagate_av_labels(graph, av_attr=av_attr, inplace=True)

    # === Remove geometry of branches with both type ===
    # geodata.clear_branch_gdata(graph.as_branch_ids(graph.branch_attr[av_attr] == AVLabel.BOTH))

    return graph


def remove_unknown_leaf_branches(tree: VTree, av_attr: str = "av", inplace: bool = False) -> VTree:
    if not inplace:
        tree = tree.copy()

    while (unknown_leafs := ((tree.branch_attr[av_attr] == AVLabel.UNK) & tree.leaf_branch_ids(as_mask=True))).any():
        tree.delete_branch(unknown_leafs, inplace=True)

    return tree


def split_av_graph(
    trees: VTree, *, av_attr: str = "av", simplify: bool = True, center_junction_nodes: bool = True
) -> Tuple[VTree, VTree]:
    b_attr = trees.branch_attr
    a_graph = trees.delete_branch(b_attr[av_attr] == AVLabel.VEI, inplace=False)
    v_graph = trees.delete_branch(b_attr[av_attr] == AVLabel.ART, inplace=False)

    if simplify:
        from .graph_simplification import simplify_passing_nodes

        remove_unknown_leaf_branches(a_graph, av_attr=av_attr, inplace=True)
        remove_unknown_leaf_branches(v_graph, av_attr=av_attr, inplace=True)

        simplify_passing_nodes(a_graph, with_same_label=av_attr, inplace=True)
        simplify_passing_nodes(v_graph, with_same_label=av_attr, inplace=True)

    if center_junction_nodes:
        from .geometry_parsing import center_junction_nodes as center_junctions

        center_junctions(a_graph, inplace=True)
        center_junctions(v_graph, inplace=True)

    return a_graph, v_graph


def split_av_graph_by_subtree(
    tree: VTree,
    *,
    av_attr: str = "av",
    detect_major_error: bool = True,
    simplify: bool = True,
    center_junction_nodes: bool = True,
    inplace: bool = False,
) -> Tuple[VTree, VTree]:
    if not inplace:
        tree = tree.copy()

    tree.branch_attr.fillna({av_attr: AVLabel.UNK}, inplace=True)
    art_branches = tree.as_branch_ids(tree.branch_attr[av_attr] == AVLabel.ART)
    vei_branches = tree.as_branch_ids(tree.branch_attr[av_attr] == AVLabel.VEI)
    geodata = tree.geometric_data()

    if geodata.has_branch_data(VBranchGeoData.Fields.CALIBRES):
        total_calibres = [
            c.data[np.isfinite(c.data)].sum() if c is not None else 0
            for c in geodata.branch_data(VBranchGeoData.Fields.CALIBRES)
        ]
    else:
        total_calibres = geodata.branch_arc_length()
    total_calibres = np.array(total_calibres)

    def subtree_av_weight(subtree):
        # a_weight = np.sum(geodata.branch_arc_length(np.intersect1d(subtree, art_branches), fast_approximation=True))
        # v_weight = np.sum(geodata.branch_arc_length(np.intersect1d(subtree, vei_branches), fast_approximation=True))
        a_weight = total_calibres[np.intersect1d(subtree, art_branches)].sum()
        v_weight = total_calibres[np.intersect1d(subtree, vei_branches)].sum()
        return a_weight, v_weight

    # === Assign one artery or vein label to each subtree ===
    subtrees = tree.branch_ids_by_subtree()
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

        remove_unknown_leaf_branches(a_tree, av_attr=av_attr, inplace=True)
        remove_unknown_leaf_branches(v_tree, av_attr=av_attr, inplace=True)

        simplify_passing_nodes(a_tree, min_angle=110, inplace=True)
        simplify_passing_nodes(v_tree, min_angle=110, inplace=True)

        # disconnect_crossing_nodes(a_tree, inplace=True)
        # disconnect_crossing_nodes(v_tree, inplace=True)

    # === Center junction nodes ===
    if center_junction_nodes:
        from .geometry_parsing import center_junction_nodes as center_junctions
        from .geometry_parsing import snap_leaf_nodes_to_tips

        center_junctions(a_tree, inplace=True)
        center_junctions(v_tree, inplace=True)

        snap_leaf_nodes_to_tips(a_tree, inplace=True)
        snap_leaf_nodes_to_tips(v_tree, inplace=True)

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

    for root in tree.root_branch_ids():
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


def naive_infer_roots(
    graph: VGraph,
    root_pos: Point,
    *,
    force_roots: Optional[Indices] = None,
    reorder_nodes: bool = False,
    reorder_branches: bool = False,
    inplace: bool = False,
) -> VTree:
    # === Prepare graph ===
    if not inplace:
        graph = graph.copy()
    else:
        assert isinstance(graph, VTree), "Inplace conversion to VTree is only available for VTree instances."

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

    if force_roots is not None:
        for node in force_roots:
            roots[node] = np.linalg.norm(nodes_coord[node] - root_pos)

    for nodes in graph.node_connected_components():
        if np.any(np.isin(nodes, list(roots.keys()))):
            continue
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
    if inplace:
        vtree: VTree = graph  # type: ignore
        vtree._branch_tree = branch_tree
        vtree._branch_dir = branch_dirs
    else:
        vtree = VTree.from_graph(graph, branch_tree, branch_dirs, copy=False)

    if reorder_nodes:
        new_order = np.array([n.id for n in vtree.walk_nodes(traversal="dfs")], dtype=int)
        _, idx = np.unique(new_order, return_index=True)  # Take the first occurrence of each node
        vtree.reindex_nodes(new_order[np.sort(idx)], inverse_lookup=True, inplace=True)

    if reorder_branches:
        new_order = [b.id for b in vtree.walk_branches(traversal="dfs")]
        vtree.reindex_branches(new_order, inverse_lookup=True, inplace=True)

    return vtree


def naive_infer_arborescence(
    graph: VGraph,
    root_pos: Point,
    *,
    force_roots: Optional[Indices] = None,
    branch_subset: Optional[BranchIndicesLike] = None,
) -> tuple[Indices, Bool1DArray]:
    # === Prepare graph ===
    if branch_subset is not None:
        branch_mask = np.zeros(graph.branch_count, dtype=bool)
        branch_mask[graph.as_branch_ids(branch_subset)] = True
    else:
        branch_mask = np.ones(graph.branch_count, dtype=bool)

    loop_branches = graph.self_loop_branches(as_mask=True)
    if (loop_branches & branch_mask).sum() > 0:
        warnings.warn("The graph contains self loop branches in the branch subset. They will be ignored.")
        branch_mask &= ~loop_branches

    nodes_coord = graph.node_coord()
    branch_list = graph.branch_list

    # === Prepare result variables ===
    branch_tree = -np.ones((graph.branch_count,), dtype=int)
    branch_dirs = np.zeros(len(graph.branch_list), dtype=bool)
    visited_branches = np.zeros(graph.branch_count, dtype=bool)

    # === Utilities method ===
    def list_adjacent_branches(node: int) -> npt.NDArray[np.intp]:
        branches = np.argwhere(np.any(branch_list == node, axis=1)).flatten()
        branches = branches[branch_mask[branches]]
        return np.stack([branches, np.where(branch_list[branches, 0] == node, 1, 0)]).T

    ID, DIR = 0, 1

    def list_direct_successors(branch: int) -> npt.NDArray[np.intp]:
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

    if force_roots is not None:
        for node in force_roots:
            roots[node] = np.linalg.norm(nodes_coord[node] - root_pos)

    for nodes in reduce_clusters(graph._branch_list[branch_mask], drop_singleton=False):
        nodes = np.asarray(nodes, dtype=int)
        if np.any(np.isin(nodes, list(roots.keys()))):
            continue
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
                VBranchGeoData.Fields.TIPS_TANGENT, adjacent_branches, first_tip=adjacent_dirs.astype(bool)
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

    assert np.all(visited_branches[branch_mask]), "Some branches were not added to the tree."

    return branch_tree[branch_mask], branch_dirs[branch_mask]
