from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product as iter_product
from typing import TYPE_CHECKING, Callable, Literal, Optional, Sequence, Type

import numpy as np
from pygmtools.linear_solvers import hungarian
from scipy.ndimage import distance_transform_edt

from fundus_toolkits import FundusData, Point, Rect
from fundus_toolkits.transform import (
    AffineTransform,
    IdentityTransform,
    QuadraticTransform,
    RadialToRadialTransform,
    SimilarityTransform,
    Transform,
    Translation,
)
from fundus_toolkits.utils.typing import Bool2DArray, FloatPairArray, Int1DArray, IntPairArray, IntPairArrayLike

from ..segment_to_graph.graph_simplification import simplify_passing_nodes
from ..utils.graph.matching import incident_branches_similarity
from ..vascular_data_objects import VGraph
from ..vascular_data_objects.vtree import VTree
from ..vmatching.descriptor import junction_adjacent_branches_descriptor, tree_node_histogram
from ..vmatching.vgraph_edit_distance import backtrack_edges, shortest_secondary_path
from .node_matching import match_junctions, ransac_refine_node_matching

if TYPE_CHECKING:
    from ..utils.jppype import Mosaic


@dataclass
class RegistrationResult:
    transformation: Transform
    matched_nodes: Int1DArray
    mean_error: float


def register_graph(
    fix_graph: VGraph,
    moving_graph: VGraph,
    matched_nodes: Optional[IntPairArrayLike] = None,
    projection: Optional[Type[Transform]] = None,
    *,
    reindex_graphs: bool = False,
    register_branches: bool = True,
) -> RegistrationResult:
    """
    Register two vascular graphs together.

    Parameters
    ----------
    fix_graph : VGraph
        The fixed vascular graph.

    moving_graph : VGraph
        The moving vascular graph.

    projection : Optional[Type[FundusProjection]], optional
        The projection to use for the registration. By default None.

    register_branches : bool, optional
        Whether to register the branches of the vascular graph in addition to the nodes coordinates. By default True.

    Returns
    -------
    np.ndarray
        The transformation to apply to the moving vascular graph to register it to the fixed vascular graph.

    Raises
    ------
    ValueError
        If the provided vascular graphs do not overlap enough to be registered together

    """
    if matched_nodes is None:
        matched_nodes = match_junctions(fix_graph, moving_graph)
    try:
        T_mov_fix, ransac_matched_nodes, mean_error = ransac_refine_node_matching(
            fix_graph,
            moving_graph,
            matched_nodes,
            final_projection=projection,
            return_mean_error=True,
            reindex_graphs=reindex_graphs,
        )
    except ValueError:
        raise ValueError("The provided vascular graph does not overlap enough to be registered together.") from None

    return RegistrationResult(T_mov_fix, ransac_matched_nodes, mean_error)


def multi_vgraph_registration(
    vgraphs: Sequence[VGraph],
    projection: Optional[Type[Transform] | dict[int, Type[Transform]]] = None,
    iterative: bool = False,
    ensure_exact: Literal["direct", "inverse", None] = None,
) -> list[Transform]:
    """
    Register multiple vascular graphs together.

    Parameters
    ----------
    vgraphs : Sequence[VGraph]
        The vascular graphs to register.

    Returns
    -------
    list[FundusProjection]
        The transformations to apply to each vascular graph to register them together.

    Raises
    ------
    ValueError
        If the provided vascular graphs do not overlap enough to be registered together

    """
    import networkx as nx

    if projection is None:
        projection = {3: AffineTransform, 12: QuadraticTransform}

    n_graph = len(vgraphs)
    matching = {}

    # Find the transformation between each pairs of graphs
    for i_fix, i_mov in iter_product(range(n_graph), repeat=2):
        if i_mov >= i_fix:
            continue
        g_fix, g_mov = vgraphs[i_fix], vgraphs[i_mov]
        matched_nodes = match_junctions(g_fix, g_mov)

        try:
            T_mov_fix, ransac_matched_nodes, mean_error = ransac_refine_node_matching(
                g_fix, g_mov, matched_nodes, return_mean_error=True, final_projection=projection
            )
        except ValueError:
            continue
        n_ransac_matched_nodes = len(ransac_matched_nodes)
        edge_weight = -mean_error * math.sqrt(n_ransac_matched_nodes)
        matched_fix, matched_mov = matched_nodes
        matching[(i_mov, i_fix)] = (T_mov_fix, (matched_mov, matched_fix), edge_weight)

    # Find the spanning tree of graphs providing the least registration mean error
    G = nx.Graph()
    G.add_nodes_from(range(n_graph))
    G.add_weighted_edges_from([(i1, i2, weight) for (i1, i2), (_, _, weight) in matching.items()])
    ST = nx.minimum_spanning_tree(G)

    # Check that ST is a single connected component
    subtrees = list(nx.connected_components(ST))
    if len(subtrees) != 1:
        raise ValueError(
            "The provided vascular graphs does not overlap enough to be registered together.\n"
            f"The registration identified the following clusters: {subtrees}."
        )

    # Accumulate the transformation from all other graphs to the center of the spanning tree
    root = nx.center(ST)[0]
    yx0 = vgraphs[root].node_coord()
    transformations = {root: Transform.identity()}

    priority: Optional[Callable[[Sequence[int]], Sequence[int]]] = None
    if iterative:
        nodes_weight = {
            i2: matching[(i1, i2)][2] if (i1, i2) in matching else matching[(i2, i1)][2]
            for i1, i2 in nx.bfs_edges(ST, root)
        }
        nodes_weight[root] = 0

        def priority_(nodes):
            return sorted(list(nodes), key=lambda x: nodes_weight[x])

        priority = priority_

        def extended_match_coord(fundus_id):
            adj_fundus_match = {}
            # Find the matches with adjacent fundus already transformed
            for (i1, i2), (T, match, _) in matching.items():
                if i1 == fundus_id and i2 in transformations:
                    adj_fundus_match[i2] = match
                elif i2 == fundus_id and i1 in transformations:
                    adj_fundus_match[i1] = match[1], match[0]

            if len(adj_fundus_match) <= 1:
                return None, None

            # Compute the transformed coordinates of the matched nodes
            all_own_match = np.concatenate([_[0] for _ in adj_fundus_match.values()])
            ext_match, ext_match_inverse, ext_match_count = np.unique(
                all_own_match, return_counts=True, return_inverse=True
            )
            extended_adj_coord = np.zeros((len(ext_match_count), 2))
            ext_match_inverse = np.split(ext_match_inverse, np.cumsum([len(_[0]) for _ in adj_fundus_match.values()]))
            for match_id, (adj_fundus, (_, adj_match)) in zip(
                ext_match_inverse[:-1], adj_fundus_match.items(), strict=True
            ):
                # Apply the already established transformation to the matched node of the adjacent fundus
                T_adj_0 = transformations[adj_fundus]
                adj_transformed_coord = T_adj_0.transform(vgraphs[adj_fundus].node_coord()[adj_match])
                # Accumulate the transformed coordinates
                extended_adj_coord[match_id] += adj_transformed_coord
            # Normalize the accumulated coordinates by the number of fundus where each node was matched
            extended_adj_coord /= ext_match_count[:, None]

            # Return the matched nodes coordinates in the current fundus and in the already transformed fundus
            return vgraphs[fundus_id].node_coord()[ext_match], extended_adj_coord

    else:
        priority = None

    for i1, i2 in nx.bfs_edges(ST, root, sort_neighbors=priority):
        # Read the transformation from i2 to i1
        if i1 < i2:
            T12, (match1, match2), _ = matching[(i1, i2)]
            T21 = T12.invert()
        else:
            T12, (match2, match1), _ = matching[(i2, i1)]
        # If i2 instead of i1 is already transformed, invert i1 and i2
        if i2 in transformations:
            i1, i2 = i2, i1
            match1, match2 = match2, match1
            T21: Transform = T21.invert()

        # If iterative is true, check if we can recompute the transformation from scratch
        if iterative:
            ext_yx2, ext_yx0 = extended_match_coord(i2)
            if ext_yx2 is not None:
                assert ext_yx0 is not None
                # If this fundus can be registered to several already transformed fundus, recompute the transformation
                if ensure_exact == "inverse":
                    T02 = Transform.fit_to_projection(ext_yx0, ext_yx2, projection=projection)[0]
                    T20 = T02.invert()
                else:
                    T20 = Transform.fit_to_projection(ext_yx2, ext_yx0, projection=projection)[0]
                transformations[i2] = T20
                continue

        # If the transformation is inexact and we want to ensure exactness, recompute it from scratch
        if ensure_exact == "inverse" and not T21.is_inverse_exact:
            yx2 = vgraphs[i2].node_coord()[match2]
            yx1 = vgraphs[i1].node_coord()[match1]
            T12 = Transform.fit_to_projection(yx1, yx2, projection=projection)[0]
            T21 = T12.invert()
        elif ensure_exact == "direct" and not T21.is_exact:
            yx2 = vgraphs[i2].node_coord()[match2]
            yx1 = vgraphs[i1].node_coord()[match1]
            T21 = Transform.fit_to_projection(yx2, yx1, projection=projection)[0]

        # Compose the transformation with the already computed one
        T10 = transformations[i1]
        T20 = T21.compose(T10)
        transformations[i2] = T20

    return list(transformations[_] for _ in range(n_graph))


########################################################################################################################


@dataclass
class TreeRegistrationResult:
    tree1: VTree
    tree2: VTree
    fundus1: FundusData
    fundus2: FundusData
    T12: Transform
    node_match: IntPairArray
    branch_match: dict[tuple[int], tuple[int]]

    def inspect(
        self,
        common_tree: bool = False,
        *,
        draw_od_mac: bool = False,
        split_spheric_projection: bool = True,
        label: bool = True,
        height=600,
        draw_deformation_grid: bool = False,
        draw_boundaries: bool = False,
    ) -> Mosaic:
        import time

        from jppype.utils.color import colormap_by_name

        from ..utils.jppype import Mosaic, draw_tree

        t0 = time.perf_counter()

        m = Mosaic(2, cell_height=height)
        cmap = colormap_by_name("catppuccin-latte")
        N = len(self.node_match)

        if isinstance(self.T12, RadialToRadialTransform) and split_spheric_projection:
            T12, T21 = self.T12.split_spheric_projections()
        else:
            T12 = self.T12
            T21 = IdentityTransform()
        domain1 = T12.transform_domain(Rect.from_size(self.fundus1.shape))
        domain2 = T21.transform_domain(Rect.from_size(self.fundus2.shape))
        full_domain = domain1 | domain2

        # === DRAW FUNDUS ===
        print(f"({(t1 := time.perf_counter()) - t0:.2f}s) drawing fundus...")
        roi1, roi2 = fundus_roi_overlap(self.fundus1, self.fundus2, self.T12, extend_roi=75)
        fundus1 = self.fundus1.image.transpose((1, 2, 0)) * (0.5 * roi1[:, :, None] + 0.5)
        fundus2 = self.fundus2.image.transpose((1, 2, 0)) * (0.5 * roi2[:, :, None] + 0.5)
        print(f"   warp ROI done in ({(t2 := time.perf_counter()) - t1:.2f}s)")
        m[0].domain = full_domain
        m[0].add_image(T12.warp(fundus1, warped_domain=full_domain)[0], "fundus").domain = full_domain
        m[1].domain = full_domain
        m[1].add_image(T21.warp(fundus2, warped_domain=full_domain)[0], "fundus").domain = full_domain
        print(f"   warp fundus done in ({time.perf_counter() - t2:.2f}s)")

        # === DRAW OD AND MACULA ===
        if draw_od_mac:
            od_mac1 = (1 * self.fundus1.od + 2 * self.fundus1.macula).astype(np.uint8)
            m[0].add_label(
                T12.warp(od_mac1, warped_domain=full_domain)[0],
                "OD",
                {1: "red", 2: "green"},
                opacity=0.8,
            ).domain = full_domain
            m[0].add_graph(
                np.empty((0, 2)),
                T12.transform([self.fundus1.od_center, self.fundus1.macula_center]),
                name="OD",
            )
            print("OD MAC 1:", T12.transform([self.fundus1.od_center, self.fundus1.macula_center]))

            od_mac2 = (1 * self.fundus2.od + 2 * self.fundus2.macula).astype(np.uint8)
            m[1].add_label(
                T21.warp(od_mac2, warped_domain=full_domain)[0],
                "OD",
                {1: "red", 2: "green"},
                opacity=0.8,
            ).domain = full_domain
            m[1].add_graph(
                np.empty((0, 2)),
                T21.transform([self.fundus2.od_center, self.fundus2.macula_center]),
                name="OD",
            )
            print("OD MAC 2:", T21.transform([self.fundus2.od_center, self.fundus2.macula_center]))

        # === DRAW TREES ===
        if common_tree:
            tree1, tree2, branch_match1, branch_match2 = self.common_tree()
            branch_match1 = np.split(np.arange(tree1.branch_count), np.unique(branch_match1, return_index=True)[1])
            branch_match2 = np.split(np.arange(tree2.branch_count), np.unique(branch_match2, return_index=True)[1])
        else:
            node_match1: Int1DArray
            node_match2: Int1DArray
            node_match1, node_match2 = self.node_match.T  # type: ignore

            tree1 = self.tree1.reindex_nodes(node_match1, inverse_lookup=True)
            tree2 = self.tree2.reindex_nodes(node_match2, inverse_lookup=True)
            branch_match1 = self.branch_match.keys()
            branch_match2 = self.branch_match.values()

        print(f"({(t3 := time.perf_counter()) - t0:.2f}s) drawing trees...")
        tree1.transform(T12, inplace=True)
        layer1 = draw_tree(tree1, view=m[0], node_labels=label, edge_labels=label)
        layer1.nodes_cmap = {None: cmap} | {n: "#555555" for n in range(N, len(tree1.node_attr))}
        layer1.nodes_labels = {i: str(i) for i in range(N)}

        tree2.transform(T21, inplace=True)
        layer2 = draw_tree(tree2, view=m[1], branch_color="av", node_labels=label, edge_labels=label)
        layer2.nodes_cmap = {None: cmap} | {n: "#555555" for n in range(N, len(tree2.node_attr))}
        layer2.nodes_labels = {i: str(i) for i in range(N)}
        print(f"   warp trees done in ({(t4 := time.perf_counter()) - t3:.2f}s)")

        branch_labels1, branch_cmap1 = {}, {}
        branch_labels2, branch_cmap2 = {}, {}
        for i, (b1s, b2s) in enumerate(zip(branch_match1, branch_match2, strict=False)):
            for i1, b in enumerate(b1s):
                branch_labels1[b] = str(i + 1) + ("abcdefgh"[i1] if len(b1s) > 1 else "")
                branch_cmap1[int(b)] = cmap[i % len(cmap)]
            for i2, b in enumerate(b2s):
                branch_labels2[b] = str(i + 1) + ("abcdefgh"[i2] if len(b2s) > 1 else "")
                branch_cmap2[int(b)] = cmap[i % len(cmap)]
        layer1.edges_cmap = {None: "#555555"} | branch_cmap1
        layer1.edges_labels = branch_labels1
        layer2.edges_cmap = {None: "#555555"} | branch_cmap2
        layer2.edges_labels = branch_labels2

        # === DRAW GRID ===
        if draw_deformation_grid:
            print(f"({(t5 := time.perf_counter()) - t0:.2f}s) drawing grid...")
            grid, grid_domain = T12.draw_grid(Rect.from_size(self.fundus1.shape))
            m[0].add_label(grid * 1, "deformation_grid", "white", opacity=0.5).domain = grid_domain
            if not T21.is_identity():
                grid, grid_domain = T21.draw_grid(Rect.from_size(self.fundus2.shape))
                m[1].add_label(grid * 1, "deformation_grid", "white", opacity=0.5).domain = grid_domain
            print(f"   warp grid done in ({time.perf_counter() - t5:.2f}s)")

        # === DRAW BOUNDARIES ===
        if draw_boundaries:
            print(f"({(t6 := time.perf_counter()) - t0:.2f}s) drawing boundaries...")
            bound1 = tree1.geometric_data().skeleton_label_map(skeleton=False, boundaries=True, interpolate=True) > 0
            bound2 = tree2.geometric_data().skeleton_label_map(skeleton=False, boundaries=True, interpolate=True) > 0
            bound = 1 * full_domain.crop_pad_image(bound1, -domain1.top_left)
            bound += 2 * full_domain.crop_pad_image(bound2, -domain2.top_left)
            m[0].add_label(bound, "boundaries", {1: "green", 2: "red", 3: "white"}, opacity=0.8).domain = full_domain
            m[1].add_label(bound, "boundaries", {2: "green", 1: "red", 3: "white"}, opacity=0.8).domain = full_domain
            print(f"   draw boundaries done in ({time.perf_counter() - t6:.2f}s)")

        return m

    def common_tree(self) -> tuple[VTree, VTree, Int1DArray, Int1DArray]:
        out_trees = []
        out_branch_match = []
        for tree, node_match, branch_match in [
            (self.tree1, self.node_match[:, 0], self.branch_match.keys()),
            (self.tree2, self.node_match[:, 1], self.branch_match.values()),
        ]:
            tree: VTree
            tree = tree.reindex_nodes(node_match, inverse_lookup=True)

            branch_lookup = []
            for i, bs in enumerate(branch_match):
                branch_lookup += [np.stack([list(bs), np.full(len(bs), i)], axis=1)]
            branch_lookup = np.concatenate(branch_lookup, axis=0)

            tree = tree.reindex_branches(branch_lookup[:, 0], inverse_lookup=True)
            tree = tree.delete_branch(np.arange(len(branch_lookup), tree.branch_count))

            nodes_to_fuse, incident_branches = tree.passing_nodes_with_branch_index(exclude_loop=True)
            _, _, branch_merge_lookup, branches_to_delete = tree._merge_consecutive_branches(
                incident_branches, nodes_to_fuse, remove_orphan_nodes=True
            )
            tree = simplify_passing_nodes(tree, not_fusable=np.arange(len(node_match)))
            branch_lookup = np.delete(branch_lookup[:, 1], branches_to_delete)

            out_trees.append(tree)
            out_branch_match.append(branch_lookup)

        return tuple(out_trees) + tuple(out_branch_match)  # type: ignore

    def refine_transform(self, same_k: bool = False, verbose=False) -> tuple[Transform, float]:
        # TODO: use matched branch curvatures roots as additional matching points to refine the transformation
        # TODO: visualisation FundusProjection.draw_grid(domain: Rect, subdivision: int | tuple(int, int), resolution: float) -> tuple[Bool2DArray, Rect]
        src = self.tree1.node_coord()[self.node_match[:, 0]]
        dst = self.tree2.node_coord()[self.node_match[:, 1]]
        # return AffineProjection.fit(src, dst)
        return RadialToRadialTransform.fit(
            src=src,
            dst=dst,
            center_src=np.array(self.fundus1.shape) / 2,
            center_dst=np.array(self.fundus2.shape) / 2,
            same_k=same_k,
            verbose=verbose,
        )


def naive_register_trees(
    tree1, tree2, fundus1: FundusData, fundus2: FundusData, *, max_adj_branch: int = 5, match_max_distance: float = 100
) -> TreeRegistrationResult:
    if not fundus1.has_od_center or fundus1.od_center is None or not fundus2.has_od_center or fundus2.od_center is None:
        raise NotImplementedError("Current implementation of tree registration requires OD centers.")

    # === 1. Estimate transformation and  ROI Overlap ===
    # TODO: extend the following method to work with the tree structure only
    T12, error = od_macula_registration(fundus1, fundus2, only_translation=None)
    # roi1, roi2 = fundus_roi_overlap(fundus1, fundus2, T12, extend_roi=75)

    # Exclude the OD from the ROI to avoid registering the junctions inside
    # roi1 &= ~(distance_transform_edt(fundus1.od) > fundus1.od_diameter * 0.15)
    # roi2 &= ~(distance_transform_edt(fundus2.od) > fundus2.od_diameter * 0.15)

    # === 2. Find candidates for bifurcations matching ===
    def get_bifurcations(tree, T: Transform, fundus: FundusData) -> tuple[Int1DArray, FloatPairArray]:
        """Get the bifurcations of the tree that are in the ROI as well as their coordinates."""
        biff = np.where((tree.node_outdegree() > 1) & (tree.node_indegree() == 1))[0]
        biff_yx = tree.geometric_data().node_coord(biff).astype(np.int_)
        biff_r = np.linalg.norm(T.transform(biff_yx) - (Point(*fundus.shape) / 2).numpy(), axis=1)
        biff_in_roi = biff_r < (fundus.shape[0] / 2 + fundus.od_diameter * 0.15)
        return biff[biff_in_roi], biff_yx[biff_in_roi]  # type: ignore

    # Find matching candidates for bifurcations based on their distance after transformation
    biff1, biff1_yx = get_bifurcations(tree1, T12, fundus2)
    biff2, biff2_yx = get_bifurcations(tree2, T12.invert(), fundus1)
    B1, B2 = len(biff1), len(biff2)
    dist = np.linalg.norm(T12.transform(biff1_yx)[:, None, :] - biff2_yx[None, :, :], axis=2)
    match_candidates = np.argwhere(dist < match_max_distance)

    # Only keep candidates with the same AV label, if available
    if "av" in tree1.node_attr.columns and "av" in tree2.node_attr.columns:
        av1 = tree1.node_attr.loc[biff1, "av"].to_numpy()
        av2 = tree2.node_attr.loc[biff2, "av"].to_numpy()
        match_candidates = match_candidates[av1[match_candidates[:, 0]] == av2[match_candidates[:, 1]]]

    match_mask = np.zeros((B1, B2), dtype=bool)
    match_mask[match_candidates[:, 0], match_candidates[:, 1]] = True

    # === 3. Match bifurcations and incident branches ===
    desc1 = junction_adjacent_branches_descriptor(tree1, biff1)
    desc2 = junction_adjacent_branches_descriptor(tree2, biff2)

    sim, branches_match, n_iter = incident_branches_similarity(
        dot_features_1=desc1.dot_features_block(max_adj_branch),
        dot_features_2=desc2.dot_features_block(max_adj_branch),
        l2_features_1=desc1.scalar_features_block(max_adj_branch),
        l2_features_2=desc2.scalar_features_block(max_adj_branch),
        l2_features_std=desc1.scalar_features_std,
        n_adjacent_branches_1=desc1.adj_branch_count,
        n_adjacent_branches_2=desc2.adj_branch_count,
        branch_uvector_1=desc1.adj_branch_tan_block(max_adj_branch),
        branch_uvector_2=desc2.adj_branch_tan_block(max_adj_branch),
        matchable_nodes=match_mask,
    )

    biff1_hist = tree_node_histogram(tree1, biff1)
    biff2_hist = tree_node_histogram(tree2, biff2)
    all_hist = np.concatenate([biff1_hist, biff2_hist])
    all_hist = all_hist[all_hist > 0]
    hist_norm = np.percentile(all_hist, 90)
    biff1_hist = np.minimum(biff1_hist / hist_norm, 1)
    biff2_hist = np.minimum(biff2_hist / hist_norm, 1)
    node_sim = (biff1_hist[:, None] * biff2_hist[None, :]).sum(axis=2) * match_mask
    sim += node_sim

    node_match_ = hungarian(sim, B1, B2, unmatch1=np.full(B1, 0.01), unmatch2=np.full(B2, 0.01))
    node_match: IntPairArray = np.argwhere(node_match_)  # type: ignore
    dirty_branch_match_ = []
    for n1, n2 in node_match:
        branch_match = branches_match[n1][n2]
        branch_match[:, 0] = desc1.adj_branch_ids[n1][branch_match[:, 0]]
        branch_match[:, 1] = desc2.adj_branch_ids[n2][branch_match[:, 1]]
        dirty_branch_match_.append(branch_match)
    dirty_branch_match: IntPairArray = np.concatenate(dirty_branch_match_, axis=0)  # type: ignore

    node_match[:, 0] = biff1[node_match[:, 0]]
    node_match[:, 1] = biff2[node_match[:, 1]]
    node_match, branch_match = branch_matching_cleanup(tree1, tree2, node_match, dirty_branch_match)

    return TreeRegistrationResult(
        tree1=tree1,
        tree2=tree2,
        fundus1=fundus1,
        fundus2=fundus2,
        T12=T12,
        node_match=node_match,
        branch_match=branch_match,
    )


def branch_matching_cleanup(
    graph1: VTree | IntPairArray,
    graph2: VTree | IntPairArray,
    node_match: IntPairArray,
    dirty_branch_match: IntPairArray,
    directed_edge: Optional[bool] = None,
    extend_peripheral_branch: bool = True,
) -> tuple[IntPairArray, dict[tuple[int], tuple[int]]]:
    """Clean up the branch matching by removing conflicting one-to-one matches in dirty_branch_match and replace them by many-to-many matches. The output dictionary is a many-to-many matching pairing branches of the first graph to branches of the second graph based on the shortest path to connect matched nodes.

    Parameters
    ----------
    graph1 : VGraph | IntPairArray
        The first vascular graph or its branch list.

    graph2 : VGraph | IntPairArray
        The second vascular graph or its branch list.

    node_match : IntPairArray
        The node matching as a 2D array of shape (n_match, 2) where n_match is the number of matched nodes. The first column contains the node indices in the first graph and the second column contains the node indices in the second graph.

    dirty_branch_match : IntPairArray
        The initial branch matching as a 2D array of shape (n_match, 2) where n_match is the number of matched nodes. The first column contains the branch indices in the first graph and the second column contains the branch indices in the second graph. This matching can contain conflicting one-to-one matches that need to be cleaned up.

    Returns
    -------
    node_match: IntPairArray
        A version of node_match where node matches inducing non-resolvable conflicts of branch matches have been removed.

    branch_match: dict[list[int], list[int]]
        A many-to-many matching pairing branches of the first graph to branches of the second graph based on the shortest path to connect matched nodes. The keys are tuples of branch indices in the first graph and the values are lists of branch indices in the second graph that are matched to the key branch.
    """  # noqa: E501
    if isinstance(graph1, VGraph):
        edge_list1 = graph1.tree_branch_list() if isinstance(graph1, VTree) else graph1.branch_list
        node_count1 = graph1.node_count
    else:
        edge_list1 = graph1
        node_count1 = np.max(edge_list1) + 1
    if isinstance(graph2, VGraph):
        edge_list2 = graph2.tree_branch_list() if isinstance(graph2, VTree) else graph2.branch_list
        node_count2 = graph2.node_count
    else:
        edge_list2 = graph2
        node_count2 = np.max(edge_list2) + 1
    if directed_edge is None:
        directed_edge = isinstance(graph1, VTree) and isinstance(graph2, VTree)

    # Remove duplicates
    dirty_branch_match = np.unique(dirty_branch_match, axis=0)  # type: ignore

    # One to one match
    _, m1_idx, m1_count = np.unique(dirty_branch_match[:, 0], return_counts=True, return_index=True)
    _, m2_idx, m2_count = np.unique(dirty_branch_match[:, 1], return_counts=True, return_index=True)
    valid_one_to_one = np.zeros_like(dirty_branch_match, dtype=bool)
    valid_one_to_one[m1_idx[m1_count == 1], 0] = True
    valid_one_to_one[m2_idx[m2_count == 1], 1] = True
    one_to_one_matches = dirty_branch_match[np.all(valid_one_to_one, axis=1)]

    node_match1, node_match2 = node_match.T

    # Shortest path matches
    shortest1, backtrack1 = shortest_secondary_path(edge_list1, node_match1, node_count1, directed_edge)
    shortest2, backtrack2 = shortest_secondary_path(edge_list2, node_match2, node_count2, directed_edge)

    adj_matched_nodes = np.argwhere((shortest1[:, node_match1] > 0) & (shortest2[:, node_match2] > 0))
    adj_nodes1 = adj_matched_nodes.copy()
    adj_nodes1[:, 1] = node_match1[adj_nodes1[:, 1]]
    adj_nodes2 = adj_matched_nodes.copy()
    adj_nodes2[:, 1] = node_match2[adj_nodes2[:, 1]]

    branch_paths1 = backtrack_edges(adj_nodes1, backtrack1, node_match1)
    branch_paths2 = backtrack_edges(adj_nodes2, backtrack2, node_match2)

    b1_paired = np.zeros(edge_list1.shape[0], dtype=bool)
    b2_paired = np.zeros(edge_list2.shape[0], dtype=bool)

    branch_match = {}
    for p1, p2 in zip(branch_paths1, branch_paths2, strict=True):
        if b1_paired[p1].any() or b2_paired[p2].any():
            print(f"Conflict pairing {p1} with {p2}. Ignoring this match.")
            continue
        branch_match[tuple(p1)] = tuple(p2)
        b1_paired[p1] = True
        b2_paired[p2] = True
        # TODO: add checks?

    for b1, b2 in one_to_one_matches:
        if (b1,) in branch_match:
            if branch_match[(b1,)] != (b2,):
                print(f"Conflict pairing {b1} to {b2}: already matched {branch_match[(b1,)]}. Ignoring this match.")
            continue
        if b1_paired[b1] or b2_paired[b2]:
            print(f"Conflict for branch {b1} in graph1 and branch {b2} in graph2. Ignoring this match.")
            continue
        b1, b2 = (int(b1),), (int(b2),)
        b1_paired[b1] = True
        b2_paired[b2] = True
        # if extend_peripheral_branch:
        #     assert isinstance(graph1, VTree) and isinstance(graph2, VTree)
        #     d1 = graph1.branch(b1[0]).node_to_node_length()
        #     d2 = graph2.branch(b2[0]).node_to_node_length()
        #     if d1 > d2 * 1.2:
        #         b2s = [b2[0]]
        #         while d1 > d2 * 1.2 and len(succs := [graph2.branch(b2s[-1]).adjacent_branch_ids) > 0:
        #             if len(succs) > 1:
        #                 succ = succs[0]
        #             else:
        #                 # TODO: compute similarity between b1 and the succesors of b2s[-1]
        #                 succ = succs[0]
        #             b2s.append(succ)
        #             d2 += graph2.branch(succ).node_to_node_length()
        #         b2 = tuple(int(_) for _ in b2s)
        #     elif d2 > d1 * 1.2:
        #         b1s = [b1[0]]
        #         while d2 > d1 * 1.2 and len(succs := graph1.branch(b1s[-1]).successors_ids) > 0:
        #             if len(succs) > 1:
        #                 succ = succs[0]
        #             else:
        #                 # TODO: compute similarity between b1 and the succesors of b1s[-1]
        #                 succ = succs[0]
        #             b1s.append(succ)
        #             d1 += graph1.branch(succ).node_to_node_length()
        #         b1 = tuple(int(_) for _ in b1s)
        branch_match[b1] = b2

    return node_match, branch_match


def od_macula_registration(
    fundus_src: FundusData,
    fundus_dst: FundusData,
    *,
    only_translation: None | bool = None,
) -> tuple[SimilarityTransform, float]:
    """Compute the transformation to align two fundus images based on their optic disc and macula centers if available.
    If both fundus have OD and macula centers, use a similarity transformation to align them, otherwise use a simple translation.

    Parameters
    ----------
    fundus_src : FundusData
        The source fundus to register.

    fundus_dst : FundusData
        The destination fundus to register.

    only_translation : None | bool, optional
        Whether to use only a translation to align the fundus images:
            - if ``True`` use a simple translation;
            - if ``False``, use a similarity transformation if both fundus have macula centers, otherwise raise an error;
            - If ``None`` (by default), use a similarity transformation if both fundus have macula centers, otherwise use a translation.

    Returns
    -------
    transform: SimilarityTransform
        The transformation to apply to the source fundus to align it to the destination fundus.

    mse: float
        The mean squared error between the transformed source fundus and the destination fundus based on the distance between the OD and macula centers after transformation.
    """  # noqa: E501
    if (
        not fundus_src.has_od_center
        or fundus_src.od_center is None
        or not fundus_dst.has_od_center
        or fundus_dst.od_center is None
    ):
        raise NotImplementedError("Current implementation of OD-Macula registration requires OD centers.")
    od_src = fundus_src.od_center
    od_dst = fundus_dst.od_center
    mac_src = fundus_src.macula_center if fundus_src.has_macula_center else None
    mac_dst = fundus_dst.macula_center if fundus_dst.has_macula_center else None

    if only_translation is False and (mac_src is None or mac_dst is None or od_src is None or od_dst is None):
        raise ValueError(
            "Only translation can be used for OD-Macula registration when one of the fundus does not have a macula center."  # noqa: E501
        )

    if mac_src is None or mac_dst is None:
        return Translation.fit(od_src, od_dst)
    elif od_src is None or od_dst is None:
        return Translation.fit(mac_src, mac_dst)
    elif only_translation is True:
        # return Translation.fit([od_src, mac_src], [od_dst, mac_dst])
        return Translation.fit(od_src, od_dst)
    else:
        return SimilarityTransform.fit([od_src, mac_src], [od_dst, mac_dst])


def fundus_roi_overlap(
    fundus1: FundusData,
    fundus2: FundusData,
    T12: Transform,
    *,
    extend_roi: int = 0,
) -> tuple[Bool2DArray, Bool2DArray]:
    """Compute the overlap between the fundus ROI masks of two fundus based on the provided transformation.

    Parameters
    ----------
    fundus1 : FundusData
        The first fundus to compare.

    fundus2 : FundusData
        The second fundus to compare.

    T12 : FundusProjection
        The transformation from the first fundus to the second fundus.

    extend_roi : int, optional
        The distance in pixels to extend the ROI masks before computing the overlap. By default 0 (no extension).

    Returns
    -------
    tuple[Bool2DArray, Bool2DArray]
        The overlap between the fundus ROI masks of the two fundus as two boolean arrays of shape (H, W) corresponding to the ROI masks of the first and second fundus, respectively.
    """  # noqa: E501
    assert fundus1.roi_mask is not None and fundus2.roi_mask is not None, (
        "Both fundus must have an ROI mask to compute the overlap."
    )
    roi1 = fundus1.roi_mask
    roi2 = fundus2.roi_mask

    if extend_roi > 0:
        roi1 = distance_transform_edt(~roi1) < extend_roi
        roi2 = distance_transform_edt(~roi2) < extend_roi

    roi1 = roi1 & (T12.invert().warp(roi2 * np.uint8(255), warped_domain="same")[0] > 127)
    roi2 = roi2 & (T12.warp(roi1 * np.uint8(255), warped_domain="same")[0] > 127)

    return roi1, roi2  # type: ignore
