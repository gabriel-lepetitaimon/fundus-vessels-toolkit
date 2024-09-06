import math
from itertools import product as iter_product
from typing import Iterable, Literal, Optional, Type

import numpy as np

from ..utils.fundus_projections import AffineProjection, FundusProjection, QuadraticProjection
from ..vascular_data_objects import VGraph
from .node_matching import match_junctions, ransac_refine_node_matching


def vgraph_registration(
    fix_graph: VGraph,
    moving_graph: VGraph,
    matched_nodes: Optional[dict[int, int]] = None,
    projection: Optional[Type[FundusProjection]] = None,
    register_branches: bool = True,
) -> np.ndarray:
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
            fix_graph, moving_graph, matched_nodes, final_projection=projection, return_mean_error=True
        )
    except ValueError:
        raise ValueError("The provided vascular graph does not overlap enough to be registered together.") from None

    return T_mov_fix


def multi_vgraph_registration(
    vgraphs: Iterable[VGraph],
    projection: Optional[Type[FundusProjection]] = None,
    iterative: bool = False,
    ensure_exact: Literal["direct", "inverse", None] = None,
) -> list[np.ndarray]:
    """
    Register multiple vascular graphs together.

    Parameters
    ----------
    vgraphs : Iterable[VGraph]
        The vascular graphs to register.

    Returns
    -------
    list[np.ndarray]
        The transformations to apply to each vascular graph to register them together.

    Raises
    ------
    ValueError
        If the provided vascular graphs do not overlap enough to be registered together

    """
    import networkx as nx

    if projection is None:
        projection = {3: AffineProjection, 12: QuadraticProjection}

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
    yx0 = vgraphs[root].nodes_coord()
    transformations = {root: FundusProjection.identity()}

    if iterative:
        nodes_weight = {
            i2: matching[(i1, i2)][2] if (i1, i2) in matching else matching[(i2, i1)][2]
            for i1, i2 in nx.bfs_edges(ST, root)
        }
        nodes_weight[root] = 0

        def priority(nodes):
            return sorted(list(nodes), key=lambda x: nodes_weight[x])

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
                adj_transformed_coord = T_adj_0.transform(vgraphs[adj_fundus].nodes_coord()[adj_match])
                # Accumulate the transformed coordinates
                extended_adj_coord[match_id] += adj_transformed_coord
            # Normalize the accumulated coordinates by the number of fundus where each node was matched
            extended_adj_coord /= ext_match_count[:, None]

            # Return the matched nodes coordinates in the current fundus and in the already transformed fundus
            return vgraphs[fundus_id].nodes_coord()[ext_match], extended_adj_coord

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
            T21: FundusProjection = T21.invert()

        # If iterative is true, check if we can recompute the transformation from scratch
        if iterative:
            ext_yx2, ext_yx0 = extended_match_coord(i2)
            if ext_yx2 is not None:
                # If this fundus can be registered to several already transformed fundus, recompute the transformation
                if ensure_exact == "inverse":
                    T02 = FundusProjection.fit_to_projection(ext_yx0, ext_yx2, projection=projection)[0]
                    T20 = T02.invert()
                else:
                    T20 = FundusProjection.fit_to_projection(ext_yx2, ext_yx0, projection=projection)[0]
                transformations[i2] = T20
                continue

        # If the transformation is inexact and we want to ensure exactness, recompute it from scratch
        if ensure_exact == "inverse" and not T21.is_inverse_exact:
            yx2 = vgraphs[i2].nodes_coord()[match2]
            yx1 = vgraphs[i1].nodes_coord()[match1]
            T12 = FundusProjection.fit_to_projection(yx1, yx2, projection=projection)[0]
            T21 = T12.invert()
        elif ensure_exact == "direct" and not T21.is_exact:
            yx2 = vgraphs[i2].nodes_coord()[match2]
            yx1 = vgraphs[i1].nodes_coord()[match1]
            T21 = FundusProjection.fit_to_projection(yx2, yx1, projection=projection)[0]

        # Compose the transformation with the already computed one
        T10 = transformations[i1]
        T20 = T21.compose(T10)
        transformations[i2] = T20

    return list(transformations[_] for _ in range(n_graph))
