from typing import Iterable, Optional, Tuple

import numpy as np

from ..utils.fundus_projections import FundusProjection, ransac_fit_projection
from ..vascular_data_objects import VGraph
from .descriptor import NodeFeaturesCallback, junction_bspline_descriptor


def junctions_matching(
    vgraph1: VGraph,
    vgraph2: VGraph,
    features: Optional[NodeFeaturesCallback] = None,
    min_weight: float | None = None,
    reindex_graphs: bool = False,
):
    """
    Match nodes from two graphs based on the features computed by the features_function.

    Each node from the first graph is matched to the closest node from the second graph, if their distance is below
      max_matching_distance. When multiple nodes from both graph are near each other, minimize the sum of the distance
      between matched nodes.

    Parameters
    ----------
        vgraph1:
            The first graph to compare.

        vgraph2:
            The second graph to compare.

        features_function:
            The function that computes the features of the nodes.

        max_matching_distance:
            The maximum distance between two nodes to be considered as a match.

        reindex_graphs:
            If True, reorder the nodes of the graphs so that the selected nodes are at the beginning of the list.

    Returns
    -------
        node1_matched, node2_matched:
            Every (node1_matched[i], node2_matched[i]) encode a match, node1_matched being the index of a node from the
        first graph and node2_matched the index of the corresponding node from the second graph. ``node1_matched`` and ``node2_matched`` are vectors of the same length.

    """  # noqa: E501
    from pygmtools.linear_solvers import hungarian

    if features is None:
        features = junction_bspline_descriptor

    cos_f1, l2_f1, junction_id1 = features(vgraph1)
    cos_f2, l2_f2, junction_id2 = features(vgraph2)
    n1 = len(junction_id1)
    n2 = len(junction_id2)

    cos_cross = np.dot(cos_f1, cos_f2.T)
    l2_cross = np.linalg.norm(l2_f1[:, None] - l2_f2[None, :], axis=2)
    l2_cross_normalized = 1 - l2_cross / np.max(l2_cross)

    weight = cos_cross + l2_cross_normalized
    if min_weight is None:
        min_weight = np.percentile(weight, 100 * (1 - min(weight.shape) / np.prod(weight.shape))) / 2
    min_weight1 = np.repeat([min_weight], n1, axis=0)
    min_weight2 = np.repeat([min_weight], n2, axis=0)

    matched_nodes = hungarian(weight, n1, n2, min_weight1, min_weight2)
    matched_nodes = np.where(matched_nodes)

    matched_n1 = junction_id1[matched_nodes[0]]
    matched_n2 = junction_id2[matched_nodes[1]]

    if reindex_graphs:
        vgraph1.reindex_nodes(matched_n1, inverse_lookup=True)
        vgraph2.reindex_nodes(matched_n2, inverse_lookup=True)

    matched_nodes = np.stack([matched_n1, matched_n2], axis=0)
    return matched_nodes


def ransac_refine_node_matching(
    src_graph: VGraph,
    dst_graph: VGraph,
    src_matched_nodes: Iterable[int],
    dst_matched_nodes: Iterable[int],
    *,
    reindex_graphs=False,
    return_mean_error=False,
) -> Tuple[FundusProjection, np.ndarray, float] | Tuple[FundusProjection, np.ndarray]:
    """
    Refine the node matching using the RANSAC algorithm and the nodes coordinates to estimate the geometrical transformation between the two graphs.

    Parameters
    ----------
        src_graph:
            The first graph to compare.

        dst_graph:
            The second graph to compare.

        src_matched_nodes:
            The matched nodes from the first graph.

        dst_matched_nodes:
            The matched nodes from the second graph.

        reindex_graphs:
            If True, reorder the nodes of the graphs so that the selected nodes are at the beginning of the list.

    Returns
    -------
        T: FundusProjection
            The best transformation of type ``final_projection`` found to map points from ``src`` to ``dst``.

        matched_nodes: np.ndarray
            Indices of the matched nodes as a (2, N) array where N is the number of matched nodes, ``matched_nodes[0]`` are the indices of the matched nodes from the first graph and ``matched_nodes[1]`` are the indices of the matched nodes from the second graph.

        error: float
            Mean distance of the best transformation. Only returned if ``return_mean_error`` is True.

    """  # noqa: E501
    assert dst_matched_nodes.ndim == 1, "matched_nodes1 must be a 1D array"
    assert (
        dst_matched_nodes.shape == src_matched_nodes.shape
    ), "matched_nodes1 and matched_nodes2 must have the same shape"

    dst_yx = dst_graph.nodes_coord()[dst_matched_nodes]
    src_yx = src_graph.nodes_coord()[src_matched_nodes]

    T, mean_error, valid_nodes = ransac_fit_projection(
        src_yx, dst_yx, n=3, inliers_tolerance=20, min_inliers=5, max_iterations=200, early_stop_tolerance=5
    )

    dst_matched_nodes = dst_matched_nodes[valid_nodes]
    src_matched_nodes = src_matched_nodes[valid_nodes]

    if reindex_graphs:
        dst_graph.reindex_nodes(dst_matched_nodes, inverse_lookup=True)
        src_graph.reindex_nodes(src_matched_nodes, inverse_lookup=True)

    matched_nodes = np.stack([dst_matched_nodes, src_matched_nodes], axis=0)

    if return_mean_error:
        return T, matched_nodes, mean_error
    return T, matched_nodes


def euclidien_node_matching(
    node1_yx: np.ndarray,
    node2_yx: np.ndarray,
    max_matching_distance: float | None = None,
    min_distance: float | None = None,
    density_sigma: float | None = None,
    gamma: float = 0,
    return_distance: bool = False,
):
    """
    Match nodes from two graphs based on the euclidean distance between nodes.

    Each node from the first graph is matched to the closest node from the second graph, if their distance is below
      max_matching_distance. When multiple nodes from both graph are near each other, minimize the sum of the distance
      between matched nodes.

    This implementation use the Hungarian algorithm to maximize the sum of the inverse of the distance between
        matched nodes.


    Args:
        node1_yx: tuple (y, x), where y and x are vectors of the same length and encode the coordinates of the nodes
                    of the first graph.
        node2_yx: same format as node1_yx but for the second graph.
        max_matching_distance: maximum distance between two nodes to be considered as a match.
        min_distance: set a lower bound to the distance between two nodes to prevent immediate match of superposed nodes.
        gamma: parameter to exponentially increase the penalty of large distance.
            If 0 (default), the algorithm simply minimize the sum of the distance between matched nodes.
        return_distance: if True, return the distance between each matched nodes.
    Returns:
        A tuple (node1_matched, node2_matched), where node1_matched and node2_matched are vectors of the same length.
          Every (node1_matched[i], node2_matched[i]) encode a match, node1_matched being the index of a node from the
          first graph and node2_matched the index of the corresponding node from the second graph.

        If return_distance is True, returns ((node1_matched, node2_matched), nodes_distance), where nodes_distance
          is a vector of the same length as node1_matched and node2_matched, and encode the distance between each
          matched nodes.
    """  # noqa: E501
    from pygmtools.linear_solvers import hungarian

    n1 = len(node1_yx)
    n2 = len(node2_yx)

    # Compute the euclidean distance between each node
    cross_euclidean_distance = np.linalg.norm(node1_yx[:, None] - node2_yx[None, :], axis=2)
    max_distance = cross_euclidean_distance.max()

    if gamma > 0:

        def dist2weight(dist):
            return np.exp(-gamma * dist / 1e3)

    else:

        def dist2weight(dist):
            return max_distance - dist

    # Compute the weight as the distance inverse (the Hungarian method maximise the sum of the weight)
    weight = dist2weight(cross_euclidean_distance)

    # Set the cost of not matched nodes to half the inverse of the maximum distance,
    #  so that nodes separated by more than max_distance are better left unmatched.
    if density_sigma is None:
        min_weight = dist2weight(max_matching_distance) / 2 if max_matching_distance is not None else 0
        min_weight1 = np.repeat([min_weight], n1, axis=0)
        min_weight2 = np.repeat([min_weight], n2, axis=0)
    else:
        if min_distance is None:
            min_distance = np.clip(max_matching_distance / 20, 1)

        # Compute node density as the sum of the Gaussian kernel centered on each node
        def influence(x):
            return np.exp(-(x**2) / (2 * density_sigma**2))

        self_euclidean_distance1 = np.linalg.norm(node1_yx[:, None] - node1_yx[None, :], axis=2)
        self_euclidean_distance2 = np.linalg.norm(node2_yx[:, None] - node2_yx[None, :], axis=2)
        self_density1 = np.clip(influence(self_euclidean_distance1).sum(axis=1) - 1, 0, 1)
        self_density2 = np.clip(influence(self_euclidean_distance2).sum(axis=1) - 1, 0, 1)

        cross_density = influence(cross_euclidean_distance)
        cross_density1 = np.clip(cross_density.sum(axis=1), 0, 1)
        cross_density2 = np.clip(cross_density.sum(axis=0), 0, 1)

        density1 = (self_density1 + cross_density1) / 2
        density2 = (self_density2 + cross_density2) / 2

        # Compute a maximum matching distance per node based on the density of its closest neighbor in the other graph
        closest_node1 = cross_euclidean_distance.argmin(axis=1)
        closest_node2 = cross_euclidean_distance.argmin(axis=0)
        factor1 = 1 - density2[closest_node1]
        factor2 = 1 - density1[closest_node2]

        max_matching_distance1 = np.clip(max_matching_distance * factor1, min_distance, max_matching_distance)
        max_matching_distance2 = np.clip(max_matching_distance * factor2, min_distance, max_matching_distance)

        min_weight1 = dist2weight(max_matching_distance1) / 2
        min_weight2 = dist2weight(max_matching_distance2) / 2

    # Compute the Hungarian matching
    matched_nodes = hungarian(
        weight,
        n1,
        n2,
        min_weight1,
        min_weight2,
    )
    matched_nodes = np.argwhere(matched_nodes).T

    if return_distance:
        return matched_nodes, cross_euclidean_distance[matched_nodes]
    return matched_nodes
