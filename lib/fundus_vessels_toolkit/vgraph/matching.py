import numpy as np


def simple_graph_matching(node1_yx: tuple[np.ndarray, np.ndarray], node2_yx: tuple[np.ndarray, np.ndarray],
                          max_matching_distance: float | None = None, return_distance: bool = False, gamma: float = 0):
    """
    Match nodes from two graphs based on the euclidian distance between nodes.

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
        return_distance: if True, return the distance between each matched nodes.
        gamma: parameter to exponentially increase the penalty of large distance.
            If 0 (default), the algorithm simply minimize the sum of the distance between matched nodes.
    Returns:
        A tuple (node1_matched, node2_matched), where node1_matched and node2_matched are vectors of the same length.
          Every (node1_matched[i], node2_matched[i]) encode a match, node1_matched being the index of a node from the
          first graph and node2_matched the index of the corresponding node from the second graph.

        If return_distance is True, returns ((node1_matched, node2_matched), nodes_distance), where nodes_distance
          is a vector of the same length as node1_matched and node2_matched, and encode the distance between each
          matched nodes.
    """
    from pygmtools.linear_solvers import hungarian
    if isinstance(node1_yx, tuple) and len(node1_yx) == 2:
        node1_yx = np.stack(node1_yx, axis=1)
    if isinstance(node2_yx, tuple) and len(node2_yx) == 2:
        node2_yx = np.stack(node2_yx, axis=1)
    n1 = len(node1_yx)
    n2 = len(node2_yx)

    # Compute the euclidian distance between each node
    euclidian_distance = np.linalg.norm(node1_yx[:, None] - node2_yx[None, :], axis=2)
    max_distance = euclidian_distance.max()
    def dist2weight(dist):
        return np.exp(-gamma * dist / 1e3) if gamma != 0 else max_distance - dist

    # Compute the weight as the distance inverse (the hungarian method maximise the sum of the weight)
    weight = dist2weight(euclidian_distance)

    # Set the cost of unmatch nodes to half the inverse of the maximum distance,
    #  so that nodes separated by more than max_distance are better left unmatched.
    min_weight = dist2weight(max_matching_distance)/2 if max_matching_distance is not None else 0

    # Compute the hungarian matching
    matched_nodes = hungarian(weight[None, ...], [n1], [n2], np.repeat([[min_weight]], n1, axis=1),
                              np.repeat([[min_weight]], n2, axis=1))[0]
    matched_nodes = np.where(matched_nodes)

    if return_distance:
        return matched_nodes, euclidian_distance[matched_nodes]
    return matched_nodes
