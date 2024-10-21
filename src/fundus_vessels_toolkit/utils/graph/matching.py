from typing import Optional

import numpy as np
import torch

from ..cpp_extensions.fvt_cpp import nodes_similarity as nodes_similarity_cpp
from ..torch import autocast_torch


@autocast_torch
def incident_branches_similarity(
    angle_features_1: torch.Tensor,
    angle_features_2: torch.Tensor,
    scalar_features_1: torch.Tensor,
    scalar_features_2: torch.Tensor,
    scalar_features_std: torch.Tensor,
    n_incident_branches_1: torch.Tensor,
    n_incident_branches_2: torch.Tensor,
    branch_uvector_1: Optional[torch.Tensor] = None,
    branch_uvector_2: Optional[torch.Tensor] = None,
    matchable_nodes: Optional[torch.Tensor] = None,  # noqa: F821
) -> torch.Tensor:
    """
    Computes the similarity between two sets of nodes.

    Parameters
    ----------
    cos_features_1 : torch.Tensor
        Tensor of shape (N1, B, F, 2) containing F angular features of the first set of nodes N1.

    cos_features_2 : torch.Tensor
        Tensor of shape (N2, B, F, 2) containing F angular features of the second set of nodes N2.

    L2_features_1 : torch.Tensor
        Tensor of shape (N1, B, F) containing F L2 features of the first set of nodes N1.

    L2_features_2 : torch.Tensor
        Tensor of shape (N2, B, F) containing F L2 features of the second set of nodes N2.

    L2_features_std : torch.Tensor
        Tensor of shape (N1, B, F) containing the standard deviation of the L2 features of the first set of nodes N1.

    branch_uvector_1 : torch.Tensor, optional
        Tensor of shape (N1, B, 2) containing the direction of the incident branches seen from their corresponding node.
        If a branch direction is pointing towards +/-15° the branch may be matched with the first or the last branch of the other node.

    branch_uvector_2 : torch.Tensor, optional
        Tensor of shape (N2, B, 2) containing the direction of the incident branches seen from their corresponding node.

    matchable_nodes : torch.Tensor, optional
        Tensor of shape (N1, N2) containing the matchable nodes. If None, all nodes are matchable. By default None.

    Returns
    -------
    torch.Tensor
        The similarity matrix of shape (n_nodes1, n_nodes2).

    torch.Tensor
    """  # noqa: E501

    if matchable_nodes is None:
        N1, N2 = angle_features_1.shape[0], angle_features_2.shape[0]
        matchable_nodes = torch.ones(N1, N2, dtype=torch.bool)

    empty = torch.empty([])
    sim, branches_matches, n_iter = nodes_similarity_cpp(
        matchable_nodes.cpu().bool(),
        angle_features_1.cpu().float(),
        angle_features_2.cpu().float(),
        scalar_features_1.cpu().float(),
        scalar_features_2.cpu().float(),
        scalar_features_std.cpu().float(),
        n_incident_branches_1.cpu().int(),
        n_incident_branches_2.cpu().int(),
        empty if branch_uvector_1 is None else branch_uvector_1.cpu().float(),
        empty if branch_uvector_2 is None else branch_uvector_2.cpu().float(),
        False,
    )
    return sim, branches_matches, n_iter


@autocast_torch
def incident_branches_similarity_rotation_invariant(
    angle_features_1: torch.Tensor,
    angle_features_2: torch.Tensor,
    scalar_features_1: torch.Tensor,
    scalar_features_2: torch.Tensor,
    scalar_features_std: torch.Tensor,
    n_incident_branches_1: torch.Tensor,
    n_incident_branches_2: torch.Tensor,
    branch_uvector_1: torch.Tensor,
    branch_uvector_2: torch.Tensor,
    matchable_nodes: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Computes the similarity between two sets of nodes.

    Parameters
    ----------
    nodes1 : torch.Tensor
        The first set of nodes.

    nodes2 : torch.Tensor
        The second set of nodes.

    similarity : str, optional
        The similarity to compute. Either "cosine" or "L2". By default "cosine".

    Returns
    -------
    torch.Tensor
        The similarity matrix of shape (n_nodes1, n_nodes2).
    """
    if matchable_nodes is None:
        N1, N2 = angle_features_1.shape[0], angle_features_2.shape[0]
        matchable_nodes = torch.ones(N1, N2, dtype=torch.bool)

    return nodes_similarity_cpp(
        matchable_nodes.cpu().bool(),
        angle_features_1.cpu().float(),
        angle_features_2.cpu().float(),
        scalar_features_1.cpu().float(),
        scalar_features_2.cpu().float(),
        scalar_features_std.cpu().float(),
        n_incident_branches_1.cpu().int(),
        n_incident_branches_2.cpu().int(),
        branch_uvector_1.cpu().float(),
        branch_uvector_2.cpu().float(),
        True,
    )


def euclidien_matching(
    yx1: np.ndarray,
    yx2: np.ndarray,
    max_matching_distance: Optional[float] = None,
    min_distance: Optional[float] = None,
    density_sigma: Optional[float] = None,
    gamma: float = 0,
    matchable: Optional[np.ndarray] = None,
    return_distance: bool = False,
) -> np.ndarray:
    """
    Match nodes from two graphs based on the euclidean distance between nodes.

    Each node from the first graph is matched to the closest node from the second graph, if their distance is below
      max_matching_distance. When multiple nodes from both graph are near each other, minimize the sum of the distance
      between matched nodes.

    This implementation use the Hungarian algorithm to maximize the sum of the inverse of the distance between
        matched nodes.


    Parameters
    ----------
        node1_yx: np.ndarray
            The (y,x) coordinates of the nodes of the first graph as a matrix of shape (n1, 2).

        node2_yx: np.ndarray
            The (y,x) coordinates of the nodes of the second graph as a matrix of shape (n2, 2).

        max_matching_distance: float, optional
            maximum distance between two nodes to be considered as a match.

        min_distance: float, optional
            If not None, set a lower bound to the distance between two nodes to prevent immediate match of superposed nodes.

        gamma: float, optional
            If > 0, the penalty of large distances increase exponentially with a factor gamma.
            If 0 (default), the algorithm simply minimize the sum of the distance between matched nodes.

        matchable: np.ndarray, optional
            A boolean matrix of shape (n1, n2) where True indicate that the nodes are matchable. If None, all nodes are
            considered matchable.

        return_distance:
            if True, return the distance between each matched nodes.
    Returns:
        A np.ndarray of shape (2, n_match), where n_match is the number of matched nodes. The first row contains the
            indexes of the matched nodes from the first graph, while the second row contains the indexes of the matched
            nodes from the second graph.

        If return_distance is True, returns ((node1_matched, node2_matched), nodes_distance), where nodes_distance
          is a vector of the same length as node1_matched and node2_matched, and encode the distance between each
          matched nodes.
    """  # noqa: E501
    from pygmtools.linear_solvers import hungarian

    n1 = len(yx1)
    n2 = len(yx2)

    # Compute the euclidean distance between each node
    cross_euclidean_distance = np.linalg.norm(yx1[:, None] - yx2[None, :], axis=2)
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
    elif max_matching_distance is not None:
        if min_distance is None:
            min_distance = np.clip(max_matching_distance / 20, 1)

        # Compute node density as the sum of the Gaussian kernel centered on each node
        def influence(x):
            return np.exp(-(x**2) / (2 * density_sigma**2))

        self_euclidean_distance1 = np.linalg.norm(yx1[:, None] - yx1[None, :], axis=2)
        self_euclidean_distance2 = np.linalg.norm(yx2[:, None] - yx2[None, :], axis=2)
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
    else:
        min_weight1 = None
        min_weight2 = None

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


########################################################################################################################
#   === UTILITY FUNCTIONS ===
########################################################################################################################
def ensure_consistent_matches(matches: np.ndarray, return_valid_index: bool = False) -> np.ndarray:
    """
    Ensure that the matches are consistent, i.e. no match is duplicated and each element is matched at most once.

    Any invalid match is removed.

    Parameters
    ----------
    matches : np.ndarray
        The matches as a matrix of shape (2, n_matches).

    return_valid_index : bool, optional
        If True, also return the index of the valid matches in the original matches array. By default False.

    Returns
    -------
    np.ndarray
        The consistent matches as a matrix of shape (2, n_matches).

    np.ndarray
        The index of the valid matches in the original matches array if return_valid

    """
    # Remove duplicate matches
    matches, matches_id = np.unique(np.asarray(matches), axis=1, return_index=True)

    # Ensure that matches are consistent: each element can be matched at most once
    _, m1_index, m1_count = np.unique(matches[0], return_counts=True, return_index=True)
    _, m2_index, m2_count = np.unique(matches[1], return_counts=True, return_index=True)
    unique_matches = np.zeros_like(matches, dtype=bool)
    unique_matches[0, m1_index[m1_count == 1]] = True
    unique_matches[1, m2_index[m2_count == 1]] = True
    valid_matches = np.all(unique_matches, axis=0)

    # Remove invalid matches
    matches = matches[:, valid_matches]

    if not return_valid_index:
        return matches
    else:
        matches_id = matches_id[valid_matches]
        return matches, matches_id
