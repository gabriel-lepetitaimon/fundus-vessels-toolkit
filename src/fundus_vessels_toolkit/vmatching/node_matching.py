import functools
from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping, Optional, Tuple, Type, TypeAlias

import numpy as np
import pandas as pd
from pygmtools.linear_solvers import hungarian

from ..utils.fundus_projections import AffineProjection, FundusProjection, QuadraticProjection, ransac_fit_projection
from ..utils.graph.matching import ensure_consistent_matches, euclidien_matching, incident_branches_similarity
from ..utils.lookup_array import complete_lookup, invert_lookup
from ..vascular_data_objects import VGraph
from .descriptor import NodeFeaturesCallback, junction_incident_branches_descriptor


########################################################################################################################
#  === NODE MATCHING ===
########################################################################################################################
def match_nodes_by_distance(
    vgraph1: VGraph,
    vgraph2: VGraph,
    max_matching_distance: float | None = None,
    min_distance: float | None = None,
    density_sigma: float | None = None,
    gamma: float = 0,
    *,
    reindex_nodes=False,
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
        vgraph1: VGraph
            The first graph.

        vgraph2: VGraph
            The second graph.


        max_matching_distance: float, optional
            maximum distance between two nodes to be considered as a match.

        min_distance: float, optional
            If not None, set a lower bound to the distance between two nodes to prevent immediate match of superposed nodes.

        gamma:
            If > 0, the penalty of large distances increase exponentially with a factor gamma.
            If 0 (default), the algorithm simply minimize the sum of the distance between matched nodes.

        reindex_nodes: bool, optional
            If True, reorder the nodes of the graphs so that the matched nodes are at the beginning and in the same order for both graphs.
    Returns:

    """  # noqa: E501

    matched_nodes = euclidien_matching(
        vgraph1.node_coord(), vgraph2.node_coord(), max_matching_distance, min_distance, density_sigma, gamma
    )

    if reindex_nodes:
        vgraph1.reindex_nodes(matched_nodes[0], inverse_lookup=True, inplace=True)
        vgraph2.reindex_nodes(matched_nodes[1], inverse_lookup=True, inplace=True)
        matched_nodes = np.stack([np.arange(len(matched_nodes[0])), np.arange(len(matched_nodes[1]))], axis=0)

    return matched_nodes


def ransac_refine_node_matching(
    fix_graph: VGraph,
    moving_graph: VGraph,
    matched_nodes: Iterable[Iterable[int]] | np.ndarray,
    matches_probability: Optional[np.ndarray] = None,
    *,
    reindex_graphs=False,
    return_mean_error=False,
    final_projection: Optional[Type[FundusProjection] | Mapping[int, Type[FundusProjection]]] = None,
) -> Tuple[FundusProjection, np.ndarray, float] | Tuple[FundusProjection, np.ndarray]:
    """
    Refine the node matching using the RANSAC algorithm and the nodes coordinates to estimate the geometrical transformation between the two graphs.

    Parameters
    ----------
        fix_graph:
            First graph to compare. The nodes of this graph will be considered as the reference for the estimated transformations.

        moving_graph:
            Second graph to compare. The nodes of this graph will be transformed to match the nodes of the first graph.

        matched_nodes:
            The matched nodes from the two graphs as a tuple of two arrays of indices (or an array of shape (2, N)).

        match_probability:
            The probability of the matches as a float array of shape (N,). If not None, the probability of the matches is used to weight the RANSAC algorithm.

        reindex_graphs:
            If True, reorder the nodes of the graphs so that the selected nodes are at the beginning of the list.

        return_mean_error:
            If True, return the mean error of the best transformation.

        final_projection:
            The type of transformation to use to map the points from the source to the destination graph. By default, AffineProjection

    Returns
    -------
        T: FundusProjection
            The best transformation of type ``final_projection`` found to transform the nodes of ``moving_graph`` to match the nodes of ``fix_graph``.

        matched_nodes: np.ndarray
            Indices of the matched nodes as a (2, N) array where N is the number of matched nodes, ``matched_nodes[0]`` are the indices of the matched nodes from the first graph and ``matched_nodes[1]`` are the indices of the matched nodes from the second graph.

        error: float
            Mean distance of the best transformation. Only returned if ``return_mean_error`` is True.

    """  # noqa: E501
    try:
        matched_nodes = np.asarray(matched_nodes)
        assert matched_nodes.ndim == 2 and matched_nodes.shape[0] == 2
    except Exception:
        raise ValueError("matched_nodes must be a tuple of two arrays or an array of shape (2, N)") from None
    fix_matched_nodes, mov_matched_nodes = matched_nodes

    fix_yx = fix_graph.node_coord()[fix_matched_nodes]
    mov_yx = moving_graph.node_coord()[mov_matched_nodes]

    if final_projection is None:
        final_projection = {12: QuadraticProjection}

    T, mean_error, valid_nodes = ransac_fit_projection(
        fix_yx,
        mov_yx,
        sampling_probability=matches_probability,
        n=3,
        initial_inliers_tolerance=30,
        final_inliers_tolerance=10,
        min_initial_inliers=5,
        max_iterations=300,
        early_stop_mean_error=3,
        final_projection=final_projection,
    )

    fix_matched_nodes = fix_matched_nodes[valid_nodes]
    mov_matched_nodes = mov_matched_nodes[valid_nodes]

    if reindex_graphs:
        fix_graph.reindex_nodes(fix_matched_nodes, inverse_lookup=True, inplace=True)
        moving_graph.reindex_nodes(mov_matched_nodes, inverse_lookup=True, inplace=True)
        fix_matched_nodes = np.arange(len(fix_matched_nodes))
        mov_matched_nodes = np.arange(len(mov_matched_nodes))

    matched_nodes = np.stack([fix_matched_nodes, mov_matched_nodes], axis=0)

    if return_mean_error:
        return T, matched_nodes, mean_error
    return T, matched_nodes


########################################################################################################################
#   === JUNCTION MATCHING ===
########################################################################################################################
MatchableArg: TypeAlias = np.ndarray | Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]


class NodeSimilarityEstimator(ABC):
    @abstractmethod
    def __call__(
        self,
        vgraph1: VGraph,
        vgraph2: VGraph,
        matchable: Optional[MatchableArg] = None,
        ctx: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute the similarity between the nodes of two graphs.

        Parameters
        ----------
        vgraph1 : VGraph
            The first graph containing N1 nodes.

        vgraph2 : VGraph
            The second graph containing N2 nodes.

        matchable : np.ndarray
            This parameter restricts the nodes that can be matched and whose similarity is computed. It can be:

            - A boolean array of shape (N1,N2) where True indicates that the node at the corresponding index is matchable;
            - A tuple of two boolean arrays of shape (N1,) and (N2,) where the first array indicates the matchable nodes of the first graph and the second array indicates the matchable nodes of the second graph.
            - A tuple of two integer 1D arrays containing the indices of the matchable nodes of the first and second graph.
            - ``None``, in which case all nodes are matchable.

        ctx : dict, optional
            A dictionary to store complementary or debug information.


        Returns
        -------
        np.ndarray
            The similarity matrix of shape between the nodes of the two graphs.

            - If ``matchable`` is a boolean array or None, the similarity matrix is of shape (N1, N2).
            - If ``matchable`` is a tuple of two integer arrays, the similarity matrix is of shape (len(matchable[0]), len(matchable[1])).
            - If ``matchable`` is a tuple of two boolean arrays, the similarity matrix is of shape (sum(matchable[0]), sum(matchable[1])).

        """  # noqa: E501
        pass

    def _check_matchable(
        self, vgraph1: VGraph, vgraph2: VGraph, matchable: Optional[MatchableArg] = None
    ) -> np.ndarray:
        if matchable is None:
            return None, None, None
        elif isinstance(matchable, tuple):
            if len(matchable) == 2:
                n1, n2 = matchable
                assert all(
                    isinstance(_, np.ndarray) for _ in matchable
                ), "matchable must be a tuple of two or three arrays"
                m = None
            elif len(matchable) == 3:
                m, n1, n2 = matchable
                assert all(isinstance(_, np.ndarray) for _ in (n1, n2))
                assert m is None or isinstance(m, np.ndarray)
            else:
                raise ValueError("matchable must be a tuple of two or three arrays")

            if n1.dtype == bool:
                assert n1.shape == (vgraph1.node_count,), "matchable[0] must have the same length as the first graph"
                n1 = n1.nonzero()[0]
            else:
                assert n1.dtype == int, "matchable[0] must be an array of integers"
            if n2.dtype == bool:
                assert n2.shape == (vgraph2.node_count,), "matchable[1] must have the same length as the second graph"
                n2 = n2.nonzero()[0]
            else:
                assert n2.dtype == int, "matchable[1] must be an array of integers"
            if m is not None:
                assert m.dtype == bool and m.shape == (len(n1), len(n2)), "matchable[2] must be a boolean array"
            return m, n1, n2
        elif isinstance(matchable, np.ndarray):
            assert matchable.dtype == bool and matchable.shape == (
                vgraph1.node_count,
                vgraph2.node_count,
            ), "matchable must be a boolean array of shape (N1,N2)"
            n1_mask = ~np.any(matchable, axis=0)
            n2_mask = ~np.any(matchable, axis=1)
            n1 = np.argwhere(n1_mask)[0]
            n2 = np.argwhere(n2_mask)[0]
            return matchable[n1_mask, n2_mask], n1, n2
        raise ValueError("matchable must be a tuple of two arrays or a boolean array")

    def branch_matching_available(self) -> bool:
        return False

    @abstractmethod
    def min_weight(self, vgraph1: VGraph, vgraph2: VGraph, n1: np.ndarray | None, n2: np.ndarray | None) -> np.ndarray:
        pass

    def hungarian(
        self,
        vgraph1: VGraph,
        vgraph2: VGraph,
        matchable: Optional[MatchableArg] = None,
        ctx: Optional[dict] = None,
    ):
        m, n1, n2 = self._check_matchable(vgraph1, vgraph2, matchable)
        if matchable is None:
            if m is not None:
                matchable = m
            elif n1 is not None and n2 is not None:
                matchable = (n1, n2)
        similarity = self(vgraph1, vgraph2, matchable, ctx=ctx)
        if m is not None:
            similarity[~m] = 0
        if ctx is not None:
            ctx["similarity"] = similarity
        min_weight1, min_weight2 = self.min_weight(vgraph1, vgraph2, n1, n2)
        matches = np.where(hungarian(similarity, unmatch1=min_weight1, unmatch2=min_weight2))
        if ctx is not None:
            ctx["matches"] = matches
            ctx["matches_similarity"] = similarity[matches]
            if "branches_match" in ctx:
                branches_match = ctx["branches_match"]
                ctx["branches_match"] = np.concatenate(
                    [branches_match[n1][n2] for n1, n2 in zip(*matches, strict=True)]
                ).T
        return np.array([m if n is None else n[m] for m, n in [(matches[0], n1), (matches[1], n2)]])

    def inspect(
        self,
        vgraph1: VGraph,
        vgraph2: VGraph,
        raw1: np.ndarray,
        raw2: np.ndarray,
        matching_gt: Optional[np.ndarray] = None,
        matchable: Optional[MatchableArg] = None,
        ctx: Optional[dict] = None,
    ):
        from .visualisation_tools import inspect_matching

        matcheable = self._check_matchable(vgraph1, vgraph2, matchable)
        _, n1, n2 = matcheable

        if ctx is None:
            ctx = {}
        self.hungarian(vgraph1, vgraph2, matchable, ctx)

        if self.branch_matching_available():
            branches_match = ctx["branches_match"]
            n_branch_match = len(_reindex_branch_matches(branches_match, vgraph1, vgraph2))
            branches_match = np.broadcast_to(np.arange(n_branch_match), (2, n_branch_match))
        else:
            branches_match = None

        if matching_gt is not None:
            n1_lookup = invert_lookup(n1)
            n2_lookup = invert_lookup(n2)
            valid_gt = np.isin(matching_gt[0], n1) & np.isin(matching_gt[1], n2)
            matching_gt = n1_lookup[matching_gt[0][valid_gt]], n2_lookup[matching_gt[1][valid_gt]]

        return inspect_matching(
            nodes_similarity=ctx["similarity"],
            matching=ctx["matches"],
            src_nodes_id=n1,
            dst_nodes_id=n2,
            src_graph=vgraph1,
            dst_graph=vgraph2,
            src_raw=raw1,
            dst_raw=raw2,
            src_features=self.format_node_features(vgraph1, n1),
            dst_features=self.format_node_features(vgraph2, n2),
            branch_matching=branches_match,
            matching_gt=matching_gt,
        )

    @abstractmethod
    def format_node_features(self, vgraph: VGraph, nodes_id: np.ndarray) -> pd.DataFrame:
        pass


class JunctionSimilarity(NodeSimilarityEstimator):
    def __init__(self, optimal_partial_incident_matching: bool = False, N_max_branches: int = 5):
        super().__init__()
        self.optimal_partial_incident_matching = optimal_partial_incident_matching
        self.N_max_branches = N_max_branches

    def _check_matchable(
        self, vgraph1: VGraph, vgraph2: VGraph, matchable: np.ndarray | Tuple[np.ndarray] | None = None
    ) -> np.ndarray:
        if matchable is None:
            matchable = (vgraph1.node_degree() > 2, vgraph2.node_degree() > 2)

        return super()._check_matchable(vgraph1, vgraph2, matchable)

    def __call__(
        self,
        vgraph1: VGraph,
        vgraph2: VGraph,
        matchable: Optional[np.ndarray | Tuple[np.ndarray, np.ndarray]] = None,
        ctx: Optional[dict] = None,
    ) -> np.ndarray:
        matchable, n1, n2 = self._check_matchable(vgraph1, vgraph2, matchable)
        if ctx is not None:
            ctx["n1"] = n1
            ctx["n2"] = n2

        maxB = self.N_max_branches
        ang_f1, scal_f1, branch1, u1, scal_std = self.node_features(vgraph1, n1)
        ang_f2, scal_f2, branch2, u2, _ = self.node_features(vgraph2, n2)

        B1 = np.asarray([len(_) for _ in branch1], dtype=int)
        B2 = np.asarray([len(_) for _ in branch2], dtype=int)

        if self.optimal_partial_incident_matching:
            sim, branches_match, n_iter = incident_branches_similarity(
                ang_f1, ang_f2, scal_f1, scal_f2, scal_std, B1, B2, u1, u2, matchable_nodes=matchable
            )
            if ctx is not None:
                ctx["n_iter"] = n_iter
                for i1, bmatch1 in enumerate(branches_match):
                    for i2, bmatch in enumerate(bmatch1):
                        if bmatch is not None:
                            bmatch1[i2] = np.array([(branch1[i1][b1], branch2[i2][b2]) for b1, b2 in bmatch])
                ctx["branches_match"] = branches_match
            return sim
        else:
            N_f_angle = ang_f1.shape[2]
            N_f_scalar = scal_f1.shape[2]
            N_branches = np.minimum(B1[:, None], B2[None, :])
            if ctx is not None:
                ctx["branches_match"] = [
                    [
                        np.stack([branch1[i1][: N_branches[i1][i2]], branch2[i2][: N_branches[i1][i2]]], axis=1)
                        for i2 in range(len(n2))
                    ]
                    for i1 in range(len(n1))
                ]
            N_branches = N_branches + 1e-3

            angle_cross = np.einsum("MBFx,NBFx->MN", ang_f1, ang_f2)
            scalar_diff = np.abs(scal_f1[:, None] - scal_f2[None, :])
            scalar_cross = np.exp(-(scalar_diff**2) / (2 * scal_std**2)).sum(axis=3)
            scalar_cross.transpose((0, 2, 1))[B1[:, None] > np.arange(maxB)[None, :]] = 0
            scalar_cross.transpose((1, 2, 0))[B2[:, None] > np.arange(maxB)[None, :]] = 0
            scalar_cross = scalar_cross.sum(axis=2)
            similarity = (angle_cross + scalar_cross) / (N_f_angle + N_f_scalar)
            return similarity / N_branches

    def node_features(
        self, vgraph: VGraph, nodes_id: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
        """Compute the features describing the nodes of the graph.

        Parameters
        ----------
        vgraph : VGraph
            The graph containing the nodes.

        nodes_id : np.ndarray
            The indices of the nodes for which to compute the features.

        Returns
        -------
        np.ndarray
            Unit vectors of angular features as an array of shape (N, B, F_ang, 2) where N is the number of nodes, B is the number of branches, and F_ang is the number of angular features.

        np.ndarray
            Scalar features as an array of shape (N, B, F_scalar) where N is the number of nodes, B is the number of branches, and F_scalar is the number of scalar features.

        List[np.ndarray]
            The branches incident to the nodes as a list containing N arrays of length at most B.

        np.ndarray
            Unit vectors used to order the incident branches as an array of shape (N, B, 2) where N is the number of nodes and B is the number of branches.

        np.ndarray
            The standard deviation of the scalar features as an array of shape (F_scalar,).

        """  # noqa: E501
        ang_f1, scal_f1, branch1, u1, scal_std = junction_incident_branches_descriptor(
            vgraph,
            junctions_id=nodes_id,
            N_max_branches=self.N_max_branches,
            return_incident_branches_id=True,
            return_incident_branches_u=True,
            return_scalar_features_std=True,
        )
        return ang_f1, scal_f1, branch1, u1, scal_std

    def format_node_features(self, vgraph: VGraph, nodes_id: np.ndarray) -> pd.DataFrame:
        maxB = self.N_max_branches
        ang_f, scal_f, branch, u, scal_std = self.node_features(vgraph, nodes_id)
        N = len(nodes_id)
        ang_f = ang_f.reshape(N, maxB, -1, 2)
        scal_f = scal_f.reshape(N, maxB, -1)

        features = {}
        for n, n_id in enumerate(nodes_id):
            node_features = features[n_id] = {}
            for b, (b_ang_f, b_scal_f) in enumerate(zip(ang_f[n], scal_f[n], strict=True)):
                if len(branch[n]) <= b:
                    node_features[f"B{b+1}"] = ""
                    for i, _ in enumerate(b_ang_f):
                        node_features[f"B{b+1} cos-{i}"] = ""
                    for i, _ in enumerate(b_scal_f):
                        node_features[f"B{b+1} l2-{i}"] = ""
                else:
                    node_features[f"B{b+1}"] = branch[n][b]
                    for i, f in enumerate(b_ang_f):
                        node_features[f"B{b+1} cos-{i}"] = f"{np.arctan2(*f)*180/np.pi:.0f}°"
                    for i, f in enumerate(b_scal_f):
                        node_features[f"B{b+1} l2-{i}"] = f"{f:.2f}"
        return pd.DataFrame(features).T

    def branch_matching_available(self) -> bool:
        return True

    def min_weight(self, vgraph1: VGraph, vgraph2: VGraph, n1: np.ndarray | None, n2: np.ndarray | None) -> np.ndarray:
        N1 = vgraph1.node_count if n1 is None else len(n1)
        N2 = vgraph2.node_count if n2 is None else len(n2)
        return np.full((N1,), 0.4), np.full((N2,), 0.4)


def match_junctions(
    vgraph1: VGraph,
    vgraph2: VGraph,
    pre_registration: bool = True,
    similarity_estimator: Optional[NodeSimilarityEstimator] = None,
    reindex: bool = False,
    match_branches: bool = False,
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

        similarity_estimator:
            The function that computes the similarity between the nodes.

        reindex: bool
            If True, reorder the nodes of the graphs so that the selected nodes are at the beginning of the list.

    Returns
    -------
        node1_matched, node2_matched:
            Every (node1_matched[i], node2_matched[i]) encode a match, node1_matched being the index of a node from the
        first graph and node2_matched the index of the corresponding node from the second graph. ``node1_matched`` and ``node2_matched`` are vectors of the same length.

    """  # noqa: E501
    if similarity_estimator is None:
        similarity_estimator = JunctionSimilarity(match_branches or pre_registration)
    else:
        if match_branches and not similarity_estimator.branch_matching_available():
            raise ValueError("The provided similarity estimator does not support branch matching.")

    n1, n2 = vgraph1.node_degree() > 2, vgraph2.node_degree() > 2

    matchable = None
    if pre_registration:
        pre_ctx = {}
        matched_nodes = JunctionSimilarity().hungarian(vgraph1, vgraph2, (n1, n2), ctx=pre_ctx)
        matches_p = np.exp(pre_ctx["matches_similarity"])
        matches_p /= matches_p.sum()
        try:
            T, _ = ransac_refine_node_matching(vgraph1, vgraph2, matched_nodes, matches_p)
        except ValueError:
            matchable = None
            pre_registration = False
        else:
            dist = np.linalg.norm(
                vgraph1.node_coord()[n1, None] - T.transform(vgraph2.node_coord()[n2])[None, :], axis=2
            )
            matchable = dist < 20

    ctx = {} if match_branches else None
    matched_nodes = similarity_estimator.hungarian(vgraph1, vgraph2, (matchable, n1, n2), ctx=ctx)

    if reindex:
        vgraph1.reindex_nodes(matched_nodes[0], inverse_lookup=True, inplace=True)
        vgraph2.reindex_nodes(matched_nodes[1], inverse_lookup=True, inplace=True)
        matched_nodes = np.broadcast_to(np.arange(len(matched_nodes[0])), (2, len(matched_nodes[0])))
        if match_branches:
            branches_match = ctx["branches_match"]
            n_branch_match = len(_reindex_branch_matches(branches_match, vgraph1, vgraph2))
            ctx["branches_match"] = np.broadcast_to(np.arange(n_branch_match), (2, n_branch_match))

    if match_branches:
        return matched_nodes, ctx["branches_match"]
    else:
        return matched_nodes


def match_junctions_simple(
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
    if features is None:
        features = functools.partial(
            junction_incident_branches_descriptor,
            return_junctions_id=True,
            return_incident_branches_id=True,
            return_incident_branches_u=False,
            return_scalar_features_std=False,
            N_max_branches=4,
        )

    cos_f1, l2_f1, junction_id1, incident_branches1 = features(vgraph1)
    cos_f2, l2_f2, junction_id2, incident_branches2 = features(vgraph2)
    n1 = len(junction_id1)
    n2 = len(junction_id2)
    n_cos_f = cos_f1.shape[2]
    n_l2_f = l2_f1.shape[2]

    N_branch = np.maximum(
        np.asarray([len(_) for _ in incident_branches1])[:, None],
        np.asarray([len(_) for _ in incident_branches2])[None, :],
    )
    cos_cross = np.einsum("MBFx,NBFx->MN", cos_f1, cos_f2)

    invalid_branch1 = np.all(l2_f1 == -50, axis=-1)
    invalid_branch2 = np.all(l2_f2 == -50, axis=-1)

    delta_l2_f = np.abs(l2_f1[:, None] - l2_f2[None, :])
    std = 3
    l2_cross = np.exp(-(delta_l2_f**2) / (2 * std**2))
    l2_cross.transpose((0, 2, 1, 3))[invalid_branch1] = 0
    l2_cross.transpose((1, 2, 0, 3))[invalid_branch2] = 0
    l2_cross = l2_cross.sum(axis=(2, 3))

    weight = (cos_cross + l2_cross) / (N_branch + 1e-3) / (n_cos_f + n_l2_f)
    min_weight = 0.4
    min_weight1 = np.repeat([min_weight], n1, axis=0)
    min_weight2 = np.repeat([min_weight], n2, axis=0)

    matched_nodes = hungarian(weight, n1, n2, min_weight1, min_weight2)
    matched_nodes = np.where(matched_nodes)

    matched_n1 = junction_id1[matched_nodes[0]]
    matched_n2 = junction_id2[matched_nodes[1]]

    if reindex_graphs:
        vgraph1.reindex_nodes(matched_n1, inverse_lookup=True, inplace=True)
        vgraph2.reindex_nodes(matched_n2, inverse_lookup=True, inplace=True)
        matched_n1 = np.arange(len(matched_n1))
        matched_n2 = np.arange(len(matched_n2))

    matched_nodes = np.stack([matched_n1, matched_n2], axis=0)
    return matched_nodes


def match_junctions_by_exact_incident_branches(
    vgraph1: VGraph, vgraph2: VGraph, *, reindex_nodes=False, reindex_branches=False
):
    cos_f1, l2_f1, junction_id1, incident_branches1 = junction_incident_branches_descriptor(vgraph1)
    cos_f2, l2_f2, junction_id2, incident_branches2 = junction_incident_branches_descriptor(vgraph2)
    n1 = len(junction_id1)
    n2 = len(junction_id2)

    total_sim, branch_matches = incident_branches_similarity(cos_f1, cos_f2, l2_f1, l2_f2)
    UNMATCH_SIM = 0.3

    matched_junctions = hungarian(
        total_sim, n1, n2, np.repeat([UNMATCH_SIM], n1, axis=0), np.repeat([UNMATCH_SIM], n2, axis=0)
    )
    matched_junctions = np.where(matched_junctions)

    matched_nodes1 = junction_id1[matched_junctions[0]]
    matched_nodes2 = junction_id2[matched_junctions[1]]

    if reindex_branches:
        matched_branch = []
        for n1, n2 in zip(*matched_junctions, strict=True):
            n1, n2 = int(n1), int(n2)
            branches1 = incident_branches1[n1]
            branches2 = incident_branches2[n2]
            matched_branch.append([(branches1[b1], branches2[b2]) for b1, b2 in branch_matches[n1][n2]])
        matched_branch = np.concatenate(matched_branch)
        _reindex_branch_matches(matched_branch, vgraph1, vgraph2)

    if reindex_nodes:
        vgraph1.reindex_nodes(matched_nodes1, inverse_lookup=True, inplace=True)
        vgraph2.reindex_nodes(matched_nodes2, inverse_lookup=True, inplace=True)
        matched_nodes1 = np.arange(len(matched_nodes1))
        matched_nodes2 = np.arange(len(matched_nodes2))

    return np.stack([matched_nodes1, matched_nodes2], axis=0)


########################################################################################################################
#   === UTILITY FUNCTIONS ===
########################################################################################################################
def _reindex_branch_matches(matches: np.ndarray, graph1: VGraph, graph2: VGraph) -> np.ndarray:
    """
    Given a list of branch matches, reindex the branches in both graphs.

    The matches consistency is checked and duplicates or invalid matches are removed. Invalid matches are the matches which pair the same branch of a graph with multiple branches of the other graph.

    The branches are reindexed in both graphs.

    Parameters
    ----------
    matches : np.ndarray
        A list of branch matches as a matrix of shape (2, N). Each column contains a pair of index corresponding to matching branches. The first row contains indices of the first graph, while the second row contains indices of the second graph.

    graph1 : VGraph
        The first graph.

    graph2 : VGraph
        The second graph.

    Returns
    -------
    np.ndarray
        The indices of the kept matches.

        This array can be used to derive the index mapping between the new and original branch indices:
        .. code-block:: python

            valid_matches = reindex_branch_matches(matches, graph1, graph2)
            b1_mapping = matches[valid_matches, 0]  # b1_mapping[new_index] = old_index
            b2_mapping = matches[valid_matches, 1]  # b2_mapping[new_index] = old_index

    """  # noqa: E501

    (b1_match, b2_match), matches_id = ensure_consistent_matches(matches, return_valid_index=True)

    # Reindex the branches
    b1_lookup = complete_lookup(b1_match, graph1.branch_count - 1)
    b2_lookup = complete_lookup(b2_match, graph2.branch_count - 1)
    graph1.reindex_branches(b1_lookup, inverse_lookup=True)
    graph2.reindex_branches(b2_lookup, inverse_lookup=True)

    # Return the valid matches indices
    return matches_id
