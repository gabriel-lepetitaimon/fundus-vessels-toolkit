from dataclasses import dataclass
from re import split
from typing import Optional, Protocol, Tuple

import numpy as np

from fundus_toolkits.utils.geometric import Point

from fundus_vessels_toolkit.vascular_data_objects.fundus_data import AVLabel
from fundus_vessels_toolkit.vascular_data_objects.vtree import VTree

from ..utils.bezier import BezierCubic, BSpline
from ..utils.typing import (
    Float1DArray,
    Float2DArray,
    FloatPairArray,
    FloatPairMap,
    FloatPairVolume,
    Indices,
    Int1DArray,
)
from ..vascular_data_objects import VBranchGeoData, VGraph


class NodeFeaturesCallback(Protocol):
    def __call__(self, vgraph: VGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class JunctionAdjacentBranchDescriptor:
    node_ids: Indices
    """The index of the described junctions as a matrix of shape (node_count,)."""

    adj_branch_ids: list[Indices]
    """The index of the branches incident to the junctions as a list of ``node_count`` arrays."""

    adj_branch_tan: list[FloatPairArray]
    """The tangent of the incident branches seen from its adjacent node as ``node_count`` matrices of shape (B, 2) where B is the number of adjacent branches to each junction."""  # noqa: E501

    angle_features: list[FloatPairMap]
    """The angular features as a list of ``node_count`` matrices of shape (B, angle_features_count, 2) where B is the number of adjacent branches to each junction."""  # noqa: E501

    scalar_features: list[Float2DArray]
    """The scalar features as a list of ``node_count`` matrices of shape (B, scalar_features_count) where B is the number of adjacent branches to each junction."""  # noqa: E501

    scalar_features_std: Float1DArray
    """The expected standard deviation of the scalar features as a matrix of shape (scalar_features_count,)."""

    @property
    def node_count(self) -> int:
        return self.node_ids.shape[0]

    @property
    def angle_features_count(self) -> int:
        return self.angle_features[0].shape[1]

    @property
    def scalar_features_count(self) -> int:
        return self.scalar_features[0].shape[1]

    @property
    def adj_branch_count(self) -> Int1DArray:
        return np.array([len(adj_branch_ids) for adj_branch_ids in self.adj_branch_ids], dtype=np.int_)  # type: ignore

    def adj_branch_tan_block(self, max_branches: int) -> FloatPairMap:
        """Returns the adjacent branch tangents as a block matrix of shape (node_count, max_branches, 2) where max_branches is the maximum number of adjacent branches to the junctions. If a junction has less than max_branches adjacent branches, the remaining entries are filled with zeros."""  # noqa: E501
        adj_branch_tan_block: FloatPairMap = np.zeros((self.node_count, max_branches, 2))  # type: ignore
        for i in range(self.node_count):
            adj_branch_tan = self.adj_branch_tan[i]
            b = min(adj_branch_tan.shape[0], max_branches)
            adj_branch_tan_block[i, :b] = adj_branch_tan[:b]
        return adj_branch_tan_block

    def dot_features_block(self, max_branches: int) -> FloatPairVolume:
        """Returns the angle features as a block matrix of shape (node_count, max_branches, 2*angle_features_count) where max_branches is the maximum number of adjacent branches to the junctions. If a junction has less than max_branches adjacent branches, the remaining entries are filled with zeros."""  # noqa: E501
        angle_features_block: FloatPairVolume = np.zeros((self.node_count, max_branches, 2 * self.angle_features_count))  # type: ignore
        for i in range(self.node_count):
            angle_features = self.angle_features[i]
            b = min(angle_features.shape[0], max_branches)
            angle_features_block[i, :b] = angle_features[:b].reshape(b, -1)
        return angle_features_block

    def scalar_features_block(self, max_branches: int) -> Float2DArray:
        """Returns the scalar features as a block matrix of shape (node_count, max_branches, scalar_features_count) where max_branches is the maximum number of adjacent branches to the junctions. If a junction has less than max_branches adjacent branches, the remaining entries are filled with zeros."""  # noqa: E501
        scalar_features_block: Float2DArray = np.zeros((self.node_count, max_branches, self.scalar_features_count))  # type: ignore
        for i in range(self.node_count):
            scalar_features = self.scalar_features[i]
            b = min(scalar_features.shape[0], max_branches)
            scalar_features_block[i, :b] = scalar_features[:b]
        return scalar_features_block


def junction_adjacent_branches_descriptor(
    vgraph: VGraph,
    junctions_id: Optional[np.ndarray] = None,  # noqa: F821
    *,
    geometric_data_id: int = 0,
    bspline_name=VBranchGeoData.Fields.BSPLINE,
    calibre_name=VBranchGeoData.Fields.CALIBRES,
) -> JunctionAdjacentBranchDescriptor:
    """
    Computes the descriptor of all junctions in a graph.

    Parameters
    ----------
    vgraph : VGraph
        The vascular graph.

    geometric_data_id : int, optional
        The geometric data ID to use. By default the first one is used.

    bspline_name : str, optional
        The name of the branch attribute of the geometric data storing the B-spline. By default the first one is used.

    calibre_name : str, optional
        The name of the branch attribute of the geometric data storing the vessel calibre. By default "calibre".

    Returns
    -------
    JunctionAdjacentBranchDescriptor
        The descriptor of the junctions in the graph.
    """  # noqa: E501

    geo_data = vgraph.geometric_data(geometric_data_id)
    if junctions_id is None:
        junctions_id = np.argwhere(vgraph.node_degree() > 2).flatten()

    N = len(junctions_id)
    COS_F, L2_F = 4, 1

    # u_vectors = np.zeros((N, N_max_branches, 2))
    # cos_features = np.zeros((N, N_max_branches, COS_F, 2))
    # L2_features = np.zeros((N, N_max_branches, L2_F)) - 50
    node_ids = np.zeros(N, dtype=int)
    all_adj_branch_ids = []
    all_adj_branch_tan = []
    all_angle_features = []
    all_scalar_features = []

    for junction_i, node_id in enumerate(junctions_id):
        node_ids[junction_i] = node_id
        p = geo_data.node_coord(node_id)

        adj_branch_ids, are_outgoing = vgraph.adjacent_branches(node_id, return_branch_direction=True)
        n_adj_branch = len(adj_branch_ids)

        # Read the bezier cubic curve and calibres of incident branches to the junction
        bsplines: list[BSpline] = geo_data.branch_bspline(adj_branch_ids, attr=bspline_name)
        adj_beziers: list[BezierCubic] = []
        incident_calibre: Float1DArray = np.zeros(len(adj_branch_ids))
        for branch_i in range(len(adj_branch_ids)):
            branch_id, is_outgoing, bspline = adj_branch_ids[branch_i], are_outgoing[branch_i], bsplines[branch_i]
            if len(bspline) == 0 or (len(bspline) == 1 and bspline[0].chord_length() < 20):
                p0_id, p1_id = vgraph.branch_list[branch_id]
                if not is_outgoing:
                    p0_id, p1_id = p1_id, p0_id
                p0, p1 = Point(*geo_data.node_coord(p0_id)), Point(*geo_data.node_coord(p1_id))
                adj_beziers.append(BezierCubic(p0, p1, p0, p1))
            else:
                if is_outgoing:
                    calibre = np.mean(geo_data.branch_data(calibre_name, branch_id).data[:10])
                else:
                    bspline = bspline.flip()
                    calibre = np.mean(geo_data.branch_data(calibre_name, branch_id).data[-10:])
                adj_beziers.append(bspline[0])
                if not np.isnan(calibre):
                    incident_calibre[branch_i] = calibre

        # Sort the branches by descending calibre
        branch_order = np.argsort(incident_calibre)[::-1]
        all_adj_branch_ids.append(adj_branch_ids[branch_order])

        # Compute the cosine and L2 features
        adj_branch_tan = np.empty((n_adj_branch, 2))
        angle_features = np.empty((n_adj_branch, COS_F, 2))
        scalar_features = np.empty((n_adj_branch, L2_F))
        for i, branch_i in enumerate(branch_order):
            bezier = adj_beziers[branch_i]
            adj_branch_tan[i] = bezier.c0 - p

            # Cosine features
            angle_features[i, 0] = (bezier.c0 - p).normalized()  # Initial branch direction
            angle_features[i, 1] = (bezier.p1 - bezier.p0).normalized()  # Chord vector
            angle_features[i, 2] = (bezier.c0 - bezier.p0).normalized()  # Initial tangent vector
            angle_features[i, 3] = (bezier.p1 - bezier.c1).normalized()  # End tangent vector

            # L2 features
            scalar_features[i] = incident_calibre[branch_i]
            # L2_features[i, 1] = bezier.chord_length()  # Chord length

        all_adj_branch_tan.append(adj_branch_tan)
        all_angle_features.append(angle_features)
        all_scalar_features.append(scalar_features)

    return JunctionAdjacentBranchDescriptor(
        node_ids=node_ids,
        adj_branch_ids=all_adj_branch_ids,
        adj_branch_tan=all_adj_branch_tan,
        angle_features=all_angle_features,
        scalar_features=all_scalar_features,
        scalar_features_std=np.array([2]),  # type: ignore
    )


def tree_node_histogram(
    tree: VTree,
    nodes: Indices,
    *,
    landmark_nodes: Optional[Indices] = None,
    r_bins: int = 5,
    r_max: float = 250,
    theta_bins: int = 8,
    split_av: Optional[bool] = None,
) -> Float2DArray:
    """Compute a histogram of the branch directions at a node in the tree.

    Parameters
    ----------
    tree : VTree
        The vascular tree.

    nodes : Indices
        The indices of the nodes to compute the histogram for as a vector of shape (N,).

    landmark_nodes : Optional[Indices], optional
        The indices of the landmark nodes to compute the histogram with respect to as a vector of shape (N,). If None, use ``nodes`` as landmark nodes. By default None.

    Returns
    -------
    Float2DArray
        The histogram of surrounding nodes as a matrix of shape (N, r_bins*theta_bins) where N is the number of nodes in the input vector.
    """  # noqa: E501
    if split_av is None:
        split_av = "av" in tree.node_attr.columns
    if split_av is True:
        assert "av" in tree.node_attr.columns, (
            "The tree does not have an 'av' node attribute to split arteries and veins."
        )
        art_mask = tree.node_attr.loc[nodes, "av"] == AVLabel.ART
        opts = dict(tree=tree, nodes=nodes, r_bins=r_bins, r_max=r_max, theta_bins=theta_bins, split_av=False)
        art_hist = tree_node_histogram(landmark_nodes=nodes[art_mask], **opts)  # type: ignore
        vei_hist = tree_node_histogram(landmark_nodes=nodes[~art_mask], **opts)  # type: ignore
        return np.concatenate([art_hist, vei_hist], axis=1)  # type: ignore

    if landmark_nodes is None:
        landmark_nodes = nodes

    N, L = len(nodes), len(landmark_nodes)

    node_yx = tree.geometric_data().node_coord(nodes)
    landmark_yx = tree.geometric_data().node_coord(landmark_nodes)

    rel_yx = landmark_yx[None, :, :] - node_yx[:, None, :]

    r_bin_step = r_max / r_bins
    r_norm = np.linalg.norm(rel_yx, axis=2)
    r = r_norm[...] / r_bin_step  # (N, L)
    r_n = np.tile(np.arange(N)[:, None], (1, L))
    r_bin = np.floor(r).astype(int)
    r_mask = r_bin < r_bins - 1
    r_bin = r_bin[r_mask]
    r = r[r_mask]
    r_n = r_n[r_mask]
    r_hist = np.zeros((nodes.shape[0], r_bins))  # type: ignore
    np.add.at(r_hist, (r_n, r_bin), r % 1)
    np.add.at(r_hist, (r_n, r_bin + 1), 1 - r % 1)

    rel_yx[r_norm != 0, 0] /= r_norm[r_norm != 0]
    rel_yx[r_norm != 0, 1] /= r_norm[r_norm != 0]
    theta = np.linspace(0, 2 * np.pi, theta_bins, endpoint=False)
    theta_base = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (theta_bins, 2)
    theta_hist = np.maximum(rel_yx @ theta_base.T, 0).sum(axis=1)  # (N, L, theta_bins)

    hist = np.concatenate([r_hist, theta_hist], axis=1)  # (N, r_bins + theta_bins)
    return hist  # type: ignore
