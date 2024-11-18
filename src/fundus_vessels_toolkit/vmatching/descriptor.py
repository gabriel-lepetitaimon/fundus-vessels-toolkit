from typing import Callable, List, Optional, Tuple

import numpy as np

from ..utils.bezier import BezierCubic, BSpline
from ..utils.geometric import Point
from ..vascular_data_objects import VBranchGeoData, VGraph


class NodeFeaturesCallback(Callable):
    def __call__(self, vgraph: VGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


def junction_incident_branches_descriptor(
    vgraph: VGraph,
    geometric_data_id: int = 0,
    junctions_id: Optional[np.ndarray] = None,  # noqa: F821
    *,
    bspline_name: str = VBranchGeoData.Fields.BSPLINE,
    calibre_name: str = VBranchGeoData.Fields.CALIBRES,
    return_junctions_id: bool = False,
    return_incident_branches_id: bool = False,
    return_incident_branches_u: bool = False,
    return_scalar_features_std: bool = False,
    N_max_branches: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    np.ndarray
        Angular features as a matrix of shape (N, maxB, angleF, 2) where N is the number of junctions, maxB is N_max_branches and angleF is the number of angular features.

    np.ndarray
        Scalar features as a matrix of shape (N, maxB, scalarF) where N is the number of junctions, maxB is N_max_branches and scalarF is the number of scalar 2 features.

    np.ndarray
        The nodes index of the described junctions as a matrix of shape (N,). (If return_junctions_id is True)

    List[np.ndarray]
        The index of the branches incident to the junctions as a list of array.
        (If return_incident_branches_index is True)

    np.ndarray
        The direction of the incident branches seen from its node position as an matrix of shape (N, maxB, 2).
        (If return_incident_branches_u is True).

    np.ndarray
        The expected standard deviation of the scalar features as a matrix of shape (scalarF,).

    """  # noqa: E501

    geo_data = vgraph.geometric_data(geometric_data_id)
    if junctions_id is None:
        junctions_id = np.argwhere(vgraph.node_degree() > 2).flatten()

    N = len(junctions_id)
    COS_F, L2_F = 4, 1

    u_vectors = np.zeros((N, N_max_branches, 2))
    cos_features = np.zeros((N, N_max_branches, COS_F, 2))
    L2_features = np.zeros((N, N_max_branches, L2_F)) - 50
    nodes_id = np.zeros(N, dtype=int)
    branches_id = []

    for junction_i, node_id in enumerate(junctions_id):
        nodes_id[junction_i] = node_id
        p = geo_datanode_coord(node_id)

        branches, are_outgoing = vgraph.incident_branches(node_id, return_branch_direction=True)
        incident_beziers: List[BezierCubic] = []
        incident_calibre: List[np.ndarray] = []

        # Read the bezier cubic curve and calibres of incident branches to the junction
        bsplines: List[BSpline] = geo_data.branch_bspline(branches, name=bspline_name)
        for branch_id, is_outgoing, bspline in zip(branches, are_outgoing, bsplines, strict=True):
            if len(bspline) == 0 or (len(bspline) == 1 and bspline[0].chord_length() < 20):
                p0_id, p1_id = vgraph.branch_list[branch_id]
                if not is_outgoing:
                    p0_id, p1_id = p1_id, p0_id
                p0, p1 = Point(*geo_datanode_coord(p0_id)), Point(*geo_datanode_coord(p1_id))
                incident_beziers.append(BezierCubic(p0, p1, p0, p1))
                incident_calibre.append(0)
            else:
                if is_outgoing:
                    calibre = np.mean(geo_data.branch_data(calibre_name, branch_id).data[:10])
                else:
                    bspline = bspline.flip()
                    calibre = np.mean(geo_data.branch_data(calibre_name, branch_id).data[-10:])
                incident_beziers.append(bspline[0])
                incident_calibre.append(calibre if not np.isnan(calibre) else 0)
        incident_calibre = np.array(incident_calibre)

        # If there are more than N_max_branches incident branches, keep those with the largest calibre
        if len(incident_beziers) > N_max_branches:
            select_branch = np.argsort(incident_calibre)[::-1][:N_max_branches]
            incident_beziers = [incident_beziers[i] for i in select_branch]
            incident_calibre = incident_calibre[select_branch]
            branches = branches[select_branch]

        # Sort the branches descending by the angle of the initial tangent vector
        u0_vectors = np.array([bezier.c0 - p for bezier in incident_beziers])
        incident_branch_order = np.argsort(np.arctan2(u0_vectors[:, 0], u0_vectors[:, 1]))
        branches = branches[incident_branch_order]
        branches_id.append(branches)
        u_vectors[junction_i, : len(incident_branch_order)] = u0_vectors[incident_branch_order]

        # Compute the cosine and L2 features
        for i, branch_i in enumerate(incident_branch_order):
            bezier = incident_beziers[branch_i]

            # Cosine features
            cos_features[junction_i, i, 0] = (bezier.c0 - p).normalized()  # Initial branch direction
            cos_features[junction_i, i, 1] = (bezier.p1 - bezier.p0).normalized()  # Chord vector
            cos_features[junction_i, i, 2] = (bezier.c0 - bezier.p0).normalized()  # Initial tangent vector
            cos_features[junction_i, i, 3] = (bezier.p1 - bezier.c1).normalized()  # End tangent vector

            # L2 features
            L2_features[junction_i, i, 0] = incident_calibre[branch_i]
            # L2_features[junction_i, i, 1] = bezier.chord_length()  # Chord length

    outs = [cos_features, L2_features]
    if return_junctions_id:
        outs.append(nodes_id)
    if return_incident_branches_id:
        outs.append(branches_id)
    if return_incident_branches_u:
        outs.append(u_vectors)
    if return_scalar_features_std:
        outs.append(np.array([2]))
    return outs
