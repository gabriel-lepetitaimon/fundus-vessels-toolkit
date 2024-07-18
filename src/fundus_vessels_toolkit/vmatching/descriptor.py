import math
from typing import Callable, List, Tuple

import numpy as np

from ..utils.bezier import BezierCubic, BSpline
from ..utils.geometric import Point
from ..vascular_data_objects import VGraph


class NodeFeaturesCallback(Callable):
    def __call__(self, vgraph: VGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


def junction_bspline_descriptor(
    vgraph: VGraph,
    geometric_data_id: int = 0,
    *,
    bspline_name: str = "bspline",
    calibre_name: str = "calibre",
    orientation_invariant: bool = False,
    N_max_branches: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    orientation_invariant : bool, optional
        If True, the descriptor is orientation invariant. By default True.

    Returns
    -------
    np.ndarray,
        Cosine features as a matrix of shape (n_junctions, n_cos_features).

    np.ndarray,
        L2 features as a matrix of shape (n_junctions, n_L2_features).

    np.ndarray
        The nodes index of the described junctions.

    """

    geo_data = vgraph.geometric_data(geometric_data_id)

    cosine_features = []
    L2_features = []
    nodes_id = []

    for node_i in np.argwhere(vgraph.nodes_degree() > 2).flatten():
        branches, are_outgoing = vgraph.incident_branches(node_i, return_branch_direction=True)
        incident_beziers: List[BezierCubic] = []
        incident_calibre: List[np.ndarray] = []

        # Read the bezier cubic curve and calibres of incident branches to the junction
        bsplines: List[BSpline] = geo_data.branch_bspline(bspline_name, branches)
        for branch_id, is_outgoing, bspline in zip(branches, are_outgoing, bsplines, strict=True):
            if len(bspline) == 0:
                p0, p1 = vgraph.branch_list[branch_id]
                if not is_outgoing:
                    p0, p1 = p1, p0
                p0 = Point(*geo_data.nodes_coord(p0))
                p1 = Point(*geo_data.nodes_coord(p1))
                c0 = (p1 - p0).normalized()
                c1 = -c0
                incident_beziers.append(BezierCubic(p0, c0, c1, p1))
                incident_calibre.append(0)
            else:
                if is_outgoing:
                    calibre = np.mean(geo_data.branch_data(calibre_name, branch_id)[:10])
                else:
                    bspline = bspline.flip()
                    calibre = np.mean(geo_data.branch_data(calibre_name, branch_id)[-10:])
                incident_beziers.append(bspline[0])
                incident_calibre.append(calibre if not np.isnan(calibre) else 0)

        # Sort the branches descending by the length of their bezier curve
        p = geo_data.nodes_coord(node_i)
        p0_angles = np.array([(bezier.p0 - p).angle for bezier in incident_beziers])
        argsort = np.argsort(p0_angles)
        incident_beziers = [incident_beziers[i] for i in argsort]
        incident_calibre = [incident_calibre[i] for i in argsort]

        # Compute the cosine and L2 features
        cos_f = []
        L2_f = []
        for i, bezier, calibre in zip(range(N_max_branches), incident_beziers, incident_calibre, strict=False):
            # Cosine features
            chord_v = (bezier.p1 - bezier.p0).normalized()  # Chord vector
            t0 = (bezier.c0 - bezier.p0).normalized()  # Initial tangent vector

            # The following lines were removed because they are bad descriptors...
            # t1 = (bezier.c1 - bezier.p1).normalized()  # Final tangent vector
            # arc_v = ((bezier.c0 + bezier.c1) / 2 - bezier.p0).normalized()  # Elevation between the chord and the curve
            cos_f.append(np.concatenate([t0, chord_v]))

            # L2 features
            L2_f.append(np.array([calibre]))

        cos_f = np.concatenate(cos_f + [np.zeros(len(cos_f[0]))] * (N_max_branches - len(cos_f)))
        L2_f = np.concatenate(L2_f + [-50 * np.ones(len(L2_f[0]))] * (N_max_branches - len(L2_f)))

        if orientation_invariant:
            # Rotate cos_f so its first vector is vertical
            cos_f = cos_f.reshape(-1, 2)
            v, cos_f = cos_f[0], cos_f[1:]
            M = np.array([[v[0], -v[1]], [v[1], v[0]]])
            cos_f = (M @ cos_f.T).T.flatten()

        cos_norm = np.linalg.norm(cos_f)
        if cos_norm > 0:
            cos_f /= cos_norm

        cosine_features.append(cos_f)
        L2_features.append(L2_f)
        nodes_id.append(node_i)

    return np.array(cosine_features), np.array(L2_features), np.array(nodes_id)
