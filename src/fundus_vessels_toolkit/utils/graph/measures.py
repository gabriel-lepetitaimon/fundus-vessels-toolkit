from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from ..cpp_extensions.fvt_cpp import extract_branches_geometry as extract_branches_geometry_cpp
from ..cpp_extensions.fvt_cpp import (
    extract_branches_geometry_from_skeleton as extract_branches_geometry_from_skeleton_cpp,
)
from ..cpp_extensions.fvt_cpp import fast_branch_boundaries as fast_branch_boundaries_cpp
from ..cpp_extensions.fvt_cpp import fast_curve_tangent as fast_curve_tangent_cpp
from ..cpp_extensions.fvt_cpp import track_branches as track_branches_cpp
from ..geometric import Point, Rect
from ..torch import autocast_torch


@autocast_torch
def track_branches(edge_labels_map, nodes_yx, edge_list) -> list[list[int]]:
    """Track branches from a skeleton map.

    Parameters
    ----------
    edge_labels_map :
        A 2D image of the skeleton where each edge is labeled with a unique integer.

    nodes_yx :
        Array of shape (N,2) providing the coordinates of the skeleton nodes.

    edge_list :
        Array of shape (E,2) providing the list of edges in the skeleton as a pair of node indices.


    Returns
    -------
    list[np.array]
        A list of list of integers, each list represents a branch and contains the edge labels of the branch.

    """
    edge_labels_map = edge_labels_map.int()
    nodes_yx = nodes_yx.int()
    edge_list = edge_list.int()

    return track_branches_cpp(edge_labels_map, nodes_yx, edge_list)


@autocast_torch
def extract_branch_geometry(
    branch_curves: List[torch.Tensor],
    segmentation: torch.Tensor,
    adaptative_tangents: bool = True,
    return_calibre: bool = True,
    return_boundaries: bool = False,
    return_curvature: bool = False,
    return_curvature_roots: bool = False,
    extract_bspline: bool = True,
    curvature_roots_percentile_threshold: float = 0.1,
    bspline_target_error: float = 3,
    split_on_gaps: float = 2,
) -> Tuple[List[torch.Tensor], ...]:
    """Extract geometric properties of vascular branches from their curves coordinates and a vessel segmentation map.


    Parameters
    ----------
    branch_curves :
        A list of 2D tensors of shape (n, 2) containing the coordinates of the branch points.

    segmentation : torch.Tensor
        A 2D tensor of shape (H, W) containing the segmentation of the image.

    adaptative_tangents : bool, optional
        If True, the standard deviation of the gaussian weighting the curve points is set to the vessel calibre.

    return_calibre : bool, optional
        If True, the output will contain the branch width (or calibre) at each branch point. By default True.

    return_boundaries : bool, optional
        If True, the output will contain the boundaries of the branches. By default False.

    return_curvature : bool, optional
        If True, the output will contain the curvature of the branches. By default False.

    return_curvature_roots : bool, optional
        If True, the output will contain the curvature roots of the branches. By default False.

    extract_bspline : bool, optional
        If True, the output will contain the bspline interpolation of the branches. By default True.

    bspline_target_error : float, optional
        The maximum error allowed for the bspline interpolation. By default 10.

    split_on_gaps : float, optional
        The maximum gap (in pixels) between two points of the skeleton above which the curve is split. By default 2.

    curvature_roots_percentile_threshold : float, optional
        The percentile of the curvature values used to threshold the curvature roots. By default 0.15.

    Returns
    -------
    tuple[list[torch.Tensor], ...]
    This method returns a tuple containing at least three lists of length B which contains, for each branch:
    - the branch curve cleaned as an integer tensor of shape (n, 2) containing the yx coordinates of each point of the branch;
    - the index of discontinuity in the branch as an integer tensor of shape (s,);
    - the tangent at every point of the branch curve as a float tensor of shape (n, 2);

    Then depending on the options, the output may also contain for every point of the branch curve:
    - its calibre as a float tensor of shape (n,);
    - the coordinates of the nearest vessel edges as an integer tensor of shape (n, 2, 2) indexed as [point, (edgeside: left=0;right=1), (y=0;x=1)];
    - its curvature as a float tensor of shape (n,);
    - the index of the roots of the curvature as an integer tensor of shape (c,);
    """  # noqa: E501
    options = dict(
        adaptative_tangents=adaptative_tangents,
        return_calibre=return_calibre,
        return_boundaries=return_boundaries,
        return_curvature=return_curvature,
        return_curvature_roots=return_curvature_roots,
        extract_bspline=extract_bspline,
        bspline_target_error=bspline_target_error,
        split_on_gaps=split_on_gaps,
        curvature_roots_percentile_threshold=curvature_roots_percentile_threshold,
    )

    assert 0 <= curvature_roots_percentile_threshold <= 1, "curvature_roots_percentile_threshold must be in (0, 1)"
    assert bspline_target_error >= 0, "bspline_target_error must be positive"

    branch_curves = [curve.cpu().int() for curve in branch_curves]

    return tuple(extract_branches_geometry_cpp(branch_curves, segmentation, options))


@autocast_torch
def extract_branch_geometry_from_skeleton(
    branch_labels: torch.Tensor,
    node_yx: torch.Tensor,
    edge_list: torch.Tensor,
    segmentation: torch.Tensor,
    clean_branches_tips: int = 20,
    adaptative_tangents: bool = True,
    return_labels: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Track branches from a labels map and extract their geometry.


    Parameters
    ----------
    branch_labels :
        A 2D tensor of shape (H, W) containing the skeleton where each branch has a unique label.

    node_yx :
        A 2D tensor of shape (N, 2) containing the coordinates (y, x) of the nodes.

    branch_list :
        A 2D tensor of shape (B, 2) containing for each branch, the indices of the two nodes it connects.

    segmentation : torch.Tensor
        A 2D tensor of shape (H, W) containing the segmentation of the image.

    clean_branches_tips : int, optional
        The maximum number of pixels removable at branches tips. By default 20.

    adaptative_tangents : bool, optional
        If True, the standard deviation of the gaussian weighting the curve points is set to the vessel cal

    return_labels : bool, optional

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    This method returns a tuple containing three lists of length B which contains, for each branch:
    - a 2D tensor of shape (n, 2) containing the coordinates of the branch points (where n is the branch length).
    - a 2D tensor of shape (n, 2) containing the tangent vectors at each branch point.
    - a 2D tensor of shape (n,) containing the branch width (or calibre) at each branch point.

    If return_labels is True, the output will also contain:
        - a 2D tensor of shape (H,W) containing the branch labels after tips cleaning.

    """
    options = dict(
        clean_branches_tips=float(clean_branches_tips), bspline_max_error=4, adaptative_tangents=adaptative_tangents
    )

    branch_labels = branch_labels.int()
    assert branch_labels.ndim == 2, "branch_labels must be a 2D tensor"

    assert node_yx.ndim == 2 and node_yx.shape[1] == 2, "node_yx must be a 2D tensor of shape (N, 2)"
    node_yx = node_yx.int()

    assert edge_list.ndim == 2 and edge_list.shape[1] == 2, "branch_list must be a 2D tensor of shape (E, 2)"
    edge_list = edge_list.int()

    out = extract_branches_geometry_from_skeleton_cpp(branch_labels, node_yx, edge_list, segmentation, options)
    return tuple(out) + (branch_labels,) if return_labels else tuple(out)


@autocast_torch
def curve_tangent(curve_yx, std=3, eval_for=None):
    """Compute the local tangents of a curve.`

    The tangent at each point is computed by averaging the vectors starting from the current node and pointing to the next node in the curve, with those starting from the previous nodes and pointing to the current node. The vectors are weighted by a gaussian distribution centered on the current node.

    Parameters
    ----------
    curve_yx :
        A 2D tensor of shape ``(L, 2)`` containing the coordinates of the curve points.
    std : int, optional
        The standard deviation of the gaussian weighting the curve points. By default 3.
    eval_for : int or list[int], optional
        The indices of the points for which the tangent must be computed. By default None.

    Returns
    -------
    torch.Tensor
        A 2D tensor of shape ``(L', 2)`` containing the tangent vectors at each curve point.
        If eval_for is None, ``L' = L``, otherwise ``L' = len(eval_for)``.
    """  # noqa: E501
    curve_yx = curve_yx.int()
    eval_for = _check_eval_for_point(eval_for)
    return fast_curve_tangent_cpp(curve_yx, std, eval_for)


@autocast_torch
def branch_boundaries(curve_yx, segmentation, eval_for=None):
    curve_yx = curve_yx.int()
    segmentation = segmentation.bool()
    eval_for = _check_eval_for_point(eval_for)
    return fast_branch_boundaries_cpp(curve_yx, segmentation, eval_for)


def _check_eval_for_point(eval_for):
    if isinstance(eval_for, int):
        eval_for = torch.Tensor([eval_for])
    elif eval_for is None:
        eval_for = torch.Tensor([])
    else:
        eval_for = torch.as_tensor(eval_for)
    return eval_for.int()


def extract_bifurcations_parameters(branches_calibre, branches_tangent, branches_list, directed=True) -> pd.DataFrame:
    """
    Extracts the parameters of the bifurcations from the branches.
    """
    adj_list = [[] for _ in range(branches_list.max() + 1)]
    # Create the adjacency list, storing for each node the list of branches and wether they are incident to the node
    for branchID, (node0, node1) in enumerate(branches_list):
        adj_list[node0].append((branchID, False))  # The branch is outgoing from the node
        adj_list[node1].append((branchID, True))  # The branch is incident to the node

    tangents = [b.data if b is not None and len(b.data) else None for b in branches_tangent]
    calibres = [b.data if b is not None and len(b.data) else None for b in branches_calibre]

    bifurcations = []
    for nodeID, node_adjacency in enumerate(adj_list):
        if len(node_adjacency) == 3:
            if any(tangents[_[0]] is None for _ in node_adjacency):
                continue
            if directed:
                b0 = [b for b, incident in node_adjacency if incident][0]
                b1, b2 = [b for b, incident in node_adjacency if b != b0]
                dir0 = (np.arctan2(*tangents[b0][-1]) + np.pi) % (2 * np.pi)
                dir1 = np.arctan2(*tangents[b1][0])
                dir2 = np.arctan2(*tangents[b2][0])
                c0 = np.mean(calibres[b0][-10:])
                c1 = np.mean(calibres[b1][:10])
                c2 = np.mean(calibres[b2][:10])
            else:
                (b0, b0_incident), (b1, b1_incident), (b2, b2_incident) = node_adjacency
                if b0_incident:
                    dir0 = (np.arctan2(*tangents[b0][-1]) + np.pi) % (2 * np.pi)
                    c0 = np.mean(calibres[b0][-10:])
                else:
                    dir0 = np.arctan2(*tangents[b0][0])
                    c0 = np.mean(calibres[b0][:10])
                if b1_incident:
                    dir1 = (np.arctan2(*tangents[b1][-1]) + np.pi) % (2 * np.pi)
                    c1 = np.mean(calibres[b1][-10:])
                else:
                    dir1 = np.arctan2(*tangents[b1][0])
                    c1 = np.mean(calibres[b1][:10])
                if b2_incident:
                    dir2 = (np.arctan2(*tangents[b2][-1]) + np.pi) % (2 * np.pi)
                    c2 = np.mean(calibres[b2][-10:])
                else:
                    dir2 = np.arctan2(*tangents[b2][0])
                    c2 = np.mean(calibres[b2][:10])

                # Use the largest branch as the main branch
                if c1 > c0 and c1 > c2:
                    b0, b1 = b1, b0
                    dir0, dir1 = dir1, dir0
                    c0, c1 = c1, c0
                elif c2 > c0 and c2 > c1:
                    b0, b2 = b2, b0
                    dir0, dir2 = dir2, dir0
                    c0, c2 = c2, c0

            # Sort the branches by their direction
            dir1 = (dir1 - dir0) % (2 * np.pi)
            dir2 = (dir2 - dir0) % (2 * np.pi)
            if dir1 > dir2:
                b1, b2 = b2, b1
                dir1, dir2 = dir2, dir1
                c1, c2 = c2, c1

            # Compute the angles between the incident branch and the outgoing branches
            theta1 = np.pi - dir1
            theta2 = dir2 - np.pi

            # Ensure branch1 is the main branch (the one with the smallest angle with the incident branch)
            if theta1 > theta2:
                b1, b2 = b2, b1
                theta1, theta2 = theta2, theta1
                c1, c2 = c2, c1

            bifurcations.append(
                dict(
                    nodeID=int(nodeID),
                    b0=int(b0),
                    b1=int(b1),
                    b2=int(b2),
                    θ1=theta1 * 180 / np.pi,
                    θ2=theta2 * 180 / np.pi,
                    d0=c0,
                    d1=c1,
                    d2=c2,
                )
            )

    return pd.DataFrame(bifurcations).set_index("nodeID")


########################################################################################################################
#   LEGACY IMPLEMENTATION


def perimeter_from_vertices(coord: np.ndarray, close_loop: bool = True) -> float:
    """
    Compute the perimeter of a polygon defined by a list of vertices.

    Args:
        coord: A (V, 2) array or list of V vertices.
        close_loop: If True, the polygon is closed by adding an edge between the first and the last vertex.

    Returns:
        The perimeter of the polygon. (The sum of the distance between each vertex and the next one.)
    """
    coord = np.asarray(coord)
    next_coord = np.roll(coord, 1, axis=0)
    if not close_loop:
        next_coord = next_coord[:-1]
        coord = coord[:-1]
    return np.sum(np.linalg.norm(coord - next_coord, axis=1))


def nodes_tangent(
    nodes_coord: np.ndarray,
    branch_label_map: np.ndarray,
    branches_id: Iterable[int] = None,
    *,
    gaussian_offset: float = 7,
    gaussian_std: float = 7,
):
    """
    Compute the vector tangent to the skeleton at given nodes. The vector is directed as a continuation of te skeleton.

    The computed vector is the symmetric of the barycenter of the surrounding skeleton points weighted by a gaussian
    distribution on the distance to the considered node. Note that the center of the gaussian distribution is offset
    from the considered node to account for skeleton artefact near the node.


    Parameters
    ----------
    coord : np.ndarray, shape (N, 2)
        The coordinates of the nodes to consider.
    skeleton_map : np.ndarray, shape (H, W)
        The skeleton map.
    branch_id : Iterable[int], shape (N,)
        For each node, the label of the branch it is connected to.
        If 0, all the branches arround the node will be considered. (This may lead to indesirable results.)

        .. warning::

            The branch_id must follow the same indexing as the skeleton_map: especially the indices must start at 1.

    gaussian_offset : float, optional
        The offset in pixel of the gaussian weighting the skeleton arround the node. Must be positive.
        By default: 7.
    std : float, optional
        The standard deviation in pixel of the gaussian weighting the skeleton arround the node. Must be positive.
        By default 7.

    Returns
    -------
    np.ndarray, shape (N, 2)
        The tangent vectors at the given nodes. The vectors are normalized to unit length.
    """
    assert len(nodes_coord) == len(branches_id), "coord and branch_id must have the same length"
    assert nodes_coord.ndim == 2 and nodes_coord.shape[1] == 2, "coord must be a 2D array of shape (N, 2)"
    N = len(nodes_coord)
    assert branch_label_map.ndim == 2, "skeleton_map must be a 2D array"
    assert branch_label_map.shape[0] > 0 and branch_label_map.shape[1] > 0, "skeleton_map must be non empty"
    assert gaussian_offset > 0, "gaussian_offset must be positive"
    assert gaussian_std > 0, "gaussian_std must be positive"

    tangent_vectors = np.zeros((N, 2), dtype=np.float64)
    for i, ((y, x), branch_id) in enumerate(zip(nodes_coord, branches_id, strict=True)):
        pos = Point(y, x)
        window_rect = Rect.from_center(pos, 2 * (2 * gaussian_std + gaussian_offset)).clip(branch_label_map.shape)
        window = branch_label_map[window_rect.slice()]
        pos = pos - window_rect.top_left

        skel_points = np.where(window == branch_id if branch_id != 0 else window > 0)
        skel_points = np.array(skel_points, dtype=np.float64).T
        skel_points_d = pos.distance(skel_points)
        skel_points_w = np.exp(-0.5 * (skel_points_d - gaussian_offset) ** 2 / gaussian_std**2)
        barycenter = np.sum(skel_points * skel_points_w[:, None], axis=0) / np.sum(skel_points_w) - pos
        tangent_vectors[i] = -barycenter / np.linalg.norm(barycenter)

    return tangent_vectors


def compute_inflections_points(yx, dtheta_std=3, theta_std=2, angle_threshold=1.5, sampling=1):
    from scipy.signal import medfilt

    from ..math import gaussian_filter1d

    if len(yx) < 5:
        return np.array([])

    if sampling > 1:
        yx = yx[::sampling]

    dyx_smooth = curve_tangent(yx, std=theta_std)
    theta = np.arctan2(dyx_smooth[:, 0], dyx_smooth[:, 1]) * 180 / np.pi
    dtheta0 = (np.diff(theta) + 180) % 360 - 180
    dtheta = dtheta0

    dtheta = gaussian_filter1d(dtheta, dtheta_std)

    quant_dtheta = np.digitize(dtheta, [-angle_threshold, angle_threshold]) - 1
    quant_dtheta = medfilt(quant_dtheta, 5)
    v = quant_dtheta[0]
    points = {0: v}
    for i in range(1, len(quant_dtheta)):
        if quant_dtheta[i] != v:
            v = quant_dtheta[i]
            points[i] = v
    values = list(points.values())
    bins = list(points.keys()) + [len(yx)]
    bins_center = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    inflections = []

    for i in range(1, len(values) - 1):
        if values[i] == 0:
            inflections.append(bins_center[i])
    inflections = np.array(inflections).astype(int)

    return inflections
