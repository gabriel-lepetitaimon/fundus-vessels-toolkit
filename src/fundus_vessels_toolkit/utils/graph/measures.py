from typing import Iterable, List

import numpy as np
import torch

from ..cpp_extensions.graph_cpp import extract_branches_geometry as extract_branches_geometry_cpp
from ..cpp_extensions.graph_cpp import (
    extract_branches_geometry_from_skeleton as extract_branches_geometry_from_skeleton_cpp,
)
from ..cpp_extensions.graph_cpp import fast_branch_boundaries as fast_branch_boundaries_cpp
from ..cpp_extensions.graph_cpp import fast_curve_tangent as fast_curve_tangent_cpp
from ..cpp_extensions.graph_cpp import track_branches as track_branches_cpp
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
    adaptative_tangent: bool = True,
    bspline_max_error: float = 10,
    bspline_dK_threshold: float = 0.15,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Track branches from a labels map and extract their geometry.


    Parameters
    ----------
    branch_curves :
        A list of 2D tensors of shape (n, 2) containing the coordinates of the branch points.

    segmentation : torch.Tensor
        A 2D tensor of shape (H, W) containing the segmentation of the image.

    adaptative_tangent : bool, optional
        If True, the standard deviation of the gaussian weighting the curve points is set to the vessel calibre.

    bspline_max_error : float, optional
        The maximum error allowed for the bspline interpolation. By default 10.

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    This method returns a tuple containing three lists of length B which contains, for each branch:
    - a 2D tensor of shape (n, 2) containing the coordinates of the branch points (where n is the branch length).
    - a 2D tensor of shape (n, 2) containing the tangent vectors at each branch point.
    - a 2D tensor of shape (n,) containing the branch width (or calibre) at each branch point.

    If return_labels is True, the output will also contain:
        - a 2D tensor of shape (H,W) containing the branch labels after terminations cleaning.

    """
    options = dict(
        bspline_max_error=bspline_max_error,
        adaptative_tangent=adaptative_tangent,
        bspline_dK_threshold=bspline_dK_threshold,
    )
    branch_curves = [curve.cpu().int() for curve in branch_curves]

    return extract_branches_geometry_cpp(branch_curves, segmentation, options)


@autocast_torch
def extract_branch_geometry_from_skeleton(
    branch_labels: torch.Tensor,
    node_yx: torch.Tensor,
    edge_list: torch.Tensor,
    segmentation: torch.Tensor,
    clean_terminations: int = 20,
    return_labels: bool = False,
    adaptative_tangent: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Track branches from a labels map and extract their geometry.


    Parameters
    ----------
    branch_labels :
        A 2D tensor of shape (H, W) containing the skeleton where each branch has a unique label.

    node_yx :
        A 2D tensor of shape (N, 2) containing the coordinates (y, x) of the nodes.

    branch_list :
        A 2D tensor of shape (B, 2) containing for each branch, the indexes of the two nodes it connects.

    segmentation : torch.Tensor
        A 2D tensor of shape (H, W) containing the segmentation of the image.

    clean_terminations : int, optional
        The maximum number of pixels removable at branch termination. By default 20.

    return_labels : bool, optional

    Returns
    -------
    tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    This method returns a tuple containing three lists of length B which contains, for each branch:
    - a 2D tensor of shape (n, 2) containing the coordinates of the branch points (where n is the branch length).
    - a 2D tensor of shape (n, 2) containing the tangent vectors at each branch point.
    - a 2D tensor of shape (n,) containing the branch width (or calibre) at each branch point.

    If return_labels is True, the output will also contain:
        - a 2D tensor of shape (H,W) containing the branch labels after terminations cleaning.

    """
    options = dict(
        clean_terminations=float(clean_terminations), bspline_max_error=4, adaptative_tangent=adaptative_tangent
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
        The indexes of the points for which the tangent must be computed. By default None.

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
    branches_label_map: np.ndarray,
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

            The branch_id must follow the same indexing as the skeleton_map: especially the indexes must start at 1.

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
    assert branches_label_map.ndim == 2, "skeleton_map must be a 2D array"
    assert branches_label_map.shape[0] > 0 and branches_label_map.shape[1] > 0, "skeleton_map must be non empty"
    assert gaussian_offset > 0, "gaussian_offset must be positive"
    assert gaussian_std > 0, "gaussian_std must be positive"

    tangent_vectors = np.zeros((N, 2), dtype=np.float64)
    for i, ((y, x), branch_id) in enumerate(zip(nodes_coord, branches_id, strict=True)):
        pos = Point(y, x)
        window_rect = Rect.from_center(pos, 2 * (2 * gaussian_std + gaussian_offset)).clip(branches_label_map.shape)
        window = branches_label_map[window_rect.slice()]
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
