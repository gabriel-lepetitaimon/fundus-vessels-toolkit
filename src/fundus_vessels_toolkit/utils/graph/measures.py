from typing import Iterable

import numpy as np

from ..geometric import Point, Rect


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
