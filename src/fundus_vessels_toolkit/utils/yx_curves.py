import numpy as np
import torch

from .cpp_extensions.geometric_parsing_cpp import track_branches as track_branches_cpp
from .math import gaussian, gaussian_filter1d


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
    cast_numpy = isinstance(edge_labels_map, np.ndarray)
    if cast_numpy:
        edge_labels_map = torch.from_numpy(edge_labels_map)
    edge_labels_map = edge_labels_map.int()

    if isinstance(nodes_yx, np.ndarray):
        nodes_yx = torch.from_numpy(nodes_yx)
    nodes_yx = nodes_yx.int()

    if isinstance(edge_list, np.ndarray):
        edge_list = torch.from_numpy(edge_list)
    edge_list = edge_list.int()

    out = track_branches_cpp(edge_labels_map, nodes_yx, edge_list)

    return [_.numpy() for _ in out] if cast_numpy else out


def compute_extremities_tangent(yx, theta_std=2):
    start_dyx = yx[: theta_std * 3] - yx[0]
    end_dyx = yx[-theta_std * 3 :][::-1] - yx[-1]

    start_w = gaussian(np.linalg.norm(start_dyx, axis=1), theta_std)
    end_w = gaussian(np.linalg.norm(end_dyx, axis=1), theta_std)

    start_theta = np.sum(start_dyx * start_w[:, None], axis=0) / start_w.sum()
    end_theta = np.sum(end_dyx * end_w[:, None], axis=0) / end_w.sum()

    return start_theta, end_theta


def compute_tangent(curve_yx, point_id=None, std=2):
    from scipy.signal import convolve

    sigma_range = np.arange(-3 * std, 3 * std + 1)
    dyx = np.empty_like(curve_yx, dtype=float)
    curve_yx = curve_yx.copy().astype(float)
    curve_yx[1:-1, 0] = convolve(curve_yx[:, 0], np.array([1, 0, 1]) / 2, mode="valid")
    curve_yx[1:-1, 1] = convolve(curve_yx[:, 1], np.array([1, 0, 1]) / 2, mode="valid")

    if isinstance(point_id, int):
        point_id = [point_id]

    for i in range(len(curve_yx)) if point_id is None else point_id:
        window_dyx = np.concatenate(
            [
                curve_yx[i] - curve_yx[[_ for _ in i + np.arange(-3 * std, 0) if 0 <= _ < len(curve_yx)]],
                curve_yx[[_ for _ in i + np.arange(0, 3 * std + 1) if 0 <= _ < len(curve_yx)]] - curve_yx[i],
            ]
        )
        weight_yx = gaussian(np.linalg.norm(window_dyx, axis=1), std)
        dyx[i] = np.sum(window_dyx * weight_yx[:, None], axis=0) / weight_yx.sum()

    return dyx if point_id is None else dyx[point_id]


def compute_inflections_points(yx, diff_offset=3, dtheta_std=10, theta_std=2, sampling=1, true_inflections=False):
    from scipy.signal import medfilt

    if len(yx) < 5:
        return np.array([])

    if sampling > 1:
        yx = yx[::sampling]

    dyx = yx[diff_offset:] - yx[:-diff_offset]
    theta = np.arctan2(dyx[:, 0], dyx[:, 1])
    dtheta = np.diff(theta) * 4 / np.pi
    dtheta = (dtheta + 4) % 8 - 4

    dtheta = gaussian_filter1d(dtheta, dtheta_std)

    if true_inflections:
        inflections = []
        for i in range(1, len(dtheta)):
            if np.sign(dtheta[i]) != np.sign(dtheta[i - 1]):
                inflections.append(i)

        return yx[np.array(inflections).astype(int) + 1]

    avg_dtheta = np.digitize(dtheta, [-0.01, 0.01]) - 1
    dtheta = medfilt(avg_dtheta, 5)
    v = avg_dtheta[0]
    points = {0: v}
    for i in range(1, len(avg_dtheta)):
        if avg_dtheta[i] != v:
            v = avg_dtheta[i]
            points[i] = v
    values = list(points.values())
    bins = list(points.keys()) + [len(yx)]
    bins_center = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    inflections = []

    for i in range(1, len(values) - 1):
        if values[i] == 0:
            inflections.append(bins_center[i])
    inflections = np.array(inflections).astype(int)

    return inflections + 1
