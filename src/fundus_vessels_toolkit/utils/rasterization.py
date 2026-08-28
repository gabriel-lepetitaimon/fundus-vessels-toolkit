from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from .cpp_extensions.fvt_cpp import rasterize_branch as rasterize_branch_cpp
from .cpp_extensions.fvt_cpp import rasterize_topology as rasterize_topology_cpp
from .torch import autocast_torch


@autocast_torch
def rasterize_topology(
    branch_list: torch.Tensor,
    branch_tree: torch.Tensor,
    branch_dirs: torch.Tensor,
    curves: list[torch.Tensor],
    boundaries: list[torch.Tensor],
    nodes_yx: torch.Tensor,
    shape: tuple[int, int],
    fill_junctions: bool = True,
    bezier_interpolate: bool | float = 0.5,
    expand: float = 0.0,
    branch_mapping: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rasterizes the topology of branches given their curves and boundaries.

    Parameters
    ----------
    branch_list : torch.Tensor
        The branch list as a tensor of shape (num_branches, 2), where each row contains the (start_node_id, end_node_id) of a branch.

    branch_tree : torch.Tensor
        The branch tree as a tensor of shape (num_branches,), where each row contains the index of the parent branch (or -1 if it's a root branch).

    branch_dirs : torch.Tensor
        A tensor containing the direction of each branch.

    curves : list[torch.Tensor]
        A list of tensors, each representing a curve of a branch.
        The tensors shape must be (N, 2) where is the length of the branch and each row contains the (y, x) coordinates of the branch centerline.
        The first point of each curve should be the root of the branch, the last should be the leaf.

    boundaries : list[torch.Tensor]
        A list of tensors, each representing the boundaries of a branch.
        The tensors shape must be (N, 2, 2) where N is the length of the branch. The second dimensions stores the left and right boundaries of the branch at each point as (y, x) coordinates.

    nodes_yx : torch.Tensor
        A tensor containing the (y, x) coordinates of the nodes in the vascular tree.

    shape : tuple[int, int]
        The shape of the output topology map.

    bezier_interpolate: bool | float = 0.5, optional
        If a float is provided, it indicates the interpolation step for discretizing Bezier curves.
        If True, a default step of 0.5 is used. If False, no interpolation is performed.

    fill_junctions : bool, optional
        A flag indicating whether to fill junctions in the topology. Default is True.

    expand : float, optional
        A float indicating the distance to expand (dilate) the branch boundaries. Default is 0.0

    branch_mapping : Optional[torch.Tensor], optional
        A tensor of shape (num_branches,) that maps each branch ID to a new label. If provided, the rasterized branch labels will be replaced with the corresponding values from this mapping.

    Returns
    -------
    tuple
        A tuple containing three tensors:
        - branchLabelsMap: A tensor of shape `shape` containing the labels of the branches.
        - topoMap: A tensor of shape `shape` containing the topology information.
        - fuzzySkeletonMap: A tensor of shape `shape` containing the distance to the skeleton.
    """  # noqa: E501
    branchLabelsMap = torch.from_numpy(np.zeros(shape, dtype=np.int64)).long()
    topoMap = torch.from_numpy(np.zeros(shape, dtype=np.float32))
    fuzzySkeletonMap = torch.from_numpy(np.zeros(shape, dtype=np.float32))

    if branch_mapping is None:
        branch_mapping = torch.empty((0,), dtype=torch.int64)

    rasterize_topology_cpp(
        branch_list.cpu().int(),
        branch_tree.cpu().int(),
        branch_dirs.cpu().bool(),
        [c.cpu().int() for c in curves],
        [b.cpu().int() for b in boundaries],
        nodes_yx.cpu().int(),
        bezier_interpolate if isinstance(bezier_interpolate, float) else (0.5 if bezier_interpolate else -1.0),
        fill_junctions,
        expand,
        branch_mapping.cpu().long(),
        branchLabelsMap,
        topoMap,
        fuzzySkeletonMap,
    )
    return branchLabelsMap, topoMap, fuzzySkeletonMap


@autocast_torch
def rasterize_branch(
    curve: torch.Tensor,
    boundaries: torch.Tensor,
    out: torch.Tensor | tuple[int, int],
    fill_value: int = 1,
    bridge_gap_smaller_than: float = 2,
) -> torch.Tensor:
    """
    Rasterize a branch given its curve and boundaries.

    Parameters
    ----------

    curve (torch.Tensor):
        The curve of the branch.

    boundaries (torch.Tensor):
        The boundaries of the branch.

    out (torch.Tensor | tuple[int, int]):
        The output tensor or the shape of the output tensor.

    fill_value (int):
        The value to fill the rasterized branch with.

    bridge_gap_smaller_than_sqr (float):
        Threshold for bridge gap.

    Returns:
    torch.Tensor:
        The map with the rasterized branch.
    """
    if isinstance(out, torch.Tensor):
        assert out.dtype == torch.int32, "The output tensor must be of type torch.int32."
        assert out.dim() == 2, "The output tensor must be 2-dimensional."
        outTensor = out
    else:
        outTensor = torch.from_numpy(np.zeros(out, dtype=np.int32))
    curve = curve.cpu().int()
    boundaries = boundaries.cpu().int()

    return rasterize_branch_cpp(curve, boundaries, outTensor, fill_value, bridge_gap_smaller_than)


def rasterize_line(
    p0: tuple[int, int],
    p1: tuple[int, int],
) -> npt.NDArray[np.int64]:
    """
    Rasterizes a line between two points.

    Parameters
    ----------
    p0 : tuple[int, int]
        The starting point of the line (y, x).

    p1 : tuple[int, int]
        The ending point of the line (y, x).

    Returns
    -------
    torch.Tensor
        A tensor containing the (y, x) coordinates of the rasterized line.
    """
    from .cpp_extensions.fvt_cpp import discretize_line as discretize_line_cpp

    return discretize_line_cpp((p0[0], p0[1]), (p1[0], p1[1])).numpy(force=True)


@autocast_torch
def draw_lines(p0: torch.Tensor, p1: torch.Tensor, out: tuple[int, int] | torch.Tensor) -> torch.Tensor:
    """
    Draws lines between pairs of points on a raster grid.

    Parameters
    ----------
    p0 : torch.Tensor
        A tensor of shape (N, 2) containing the starting points of the lines (y, x).

    p1 : torch.Tensor
        A tensor of shape (N, 2) containing the ending points of the lines (y, x).

    out : tuple[int, int] | torch.Tensor
        The shape of the output tensor or an existing boolean tensor to draw on.

    Returns
    -------
    torch.Tensor
        A tensor of shape `shape` with lines drawn between the specified points.
    """
    from .cpp_extensions.fvt_cpp import drawLines as draw_lines_cpp

    if isinstance(out, torch.Tensor):
        assert out.dtype == torch.bool, "The output tensor must be of type torch.bool."
        assert out.dim() == 2, "The output tensor must be 2-dimensional."
        outTensor = out
    else:
        outTensor = torch.from_numpy(np.zeros(out, dtype=np.bool_))

    return draw_lines_cpp(p0.cpu().int(), p1.cpu().int(), outTensor)
