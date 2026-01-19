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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterizes the topology of branches given their curves and boundaries.

    Parameters
    ----------
    branch_list : torch.Tensor
        A tensor containing the list of branches, where each branch is represented by its ID.

    branch_tree : torch.Tensor
        A tensor containing the branch tree, where each branch's parent is represented by its ID.

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

    Returns
    -------
    tuple
        A tuple containing two tensors:
        - branchLabelsMap: A tensor of shape `shape` containing the labels of the branches.
        - topoMap: A tensor of shape `shape` containing the topology information.
    """  # noqa: E501
    branchLabelsMap = torch.from_numpy(np.zeros(shape, dtype=np.int32)).int()
    topoMap = torch.from_numpy(np.zeros(shape, dtype=np.float32))

    rasterize_topology_cpp(
        branch_list.cpu().int(),
        branch_tree.cpu().int(),
        branch_dirs.cpu().bool(),
        [c.cpu().int() for c in curves],
        [b.cpu().int() for b in boundaries],
        nodes_yx.cpu().int(),
        bezier_interpolate if isinstance(bezier_interpolate, float) else (0.5 if bezier_interpolate else -1.0),
        fill_junctions,
        branchLabelsMap,
        topoMap,
    )
    return branchLabelsMap, topoMap


@autocast_torch
def rasterize_branch(
    curve: torch.Tensor,
    boundaries: torch.Tensor,
    out: torch.Tensor | tuple[int, int],
    fill_value: int = 1,
    bridge_gap_smaller_than: float = 2,
) -> torch.Tensor:
    """
    Rasterizes a branch given its curve and boundaries.

    Parameters
    ----------

    curve (torch.Tensor):
        The curve of the branch.

    boundaries (torch.Tensor):
        The boundaries of the branch.

    branchID (int):
        The ID of the branch.

    branchRank (float):
        The rank of the branch.

    branchLabelsMap (torch.Tensor):
        The map to store branch labels.

    topoMap (torch.Tensor):
        The topology map to update.

    bridge_gap_smaller_than_sqr (float):
        Threshold for bridge gap.

    Returns:
        None
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

    return discretize_line_cpp((p0[1], p0[0]), (p1[1], p1[0])).numpy(force=True)  # Flip to (y, x)
