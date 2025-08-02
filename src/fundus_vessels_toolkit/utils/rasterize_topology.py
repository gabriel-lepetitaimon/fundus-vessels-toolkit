from typing import Tuple

import torch

from .cpp_extensions.fvt_cpp import rasterize_branch as rasterize_branch_cpp
from .cpp_extensions.fvt_cpp import rasterize_topology as rasterize_topology_cpp
from .torch import autocast_torch


@autocast_torch
def rasterize_topology(
    branch_list: torch.Tensor,
    root_branches: torch.Tensor,
    curves: torch.Tensor,
    boundaries: torch.Tensor,
    shape: Tuple[int, int],
    N_nodes=-1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterizes the topology of branches given their curves and boundaries.

    Parameters
    ----------
    branch_list : torch.Tensor
        A tensor containing the list of branches, where each branch is represented by its ID.

    root_branches : torch.Tensor
        A tensor containing the root branches, where each root branch is represented by its ID.

    curves : List[torch.Tensor]
        A list of tensors, each representing a curve of a branch.

    boundaries : List[torch.Tensor]
        A list of tensors, each representing the boundaries of a branch.

    shape : tuple
        The shape of the output topology map.

    N_nodes : int, optional
        The number of nodes in the topology. Default is -1, which means it will be determined automatically.

    Returns
    -------
    tuple
        A tuple containing two tensors:
        - branchLabelsMap: A tensor of shape `shape` containing the labels of the branches.
        - topoMap: A tensor of shape `shape` containing the topology information.
    """
    branchLabelsMap = torch.zeros(shape, dtype=torch.int32)
    topoMap = torch.zeros(shape, dtype=torch.int32)
    rasterize_topology_cpp(branch_list, root_branches, curves, boundaries, branchLabelsMap, topoMap, N_nodes)
    return branchLabelsMap, topoMap


@autocast_torch
def rasterize_branch(
    curve: torch.Tensor,
    boundaries: torch.Tensor,
    branchID: int,
    branchRank: float,
    branchLabelsMap: torch.Tensor,
    topoMap: torch.Tensor,
    bridge_gap_smaller_than_sqr: float = 2,
) -> None:
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
    rasterize_branch_cpp(curve, boundaries, branchID, branchRank, branchLabelsMap, topoMap, bridge_gap_smaller_than_sqr)
