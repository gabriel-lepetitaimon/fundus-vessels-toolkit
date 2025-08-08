from typing import List, Tuple

import numpy as np
import torch

from .cpp_extensions.fvt_cpp import rasterize_branch as rasterize_branch_cpp
from .cpp_extensions.fvt_cpp import rasterize_topology as rasterize_topology_cpp
from .torch import autocast_torch


@autocast_torch
def rasterize_topology(
    branch_list: torch.Tensor,
    root_branches: torch.Tensor,
    curves: List[torch.Tensor],
    boundaries: List[torch.Tensor],
    shape: Tuple[int, int],
    N_nodes: int = -1,
    bridge_gap_smaller_than: float = 2,
    fill_junctions: bool = True,
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
        The tensors shape must be (N, 2) where is the length of the branch and each row contains the (y, x) coordinates of the branch centerline.
        The first point of each curve should be the root of the branch, the last should be the leaf.

    boundaries : List[torch.Tensor]
        A list of tensors, each representing the boundaries of a branch.
        The tensors shape must be (N, 2, 2) where N is the length of the branch. The second dimensions stores the left and right boundaries of the branch at each point as (y, x) coordinates.

    shape : Tuple[int, int]
        The shape of the output topology map.

    N_nodes : int, optional
        The number of nodes in the topology. Default is -1, which means it will be determined automatically.

    bridge_gap_smaller_than : float, optional
        A threshold for the bridge gap. Default is 2.

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

    if N_nodes == -1:
        N_nodes = int(branch_list.max().item() + 1)

    rasterize_topology_cpp(
        branch_list.cpu().int(),
        root_branches.cpu().int(),
        curves,
        boundaries,
        N_nodes,
        bridge_gap_smaller_than,
        fill_junctions,
        branchLabelsMap,
        topoMap,
    )
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
