from typing import List

import torch  # Required for cpp extension loading

from .cpp_extensions.fvt_cpp import find_cycles as find_cycles_cpp
from .cpp_extensions.fvt_cpp import has_cycle as has_cycle_cpp
from .torch import autocast_torch


@autocast_torch
def has_cycle(parents: torch.Tensor) -> bool:
    """
    Find cycles in a graph.

    Parameters
    ----------
    parents : torch.Tensor
        A tensor of shape (N,) containing the parent of each node.

    Returns
    -------
    List[torch.Tensor]
        A list of cycles. The order of the nodes in the cycles is not deterministic.
    """
    parents = torch.as_tensor(parents).cpu().int()
    return has_cycle_cpp(parents)


@autocast_torch
def find_cycles(parents: torch.Tensor) -> List[List[int]]:
    """
    Find the root of a tree.

    Parameters
    ----------
    parents : torch.Tensor
        A tensor of shape (N,) containing the parent of each node.

    Returns
    -------
    int
        The root of the tree.
    """
    parents = torch.as_tensor(parents).cpu().int()
    return find_cycles_cpp(parents)
