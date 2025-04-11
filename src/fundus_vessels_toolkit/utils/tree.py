from typing import List

import torch  # Required for cpp extension loading

from .cpp_extensions.fvt_cpp import find_cycles as find_cycles_cpp
from .cpp_extensions.fvt_cpp import has_cycle as has_cycle_cpp
from .torch import TensorArray


def has_cycle(parents: TensorArray) -> bool:
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
    parent_tensors = torch.as_tensor(parents, device="cpu", dtype=torch.int)
    return has_cycle_cpp(parent_tensors)


def find_cycles(parents: TensorArray) -> List[List[int]]:
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
    parents_tensor = torch.as_tensor(parents, device="cpu", dtype=torch.int)
    return find_cycles_cpp(parents_tensor)
