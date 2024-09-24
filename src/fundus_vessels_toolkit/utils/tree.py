import torch  # Required for cpp extension loading

from .cpp_extensions.trees_cpp import has_cycle as has_cycle_cpp
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
