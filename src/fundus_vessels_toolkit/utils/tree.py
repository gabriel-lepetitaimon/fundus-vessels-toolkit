from typing import List

import torch  # Required for cpp extension loading

from .cpp_extensions.fvt_cpp import find_cycles as find_cycles_cpp
from .cpp_extensions.fvt_cpp import has_cycle as has_cycle_cpp
from .cpp_extensions.fvt_cpp import node_accessible_from_root as node_accessible_from_root_cpp
from .cpp_extensions.fvt_cpp import tree_distance as tree_distance_cpp
from .torch import TensorArray, autocast_torch


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


@autocast_torch
def accessible_from_root(edge_index: torch.Tensor, N: int, root: int = 0) -> torch.Tensor:
    """
    Find the nodes accessible from the root in a graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        A tensor of shape (2, E) containing the edges of the graph.
    N : int
        The number of nodes in the graph.
    root : int
        The root node.

    Returns
    -------
    torch.Tensor
        A boolean tensor of shape (N,) indicating whether each node is accessible from the root.
    """
    edge_index_tensor = torch.as_tensor(edge_index, device="cpu")
    if root_is_minus_one := root == -1:
        edge_index_tensor = edge_index_tensor + 1
        N += 1
        root = 0
    accessible_tensor = node_accessible_from_root_cpp(edge_index_tensor.to(torch.uint32), N, root)
    if root_is_minus_one:
        accessible_tensor = accessible_tensor[1:]
    return accessible_tensor


@autocast_torch
def tree_distance(tree_list: torch.Tensor) -> torch.Tensor:
    """Compute distance matrices between each pair of node of the tree.

    Parameters
    ----------
    tree_list: torch.Tensor
        A tensor of shape (N,) containing the parent index of each node. The root node should have a parent index of -1.

    Returns
    -------
    path_distance: torch.Tensor [N, N]
        The topological distance (i.e., the length of the shortest path) between each pair of node.

    common_ancestor_distance: torch.Tensor [N, N]
        The distance to the closest common ancestor of each pair of node. Distance from a parent node to its descendent are stored negatively, i.e. if node A is a child of B, dist[A][B] = 1 and dist[B][A] = -1.

    """  # noqa: E501
    return tree_distance_cpp(tree_list)
