from typing import Iterable, List, Literal, Optional, Tuple, overload

import numpy as np
import torch  # Required for cpp extension loading

from .cpp_extensions.clusters_cpp import iterative_cluster_by_distance as iterative_cluster_by_distance_cpp
from .cpp_extensions.clusters_cpp import iterative_reduce_clusters as iterative_reduce_clusters_cpp
from .cpp_extensions.clusters_cpp import remove_consecutive_duplicates as remove_consecutive_duplicates_cpp
from .cpp_extensions.clusters_cpp import solve_1d_chains as solve_1d_chains_cpp
from .cpp_extensions.clusters_cpp import solve_clusters as solve_clusters_cpp
from .geometric import Point
from .torch import autocast_torch


def reduce_clusters(clusters: Iterable[Iterable[int]], drop_singleton=True) -> List[List[int]]:
    """
    Reduce the number of clusters by merging clusters that share at least one element.

    Parameters
    ----------
    clusters : Iterable[Iterable[int]]
        The input clusters.

    drop_singleton : bool, optional
        Whether to drop clusters with only one element. By default True.

        .. warning::
            If ``drop_singleton`` is False, all indexes from 0 to max(clusters) will be returned, even those omitted in ``clusters``.

    Returns
    -------
    List[List[int]]
        The reduced clusters.
    """  # noqa: E501
    clusters = [[int(_) for _ in c] for c in clusters]
    return [c for c in solve_clusters_cpp(clusters, drop_singleton) if len(c) > 0]


@autocast_torch
def iterative_reduce_clusters(edge_list: torch.Tensor, edge_weight: torch.Tensor, max_weight: float) -> List[List[int]]:
    """
    Reduce the number of clusters by merging clusters that share at least one element.
    """
    edge_list = edge_list.cpu().int()
    edge_weight = edge_weight.cpu().float()
    clusters = iterative_reduce_clusters_cpp(edge_list, edge_weight, max_weight)
    return [list(_) for _ in clusters]


@autocast_torch
def remove_consecutive_duplicates(
    tensor: torch.Tensor, return_index: bool = False
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove consecutive duplicates vectors in a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor. Must be 2D.

    return_index : bool, optional
        Whether to return the index of the last element of each chain. By default False.

    Returns
    -------
    torch.Tensor
        The tensor with consecutive duplicates removed.
    """
    tensor = tensor.cpu().int()
    if return_index:
        tensor, ids = remove_consecutive_duplicates_cpp(tensor, True)
        return tensor, ids
    return remove_consecutive_duplicates_cpp(tensor, False)[0]


@overload
def reduce_chains(chains: List[List[int]], *, return_index: Literal[False] = False) -> List[List[int]]: ...
@overload
def reduce_chains(
    chains: List[List[int]], *, return_index: Literal[True]
) -> Tuple[List[List[int]], List[List[int]]]: ...
def reduce_chains(
    chains: List[List[int]], *, return_index: bool = False
) -> List[List[int]] | Tuple[List[List[int]], List[List[int]]]:
    """
    Reduce the number of chains by merging chains that share at least one element.

    Examples
    --------
    >>> reduce_chains([[1, 2], [2, 3, 4], [5, 4], [6, 7]])
    [[1, 2, 3, 4, 5], [6, 7]]

    >>> reduce_chains([[1, 2], [3, 2], [4,3], [4, 5], [6, 7]], return_index=True)
    ([[1, 2, 3, 4, 5], [6, 7]], [[1, -2, -3, 4], [5]])
    """
    chains = [list(c) for c in chains]
    try:
        chains, chains_id = solve_1d_chains_cpp(chains)
    except SystemError:
        raise ValueError("Chains must be non-overlapping.") from None
    return (chains, chains_id) if return_index else chains


@autocast_torch
def cluster_by_distance(
    coords: torch.Tensor | List[Point],
    max_distance: float,
    edge_list: Optional[torch.Tensor | List[Tuple[int, int]]] = None,
    iterative=False,
) -> List[List[int]]:
    """
    Cluster a set of coordinates by distance: all coordinates that are at most `max_distance` apart are in the same cluster.

    Parameters
    ----------
    coords : List[Point] | np.ndarray
        The coordinates of the points to cluster.

    max_distance : float
        The maximum distance between points in the same cluster.

    edge_list : torch.Tensor | List[Tuple[int, int]], optional
        The edge list of the graph. If not provided, it will be computed from the coordinates. By default None.

    iterative : bool, optional
        Whether to use the iterative algorithm: at each step the algorithm will merge the two closest clusters until the clusters are at least `max_distance` apart (according to their barycenter). By default False.

    Returns
    -------
    List[List[int]]
        The clusters.
    """  # noqa: E501
    if isinstance(coords, list):
        coords = torch.tensor(coords)
    assert coords.ndim == 2 and coords.shape[1] == 2, "Coordinates must be a 2D tensor of shape (n, 2)"
    coords = coords.float()

    if edge_list is not None:
        if isinstance(edge_list, list):
            edge_list = torch.tensor(edge_list)
        assert edge_list.ndim == 2 and edge_list.shape[1] == 2, "Edge list must be a 2D tensor of shape (n, 2)"
        edge_list = edge_list.int()

    if iterative:
        if edge_list is None:
            edge_list = torch.empty(0, 2, dtype=torch.int32)
        else:
            edge_list = edge_list.cpu()
        clusters = iterative_cluster_by_distance_cpp(coords.cpu(), max_distance, edge_list)
        return [list(_) for _ in clusters]
    else:
        if edge_list is None:
            dist = torch.cdist(coords, coords)
            edge_list = torch.nonzero(dist <= max_distance, as_tuple=False)
            edge_list = edge_list[edge_list[:, 0] < edge_list[:, 1]]
        else:
            edge_list = edge_list[torch.norm(coords[edge_list[:, 0]] - coords[edge_list[:, 1]], dim=1) <= max_distance]
        return reduce_clusters(edge_list, drop_singleton=False)
