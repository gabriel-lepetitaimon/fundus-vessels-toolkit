from typing import Iterable, List, Literal, Optional, Tuple, overload

import numpy as np
import torch  # Required for cpp extension loading

from .cpp_extensions.clusters_cpp import iterative_cluster_by_distance as iterative_cluster_by_distance_cpp
from .cpp_extensions.clusters_cpp import iterative_reduce_clusters as iterative_reduce_clusters_cpp
from .cpp_extensions.clusters_cpp import remove_consecutive_duplicates as remove_consecutive_duplicates_cpp
from .cpp_extensions.clusters_cpp import solve_1d_chains as solve_1d_chains_cpp
from .cpp_extensions.clusters_cpp import solve_clusters as solve_clusters_cpp
from .geometric import Point
from .torch import TensorArray, to_torch


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
            If ``drop_singleton`` is False, all indices from 0 to max(clusters) will be returned, even those omitted in ``clusters``.

    Returns
    -------
    List[List[int]]
        The reduced clusters.
    """  # noqa: E501
    clusters = [[int(_) for _ in c] for c in clusters]
    return [c for c in solve_clusters_cpp(clusters, drop_singleton) if len(c) > 0]


def iterative_reduce_clusters(edge_list: TensorArray, edge_weight: TensorArray, max_weight: float) -> List[List[int]]:
    """
    Reduce the number of clusters by merging clusters that share at least one element.
    """
    clusters = iterative_reduce_clusters_cpp(
        to_torch(edge_list, dtype=torch.int), to_torch(edge_weight, dtype=torch.float), max_weight
    )
    return [list(_) for _ in clusters]


@overload
def remove_consecutive_duplicates(array: TensorArray, return_index: Literal[False] = False) -> TensorArray: ...
@overload
def remove_consecutive_duplicates(
    array: TensorArray, return_index: Literal[True]
) -> Tuple[TensorArray, TensorArray]: ...
def remove_consecutive_duplicates(
    array: TensorArray, return_index: bool = False
) -> TensorArray | Tuple[TensorArray, TensorArray]:
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
    is_torch = isinstance(array, torch.Tensor)
    tensor = to_torch(array, dtype=torch.int)
    if return_index:
        result, ids = remove_consecutive_duplicates_cpp(tensor, True)
        return (result, ids) if is_torch else (result.numpy(force=True), ids.numpy(force=True))
    result, _ = remove_consecutive_duplicates_cpp(tensor, False)
    return result if is_torch else result.numpy(force=True)


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


def cluster_by_distance(
    coords: TensorArray | List[Point],
    max_distance: float,
    edge_list: Optional[TensorArray | List[Tuple[int, int]]] = None,
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
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
    else:
        coords_tensor = to_torch(coords, dtype=torch.float32)
    assert coords_tensor.ndim == 2 and coords_tensor.shape[1] == 2, "Coordinates must be a 2D tensor of shape (n, 2)"

    edge_list_tensor = None
    if edge_list is not None:
        if isinstance(edge_list, list):
            edge_list_tensor = torch.tensor(edge_list, dtype=torch.int32)
        else:
            edge_list_tensor = to_torch(edge_list, dtype=torch.int32)
        assert (
            edge_list_tensor.ndim == 2 and edge_list_tensor.shape[1] == 2
        ), "Edge list must be a 2D tensor of shape (n, 2)"

    if iterative:
        if edge_list_tensor is None:
            edge_list_tensor = torch.empty(0, 2, dtype=torch.int32)
        clusters = iterative_cluster_by_distance_cpp(coords_tensor, max_distance, edge_list_tensor)
        return [list(_) for _ in clusters if len(_) > 1]
    else:
        if edge_list_tensor is None:
            dist = torch.cdist(coords_tensor, coords_tensor)
            edge_list_tensor = torch.nonzero(dist <= max_distance, as_tuple=False)
            edge_list_tensor = edge_list_tensor[edge_list_tensor[:, 0] < edge_list_tensor[:, 1]]
        else:
            edge_list_tensor = edge_list_tensor[
                torch.norm(coords_tensor[edge_list_tensor[:, 0]] - coords_tensor[edge_list_tensor[:, 1]], dim=1)
                <= max_distance
            ]
        return reduce_clusters(edge_list_tensor, drop_singleton=False)  # type: ignore
