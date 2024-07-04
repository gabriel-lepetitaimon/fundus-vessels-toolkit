from typing import List, Set

import numpy as np
import torch  # Required cpp extension loading  # noqa: F401

from ..cpp_extensions.clusters_cpp import solve_1d_chains as solve_1d_chains_cpp
from ..cpp_extensions.clusters_cpp import solve_clusters as solve_clusters_cpp
from ..geometric import Point


def reduce_clusters(clusters: List[Set[int]]) -> List[Set[int]]:
    """
    Reduce the number of clusters by merging clusters that share at least one element.
    """
    clusters = [list(c) for c in clusters]
    clusters = solve_clusters_cpp(clusters, True)
    return clusters


def reduce_chains(chains: List[List[int]]) -> List[List[int]]:
    """
    Reduce the number of chains by merging chains that share at least one element.
    """
    chains = [list(c) for c in chains]
    try:
        chains = solve_1d_chains_cpp(chains)
    except SystemError:
        raise ValueError("Chains must be non-overlapping.") from None
    return chains


def cluster_by_distance(coords: List[Point] | np.ndarray, max_distance: int) -> List[Set[int]]:
    """
    Cluster a set of coordinates by distance: all coordinates that are at most `max_distance` apart are in the same cluster.

    Parameters
    ----------
    coords : List[Point] | np.ndarray
        The coordinates to cluster.


    """
    if isinstance(coords, list):
        coords = np.asarray(coords, dtype=np.int32)
    distance = np.linalg.norm(coords[:, None] - coords, axis=-1)
    identity = np.eye(len(coords), dtype=bool)
    clusters = [set(_) for _ in np.argwhere((distance <= max_distance) & (~identity))]
    return reduce_clusters(clusters)
