from typing import Dict, Tuple

import numpy as np


def add_empty_to_lookup(lookup: np.ndarray) -> np.ndarray:
    """
    Add an empty entry to a lookup table: insert a 0 at the beginning of the array and increment all other values by 1.
    """
    return np.concatenate([[0], lookup + 1], dtype=lookup.dtype)


def apply_lookup(
    array: np.ndarray | None,
    mapping: Dict[int, int] | Tuple[np.ndarray, np.ndarray] | np.array | None,
    apply_inplace_on: np.ndarray | None = None,
) -> np.ndarray:
    lookup = mapping
    if mapping is None:
        return array
    if not isinstance(mapping, np.ndarray):
        lookup = np.arange(len(array), dtype=np.int64)
        if isinstance(mapping, dict):
            mapping = tuple(zip(*mapping.items(), strict=True))
        search = mapping[0]
        replace = mapping[1]
        if not isinstance(replace, np.ndarray):
            replace = [replace] * len(search)
        for s, r in zip(search, replace, strict=False):
            lookup[lookup == s] = r

    if apply_inplace_on is not None:
        apply_inplace_on[:] = lookup[apply_inplace_on]

    return lookup if array is None else lookup[array]


def apply_lookup_on_coordinates(points_coord, lookup: np.ndarray | None, weight: np.ndarray | None = None):
    """
    Apply a lookup table on a set of coordinates to merge specific points and return their barycenter.

    Parameters
    ----------
    points_coord:
        A tuple (y, x) of two 1D arrays of the same length N containing the points coordinates.

    lookup:
        A 1D array of length N containing a lookup table of points index. Points with the same index in this table will be merged.

    weight:
        A 1D array of length N containing the weight of each point. Points with the same index in the lookup table will be merged with a weighted average.


    Returns
    -------
        A tuple (y, x) of two 1D arrays of length M (max(lookup) + 1) containing the merged coordinates.

    """
    if lookup is None:
        return points_coord
    if weight is None or weight.sum() == 0:
        weight = np.ones_like(lookup)
    weight = weight + 1e-8

    assert len(points_coord) == 2, f"nodes_coord must be a tuple of two 1D arrays. Got {len(points_coord)} elements."
    assert len(points_coord[0]) == len(points_coord[1]) == len(lookup) == len(weight), (
        f"nodes_coord, nodes_lookup and nodes_weight must have the same length. Got {len(points_coord[0])}, "
        f"{len(lookup)} and {len(weight)} respectively."
    )

    jy, jx = points_coord
    points_coord = np.zeros((np.max(lookup) + 1, 2), dtype=np.float64)
    points_total_weight = np.zeros((np.max(lookup) + 1,), dtype=np.float64)
    np.add.at(points_coord, lookup, np.asarray((jy, jx)).T * weight[:, None])
    np.add.at(points_total_weight, lookup, weight)
    points_coord = points_coord / points_total_weight[:, None]
    return points_coord.T


def invert_lookup(lookup):
    """
    Invert a lookup array. The lookup array must be a 1D array of integers. The output array is a 1D array of length
    max(lookup) + 1. The output array[i] contains the list of indices of lookup where lookup[index] == i.
    """
    splits = np.split(np.argsort(np.argsort(lookup)), np.cumsum(np.unique(lookup, return_counts=True)[1]))[:-1]
    return np.array([np.array(s[0], dtype=np.int64) for s in splits])
