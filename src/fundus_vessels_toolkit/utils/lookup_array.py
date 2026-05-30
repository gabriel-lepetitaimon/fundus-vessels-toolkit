import warnings
from typing import Dict, Literal, Mapping, Optional, Tuple, TypeVar, overload

import numpy as np
import numpy.typing as npt

from fundus_toolkits.utils.typing import Int1DArray, Int1DArrayLike, as_int_1d

from .binary_mask import index_to_mask

T = TypeVar("T")
DTYPE = TypeVar("DTYPE", bound=np.number)


def add_empty_to_lookup(lookup: npt.NDArray[DTYPE], increment_index=True) -> npt.NDArray[DTYPE]:
    """
    Add an empty entry to a lookup table: insert a 0 at the beginning of the array and increment all other values by 1.
    """
    dtype = lookup.dtype
    if increment_index:
        return np.concatenate([np.zeros(1, dtype=dtype), lookup + np.ones((), dtype=dtype)], dtype=dtype)
    else:
        return np.concatenate([-np.ones(1, dtype=dtype), lookup], dtype=dtype)


def apply_lookup(
    array: np.ndarray,
    mapping: Dict[int, int] | Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]] | npt.NDArray[np.int_],
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
    try:
        return lookup if array is None else lookup[array]
    except IndexError as e:
        if len(lookup) < array.max() + 1:
            raise ValueError(
                f"Lookup table is too short. The maximum value in the array is {array.max()}, "
                f"but the lookup table has only {len(lookup)} elements instead of {array.max() + 1}."
            ) from None
        raise e


def lookup_from_mapping(
    mapping: Mapping[int, int] | Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]], lookup_length: int
) -> Int1DArray:
    """
    Create a lookup table from a mapping.


    """
    if isinstance(mapping, tuple):
        search, replace = mapping
    elif isinstance(mapping, Mapping):
        search = np.array(list(mapping.keys()), dtype=np.int_)
        replace = np.array(list(mapping.values()), dtype=np.int_)
    else:
        raise TypeError(f"Unsupported mapping type: {type(mapping)}")

    lookup = np.arange(lookup_length, dtype=np.int_)
    lookup[search] = replace
    return lookup


def apply_lookup_on_dict(d: Mapping[int, T], lookup: npt.NDArray[np.int_] | Mapping[int, int]) -> Dict[int, T]:
    """
    Apply a lookup table on a dictionary. The keys of the dictionary are used as indices in the lookup table.
    """
    if isinstance(lookup, Mapping):
        new_keys = [lookup.get(k, k) for k in d.keys()]
    else:
        lookup = np.asarray(lookup).astype(int)
        new_keys = lookup[np.array(list(d.keys()))]
    return {k: v for k, v in zip(new_keys, d.values(), strict=True) if k >= 0}


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

    """  # noqa: E501
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


def complete_lookup(lookup: Int1DArrayLike, max_index: int, assume_valid=False) -> Int1DArray:
    """
    Complete a lookup table to have a full range of indices from 0 to max_index.
    """
    lookup = as_int_1d(lookup)
    if len(lookup) == 0:
        warnings.warn("The lookup table is empty.", stacklevel=2)
        return np.arange(max_index + 1)

    assert len(lookup) <= max_index + 1, (
        f"lookup must have less than {max_index + 1} elements but has {lookup.shape[0]}."
    )
    if not assume_valid:
        lookup_sorted, lookup_counts = np.unique(lookup, return_counts=True)
        if lookup_sorted[0] < 0:
            raise ValueError("The lookup table contains negative values.")
        if lookup_sorted[-1] > max_index:
            raise ValueError("The lookup table contains values greater than or equal to max_index.")
        if np.any(lookup_counts > 1):
            raise ValueError("The lookup table contains duplicate values.")
    if lookup.shape[0] <= max_index:
        lookup = np.concatenate((lookup, np.setdiff1d(np.arange(max_index + 1), lookup)))
    return lookup  # type: ignore


@overload
def create_removal_lookup(
    removed_mask: npt.ArrayLike,
    *,
    length: Optional[int] = None,
    replace_value: Optional[int] = None,
    add_empty: Literal["increment", "no increment", True, False] = False,
    invert: bool = False,
    return_inverse: Literal[False] = False,
) -> Int1DArray: ...
@overload
def create_removal_lookup(
    removed_mask: npt.ArrayLike,
    *,
    length: Optional[int] = None,
    replace_value: Optional[int] = None,
    add_empty: Literal["increment", "no increment", True, False] = False,
    invert: bool = False,
    return_inverse: Literal[True],
) -> tuple[Int1DArray, Int1DArray]: ...
def create_removal_lookup(
    removed_mask: npt.ArrayLike,
    *,
    length: Optional[int] = None,
    replace_value: Optional[int] = None,
    add_empty: Literal["increment", "no increment", True, False] = False,
    invert: bool = False,
    return_inverse: bool = False,
) -> Int1DArray | tuple[Int1DArray, Int1DArray]:
    """Create a lookup table to reorder index after having removed elements from an array.

    Parameters
    ----------
    removed_mask : np.ndarray
        A boolean mask specifying which indices were removed.

    replace_value : Optional[int], optional
        The value to replace the removed elements.

        If None, the removed elements are replaced by the previous non-deleted index.

    length : Optional[int], optional
        The length of the original array before removal. If None, the length is inferred from the removed_mask.

    add_empty : Literal["increment", "no increment", True, False], optional
        If not False, an empty entry is added at the beginning of the lookup table.
        If "increment", all other indices are incremented by 1.
        If "no increment" or True, the other indices are not incremented and the empty entry is set to the replace_value.

    invert : bool, optional
        If True, the mask is inverted before creating the lookup table.

    return_inverse : bool, optional
        If True, also return the inverse lookup table.

    Returns
    -------
    np.ndarray
        A 1D array of the same length as the input array containing the new index of each element.

    Examples:
    --------
    The following mask removes the second and fourth elements from an array of length 5:
    >>> removed_mask = np.array([False, True, False, False, True, False])
    >>> create_removal_lookup(removed_mask)
    array([0, 0, 1, 2, 2, 3])

    Optionally the removed elements can be replaced by a specific value:

    >>> create_removal_lookup(removed_mask, replace_value=-1)
    array([ 0, -1,  1,  2,  -1, 3])
    """  # noqa: E501
    removed_mask = np.asarray(removed_mask)
    if length is not None:
        if removed_mask.dtype != bool:
            removed_mask = index_to_mask(removed_mask, length)
        elif removed_mask.shape[0] < length:
            removed_mask = np.concatenate((removed_mask, np.zeros(length - removed_mask.shape[0], dtype=bool)))
        elif removed_mask.shape[0] > length:
            removed_mask = removed_mask[:length]
    elif removed_mask.dtype != bool:
        raise ValueError("If length is not provided, removed_mask must be a boolean mask.")
    length = removed_mask.shape[0]

    if invert:
        removed_mask = ~removed_mask

    if add_empty:
        lookup = np.cumsum(~removed_mask)
        if add_empty in ["no increment", True]:
            lookup = np.concatenate(([0 if replace_value is None else replace_value], lookup - 1))
        else:  # "increment"
            lookup = np.concatenate(([0], lookup))
        if replace_value is not None:
            lookup[1:][removed_mask] = replace_value

        if return_inverse:
            inverse = np.arange(length)[~removed_mask]
            if add_empty in ["no increment", True]:
                inverse = np.concatenate(([-1], inverse))
            else:
                inverse = np.concatenate(([0], inverse + 1))
            return lookup, inverse  # type: ignore
        return lookup  # type: ignore
    else:
        lookup = np.cumsum(~removed_mask) - 1
        if replace_value is not None:
            lookup[removed_mask] = replace_value
        return (lookup, np.arange(length)[~removed_mask]) if return_inverse else lookup  # type: ignore


def invert_lookup(lookup: Int1DArray, max_index: Optional[int] = None) -> Int1DArray:
    """
    Invert a lookup array. The lookup array must be a 1D array of integers. The output array is a 1D array of length
    max(lookup) + 1. Any -1 index is considered as deleted.
    Pseudo code:
        [ argwhere(lookup==i)[0] if i in lookup else -1 for i in range(max(lookup) + 1) ]
    """
    if max_index is None:
        max_index = int(lookup.max())

    out = np.full(max_index + 2, -1, dtype=lookup.dtype)
    out[lookup] = np.arange(len(lookup), dtype=lookup.dtype)
    return out[:-1]  # type: ignore

    # Slower version:
    # unique_id, inverse = np.unique(lookup, return_index=True)
    # return inverse[unique_id >= 0]


def reorder_array(array: npt.NDArray, indices: npt.NDArray[np.int_], max_index=None) -> npt.NDArray:
    """
    Reorder an array according to an index table.
    This is equivalent to array[invert_lookup(indices].

    """
    if max_index is None:
        max_index = indices.max()
    out = np.empty((max_index + 2,) + array.shape[1:], dtype=array.dtype)
    out[indices] = array
    return out[:-1]


def invert_lookup_legacy(lookup):
    """
    Invert a lookup array. The lookup array must be a 1D array of integers. The output array is a 1D array of length
    max(lookup) + 1. The output array[i] contains the list of indices of lookup where lookup[index] == i.
    """
    splits = np.split(np.argsort(np.argsort(lookup)), np.cumsum(np.unique(lookup, return_counts=True)[1]))[:-1]
    return np.array([np.array(s[0], dtype=np.int64) for s in splits])


def invert_complete_lookup(lookup: Int1DArray) -> Int1DArray:
    """
    Invert a complete lookup array. The lookup array must be a 1D array of unique integers.
    max(lookup) + 1. The output array[i] contains the list of indices of lookup where lookup[index] == i.
    """
    out = np.empty_like(lookup)
    out[lookup] = np.arange(len(lookup), dtype=lookup.dtype)
    return out
