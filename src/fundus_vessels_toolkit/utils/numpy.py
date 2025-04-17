from __future__ import annotations

from typing import List, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

RecursiveIntList: TypeAlias = List[int] | List["RecursiveIntList"]
IntArrayLike: TypeAlias = npt.NDArray[np.int_] | int | RecursiveIntList
Int1DArrayLike: TypeAlias = npt.NDArray[np.int_] | int | List[int]
Int2DArrayLike: TypeAlias = npt.NDArray[np.int_] | List[int] | List[List[int]]
Int3DArrayLike: TypeAlias = npt.NDArray[np.int_] | List[List[int]] | List[List[List[int]]]

RecursiveBoolList: TypeAlias = List[bool] | List["RecursiveBoolList"]
BoolArrayLike: TypeAlias = npt.NDArray[np.bool_] | bool | RecursiveBoolList
Bool1DArrayLike: TypeAlias = npt.NDArray[np.bool_] | bool | List[bool]
Bool2DArrayLike: TypeAlias = npt.NDArray[np.bool_] | List[bool] | List[List[bool]]

RecursiveFloatList: TypeAlias = List[float] | List["RecursiveFloatList"]
FloatArrayLike: TypeAlias = npt.NDArray[np.float_] | float | RecursiveFloatList
Float1DArrayLike: TypeAlias = npt.NDArray[np.float_] | float | List[float]
Float2DArrayLike: TypeAlias = npt.NDArray[np.float_] | List[float] | List[List[float]]
Float3DArrayLike: TypeAlias = npt.NDArray[np.float_] | List[List[float]] | List[List[List[float]]]

PointArrayLike: TypeAlias = npt.NDArray[np.int_] | List[int] | List[List[int]] | Tuple[int, int] | List[Tuple[int, int]]

IntPairArrayLike: TypeAlias = (
    npt.NDArray[np.int_] | List[int] | List[List[int]] | Tuple[int, int] | List[Tuple[int, int]]
)
BoolPairArrayLike: TypeAlias = (
    npt.NDArray[np.bool_] | List[bool] | List[List[bool]] | Tuple[bool, bool] | List[Tuple[bool, bool]]
)


def readonly(arr: npt.NDArray) -> npt.NDArray:
    arr.setflags(write=False)
    return arr


def np_isin_sorted(a, b, *, invert=False):
    """
    Return a boolean array indicating whether each element of a is contained in b.
    Both a and b must be sorted.
    """
    if isinstance(a, np.ndarray):
        searchsorted_id = np.searchsorted(b, a)
        isin = (searchsorted_id < len(b)) & ((b[0] == a) | (searchsorted_id > 0))
        return isin if not invert else ~isin
    else:
        if b[0] == a:
            return True if not invert else False
        isin = 0 < np.searchsorted(b, a) < len(b)
        return isin if not invert else not isin


def np_find_sorted(keys: npt.NDArray, array: npt.NDArray, assume_keys_sorted=False) -> npt.NDArray[np.int_]:
    """
    Find the index of keys in an array.

    Parameters
    ----------
    keys:
        The key or keys (as a numpy array) to find in the array.

    array:
        The array in which to find the keys. Must be sorted and unique.

    assume_keys_sorted:
        If True, assume that keys are sorted in ascending order.

    Returns
    -------
    int | np.ndarray:
        The index or indices of the keys in the array. If a key is not found, -1 is returned.
    """
    if np.isscalar(keys):
        if array[0] == keys:
            return np.zeros(1, dtype=int)
        i = np.searchsorted(array, keys)
        isin = 0 < i < len(array)
        return np.array(i, dtype=int) if 0 < i < len(array) else np.array(-1, dtype=int)
    else:
        keys = np.asarray(keys)
        if len(keys) == 0:
            return np.array([], dtype=int)
        elif len(keys) == 1:
            return np.asarray([np_find_sorted(keys[0], array)], dtype=int)

        if not assume_keys_sorted:
            searchsorted_id = np.searchsorted(array, keys)
            isin = (searchsorted_id < len(array)) & ((array[0] == keys) | (searchsorted_id > 0))
            searchsorted_id[~isin] = -1
            return searchsorted_id
        else:
            k0 = np.argmax(keys < array[0])
            if k0 != 0 or keys[0] < array[0]:
                k0 += 1
            k1 = np.argmax(keys > array[-1])
            if k1 == 0 and keys[0] <= array[-1]:
                k1 = len(array)
            if k1 == k0:
                return -np.ones(len(keys), dtype=int)

            id = np.searchsorted(array, keys[k0:k1])
            return np.concatenate([(-1,) * k0, id, (-1,) * (len(keys) - k1)])


def as_1d_array(data: npt.ArrayLike, *, dtype=None) -> Tuple[npt.NDArray, bool]:
    """Convert the data to a numpy array.

    Parameters
    ----------
    data : Any
        The data to convert.

    Returns
    -------
    np.ndarray | None
        The data as a numpy array.

    bool
        Whether the data is a scalar.
    """

    data = np.asarray(data, dtype=dtype)
    if data.ndim == 0:
        return data[None], True
    if data.ndim == 1:
        return data, False

    raise ValueError(f"Impossible to convert {data} to a 1D vector.")
