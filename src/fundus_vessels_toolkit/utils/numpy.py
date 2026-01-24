from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

from ..utils.typing import Bool2DArray


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


def np_group_by(array: npt.NDArray, keys: npt.NDArray) -> list[tuple[npt.NDArray, npt.NDArray]]:
    """
    Group the elements of an array by keys.

    Parameters
    ----------
    array : np.ndarray
        The array to group.
    keys : np.ndarray
        The keys to group by. Must be the same length as array.

    Returns
    -------
    List[np.ndarray]
        A list of arrays, each containing the elements of array corresponding to a unique key.
    """
    assert array.shape[0] == keys.shape[0], "array and keys must have the same length."
    unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
    return [(unique_keys[i], array[inverse_indices == i]) for i in range(len(unique_keys))]


def bit_invert(bits: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Invert the bits of a numpy array of uint64.

    Parameters
    ----------
    bits : np.ndarray[np.uint64]
        The bits to invert.

    Returns
    -------
    np.ndarray[np.uint64]
        The inverted bits.
    """
    return bits ^ np.uint64(0xFFFFFFFFFFFFFFFF)


def binary_sparse_conv2d[T: np.generic](
    binary_array: Bool2DArray, kernel: npt.NDArray[T], mode: Literal["same", "safe", "full"] = "same"
) -> npt.NDArray[T]:
    """Convolve a sparse binary mask with a convolution kernel."""
    assert binary_array.ndim == 2 and binary_array.dtype == np.bool_, "binary_array must be a 2D boolean array"
    kH, kW = kernel.shape
    H, W = binary_array.shape
    out = np.zeros((H + kH, W + kW), dtype=kernel.dtype)

    ys, xs = np.nonzero(binary_array)
    for ky in range(kH):
        for kx in range(kW):
            out[ys + ky, xs + kx] += kernel[ky, kx]
    if mode == "full":
        return out
    elif mode == "same":
        return out[kH // 2 : H + kH // 2, kW // 2 : W + kW // 2]
    elif mode == "safe":
        return out[kH - 1 : H, kW - 1 : W]


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


class Sparse2DAccessor[K: np.uint, T: np.generic]:
    def __init__(self, idxs: npt.NDArray[K], values: npt.NDArray[T]) -> None:
        assert idxs.ndim == 2, "idxs must be a 2D array of indices."
        self.idxs = idxs
        self.data = values
        self._key_gen = Sparse2DAccessor.KeyGen(self.idxs)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.idxs.shape

    def __getitem__(self, idx) -> npt.NDArray[T]:
        keys = idx if isinstance(idx, Sparse2DAccKey) else Sparse2DAccKey(self.idxs[idx])  # type: ignore
        if keys.has_null:
            out = np.zeros(keys.idxs.shape, dtype=self.data.dtype)

            out[keys.not_null_idxs] = self.data[keys.idxs[keys.not_null_idxs]]
            return out
        return self.data[keys.idxs]

    def to_dense(self) -> npt.NDArray[T]:
        out = np.zeros(self.idxs.shape, dtype=self.data.dtype)
        out[self.idxs != np.iinfo(self.idxs.dtype).max] = self.data
        return out

    @classmethod
    def from_array[k: np.uint, t: np.generic](
        cls, array: npt.NDArray[t], idxs: Optional[npt.NDArray[k]] = None, mask: Optional[npt.NDArray[np.bool_]] = None
    ) -> Sparse2DAccessor[k, t]:
        if idxs is None:
            INVALID = np.iinfo(np.uint32).max
            idxs_ = np.full(array.shape, INVALID, dtype=np.uint32)
            mask_ = array != array.dtype.type(0)
            idxs_[mask_] = np.arange(mask_.sum(), dtype=np.uint32)
        else:
            assert mask is not None, "If idxs is None, mask must be provided."
            assert mask.shape == array.shape, "idxs must have the same shape as array."
            assert idxs.shape == array.shape, "idxs must have the same shape as array."
            idxs_ = idxs
            mask_ = mask
        return cls(idxs_, array[mask_])  # type: ignore[return-value]

    class KeyGen[k: np.uint]:
        def __init__(self, idxs: npt.NDArray[k]) -> None:
            self.idxs = idxs

        def __getitem__(self, idx) -> Sparse2DAccKey[k]:
            return Sparse2DAccKey[k](self.idxs[idx])

    @property
    def keys(self) -> KeyGen:
        return self._key_gen


class Sparse2DAccKey[K: np.uint]:
    def __init__(self, idxs: npt.NDArray[K], has_null: bool = True) -> None:
        self.idxs = idxs
        if has_null is True:
            self.not_null_idxs = idxs != np.iinfo(idxs.dtype).max
            self.has_null = not np.all(self.not_null_idxs)
        else:
            self.not_null_idxs = np.ones(idxs.shape, dtype=bool)
            self.has_null = False

    def __getitem__(self, idx) -> Sparse2DAccKey[K]:
        return Sparse2DAccKey[K](self.idxs[idx], has_null=self.has_null)
