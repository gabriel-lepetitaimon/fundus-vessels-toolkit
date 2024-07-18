from typing import Any

import numpy as np


def ensure_superior_multiple(x, m=32):
    """
    Return y such that y >= x and y is a multiple of m.
    """
    return m - (x - 1) % m + x - 1


def gaussian(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def gaussian_filter1d(x, sigma, integrate=False):
    from scipy.signal import convolve

    t = np.arange(-sigma * 3, sigma * 3 + 1)
    kernel = gaussian(t, sigma)
    kernel = kernel / kernel.sum()
    return convolve(x, kernel, mode="same")


def angle_diff(a, b, degrees=False):
    """
    Return the difference between angles a and b in the range [-pi, pi].
    """
    diff = a - b
    if degrees:
        return (diff + 180) % 360 - 180
    else:
        return (diff + np.pi) % (2 * np.pi) - np.pi


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


def np_find_sorted(keys: Any, array: np.ndarray, assume_keys_sorted=False) -> int | np.ndarray:
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
        The index or indexes of the keys in the array. If a key is not found, -1 is returned.
    """
    if np.isscalar(keys):
        if array[0] == keys:
            return 0
        i = np.searchsorted(array, keys)
        isin = 0 < i < len(array)
        return i if 0 < i < len(array) else -1
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


def quantified_roots(x, threshold=0):
    from scipy.signal import medfilt

    x_quant = np.digitize(x, [-threshold, threshold]) - 1
    x_quant = medfilt(x_quant * 1, 5)

    root_intervals = []

    # Search first non-zero value
    i = 0
    while i < len(x_quant) and x_quant[i] == 0:
        i += 1

    last_x = x_quant[i]
    while i < len(x_quant) - 1:
        next_x = x_quant[i + 1]
        if next_x == 0:
            for j in range(i + 1, len(x_quant)):
                next_x = x_quant[j]
                if next_x != 0:
                    break
            root_intervals.append((i, j))
            i = j
        elif last_x != next_x:
            root_intervals.append((i, i))
        last_x = next_x
        i += 1

    roots = [(start + end) // 2 for start, end in root_intervals]

    return np.array(roots).astype(int)


def quantified_local_minimum(x, dthreshold=None):
    from scipy.signal import medfilt

    dx = np.diff(x)
    if dthreshold is None:
        dthreshold = np.percentile(np.abs(dx), 10)
    dx_quant = np.digitize(dx, [-dthreshold, dthreshold]) - 1
    dx_quant = medfilt(dx_quant * 1, 5)

    local_minimum_interval = []
    start = None
    for i in range(1, len(dx_quant)):
        if dx_quant[i - 1] == -1:
            if dx_quant[i] == 0:
                start = i - 1
            elif dx_quant[i] == 1:
                local_minimum_interval.append((i - 1, i))
                start = None
        elif start and dx_quant[i] == 1:
            local_minimum_interval.append((start, i))
            start = None

    roots = [(start + end + 1) // 2 for start, end in local_minimum_interval]
    return np.array(roots).astype(int)
