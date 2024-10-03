import itertools
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt


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


def quantified_roots(
    x: npt.NDArray[np.float_], threshold: float = 0, percentil_threshold: bool = False, medfilt_size: int = 5
):
    """Find the roots of a noisy signal.

    Below ``-threshold`` the signal values are considered negative, above ``threshold`` they are considered positive, in between they are considered zero.
    The returned roots are the indexes where the signal changes from negative to positive or vice versa, or the indexes at the middle of the intervals where the signal is zero.

    Parameters
    ----------
    x : npt.NDArray[np.float_]
        The signal to analyze.

    threshold : int, optional
        The threshold to consider the signal as positive or negative, by default 0
        If ``percentil_threshold`` is True, the threshold is ``np.percentile(np.abs(x), threshold * 100)``.

    percentil_threshold : bool, optional
        If True, the threshold is a percentile of the absolute values of the signal, by default False.

    medfilt_size : int, optional
        The size of the median filter to apply to the quantified signal, by default 5.

    Returns
    -------
    np.ndarray
        The indexes of the roots.
    """  # noqa: E501
    from scipy.signal import medfilt

    if len(x) <= 1:
        return np.array([])

    if percentil_threshold:
        threshold = np.percentile(np.abs(x), threshold * 100)

    x_quant = np.digitize(x, [-threshold, threshold]) - 1
    if len(x) > medfilt_size:
        x_quant = medfilt(x_quant * 1, medfilt_size)

    x_change = np.argwhere(np.diff(x_quant) != 0).flatten() + 1
    roots = []
    for i0, i1 in itertools.pairwise([0] + list(x_change) + [len(x_quant) - 1]):
        if i0 == i1 and i0 != len(x_quant) - 1:
            continue
        if x_quant[i0] == 0:
            roots.append((i1 + i0) // 2)
        elif i1 < len(x_quant) and x_quant[i1] != 0:
            roots.append(i1 - 1)

    # LEGACY
    # root_intervals = []
    # # Search first non-zero value
    # i = 0
    # while i < len(x_quant) - 1 and x_quant[i] == 0:
    #     i += 1
    # if i > 0:
    #     root_intervals.append((0, i - 1))

    # last_x = x_quant[i]
    # while i < len(x_quant) - 1:
    #     next_x = x_quant[i + 1]
    #     if next_x == 0:
    #         for j in range(i + 1, len(x_quant)):
    #             next_x = x_quant[j]
    #             if next_x != 0:
    #                 break
    #         root_intervals.append((i, j))
    #         i = j
    #     elif last_x != next_x:
    #         root_intervals.append((i, i))
    #     last_x = next_x
    #     i += 1

    # roots = [(start + end + 1) // 2 for start, end in root_intervals]

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


def extract_splits(x, medfilt_size=5) -> Dict[Tuple[int, int], float | int]:
    from scipy.signal import medfilt

    if len(x) > medfilt_size:
        x = medfilt(x, medfilt_size)
    change = np.argwhere(np.diff(x) != 0).flatten() + 1

    if len(change) == 0:
        return {(0, len(x)): x[0]}

    return {
        (0, change[0]): x[0],
        **{(change[i], change[i + 1]): x[change[i]] for i in range(len(change) - 1)},
        (change[-1], len(x)): x[change[-1]],
    }


def as_1d_array(data: Any, *, dtype=None) -> Tuple[np.ndarray | None, bool]:
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

    # TODO: Remove the None check and assume that data is valid, bad practice...
    if data is None:
        return None, False

    data = np.asarray(data, dtype=dtype)
    if data.ndim == 0:
        return data[None], True
    if data.ndim == 1:
        return data, False

    raise ValueError(f"Impossible to convert {data} to a 1D vector.")


def modulo_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def intercept_segment(
    a0: npt.ArrayLike,
    a1: npt.ArrayLike,
    b0: npt.ArrayLike,
    b1: npt.ArrayLike,
    a0_bound=True,
    a1_bound=True,
    b0_bound=True,
    b1_bound=True,
) -> np.ndarray:
    """
    Return the intersection point of two segments a0-a1 and b0-b1.

    Parameters
    ----------
    a0 : npt.ArrayLike
        The first points of the first segments as an array of shape (Na, 2).

    a1 : npt.ArrayLike
        The second points of the first segments as an array of shape (Na, 2).

    b0 : npt.ArrayLike
        The first points of the second segments as an array of shape (Nb, 2).

    b1 : npt.ArrayLike
        The second points of the second segments as an array of shape (Nb, 2).

    a0_bound : bool, optional
        Whether the intersection point can't be beyond a0 on the first segment, by default True.

    a1_bound : bool, optional
        Whether the intersection point can't be beyond a1 on the first segment, by default True.

    b0_bound : bool, optional
        Whether the intersection point can't be beyond b0 on the second segment, by default True.

    b1_bound : bool, optional
        Whether the intersection point can't be beyond b1 on the second segment, by default True.

    Returns
    -------
    np.ndarray
        The intersection points as an array of shape (Na, Nb, 2).
        If the segments do not intersect, the points are set to NaN.
    """
    ps = [np.atleast_2d(_).astype(float) for _ in [a0, a1, b0, b1]]
    assert all(p.ndim == 2 and p.shape[1] == 2 for p in ps), "All points must 2D arrays of shape (N, 2)."
    a0, a1, b0, b1 = ps
    assert a0.shape[0] == a1.shape[0], "a0 and a1 must have the same number of points."
    assert b0.shape[0] == b1.shape[0], "b0 and b1 must have the same number of points."

    ya0, xa0 = a0.T[:, :, None]
    ya1, xa1 = a1.T[:, :, None]
    yb0, xb0 = b0.T[:, None, :]
    yb1, xb1 = b1.T[:, None, :]

    d = (xa1 - xa0) * (yb1 - yb0) - (ya1 - ya0) * (xb1 - xb0)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((ya0 - yb0) * (xb1 - xb0) - (xa0 - xb0) * (yb1 - yb0)) / d
        u = ((ya0 - yb0) * (xa1 - xa0) - (xa0 - xb0) * (ya1 - ya0)) / d
        out = np.stack([ya0 + t * (ya1 - ya0), xa0 + t * (xa1 - xa0)], axis=-1)
        if a0_bound:
            out[t < 0] = np.nan
        if a1_bound:
            out[t > 1] = np.nan
        if b0_bound:
            out[u < 0] = np.nan
        if b1_bound:
            out[u > 1] = np.nan
    return out
