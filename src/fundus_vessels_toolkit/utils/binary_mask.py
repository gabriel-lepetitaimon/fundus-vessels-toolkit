from typing import Tuple

import numpy as np


def binary1d_hit_or_miss(samples, positive_patterns, negative_patterns=None):
    """
    Apply several hit or miss filters to a sequence of 1D binary vectors of length N.

    Args:
        samples (S, N): The samples on which to apply the hit-or-miss.
        positive_patterns (P, N): The positive patterns to apply.
        negative_patterns (P, N): The negative patterns to apply.
                                  If None, the negative patterns are the complement of the positive patterns.

    Returns:
        The result of the hit-or-miss for each sample and each pattern. The output shape is (S, P).
    """
    assert samples.dtype == bool, "samples must be of type bool"
    assert positive_patterns.dtype == bool, "positive_patterns must be of type bool"
    assert positive_patterns.ndim == 2 and samples.ndim == 2, "samples and positive_patterns must be 2D"
    assert positive_patterns.shape[1] == samples.shape[1], (
        "samples and positive_patterns must have the same number of columns "
        f"positive_patterns.shape = {positive_patterns.shape}, samples.shape = {samples.shape}"
    )
    if negative_patterns is None:
        negative_patterns = ~positive_patterns
    else:
        assert negative_patterns.dtype == bool, "negative_patterns must be of type bool"
        assert (
            negative_patterns.shape == positive_patterns.shape
        ), "negative_patterns must have the same shape as positive_patterns"

    # Reshape patterns to (1, N, P)
    positive_patterns = np.expand_dims(positive_patterns.transpose(), axis=0)
    negative_patterns = np.expand_dims(negative_patterns.transpose(), axis=0)

    # Reshape samples to (S, N, 1)
    samples = np.expand_dims(samples, axis=2)

    # Apply patterns
    return ~np.any((positive_patterns & ~samples) | (negative_patterns & samples), axis=1)


def fast_hit_or_miss(
    map: np.ndarray,
    mask: np.ndarray | Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    positive_patterns: np.ndarray,
    negative_patterns: np.ndarray | None = None,
    aggregate_patterns="any",
) -> np.ndarray:
    """
    Apply a hit or miss filter to a map but only for the pixel in mask.

    If a majority of the pixels in the mask are False, this method will likely be much faster than
    scipy.image.binary_hit_or_miss. However, it will require more memory as a temporary matrix containing the patches
    surrounding every positive pixel in the mask will be created.

    Args:
        map: The map to apply the filter to. Shape: (H, W) where H and W are the map height and width.
        mask : The mask to apply to the map. Shape: (H, W) where H and W are the map height and width.
               When applying this method several times with different patterns but on the same mask, it is recommended
                to precompute the patches and pass it to this parameter instead of the map. In this case, mask must be
                a tuple (patches, (y, x)) where patches is the matrix of extracted patches with shape (p, h, w) and
                (y, x) are the coordinates of the patches center in the original map and are of shape (p,), where p is
                the number of patches, and h and w are the pattern height and width.
               One can use extract_patches(map, mask, pattern_shape, return_coordinates=True) to precompute the patches.
        positive_patterns : The positive pattern to apply. Shape: (p, h, w) where p is the number of patterns, h and w are the pattern height and width.
        negative_patterns : The negative pattern to apply. Shape: (p, h, w) where p is the number of patterns, h and w are the pattern height and width.
        aggregate_patterns : The aggregation method to use. Can be 'any', 'sum' or None.

    Returns:
        The result of the hit-or-miss. Shape: (H, W)
    """
    assert positive_patterns.dtype == bool, "positive_patterns must be of type bool"
    if positive_patterns.ndim == 2:
        positive_patterns = positive_patterns[np.newaxis, ...]
    assert positive_patterns.ndim == 3, (
        "positive_patterns must be of shape (p, h, w): "
        "where p is the number of patterns, h and w are the pattern height and width."
    )
    if negative_patterns is None:
        negative_patterns = ~positive_patterns
    else:
        if negative_patterns.ndim == 2:
            negative_patterns = negative_patterns[np.newaxis, ...]
        assert negative_patterns.dtype == bool, "negative_patterns must be of type bool"
        assert (
            positive_patterns.shape == negative_patterns.shape
        ), "positive_patterns and negative_patterns must have the same shape"

    # Extract patches from the map
    pattern_shape = positive_patterns.shape[1:]
    if isinstance(mask, tuple):
        assert len(mask) == 2, "mask must be a tuple of length 2 containing the patches and their coordinates."
        assert mask[0].ndim == 3 and mask[0].shape[1:] == positive_patterns.shape[1:], (
            "mask[0] must be of shape (p, h, w): "
            "where p is the number of patches, h and w are the pattern height and width."
        )
        assert (
            isinstance(mask[1], tuple) and len(mask[1]) == 2 and all(c.shape == mask[0].shape[:1] for c in mask[1])
        ), (
            "mask[1] must be a tuple containing the patches centers as two ndarray (y, x) "
            "of shape (p,) where p is the number of patches."
        )
        patches, centers_idx = mask
    else:
        patches, centers_idx = extract_patches(map, mask, pattern_shape, return_coordinates=True)

    # Apply hit-or-miss for all patterns
    positive_patterns = positive_patterns.reshape((positive_patterns.shape[0], -1))
    negative_patterns = negative_patterns.reshape((negative_patterns.shape[0], -1))

    patches = patches.reshape((patches.shape[0], -1))

    hit_patches = binary1d_hit_or_miss(patches, positive_patterns, negative_patterns)

    # Aggregate the response of hit-or-miss for each patterns
    if aggregate_patterns == "any":
        hit_patches = hit_patches.any(axis=1)
    elif aggregate_patterns == "sum":
        hit_patches = hit_patches.sum(axis=1)
    elif aggregate_patterns is None:
        centers_idx = np.repeat(centers_idx, hit_patches.shape[1], axis=1)
        pattern_idx = np.tile(np.arange(hit_patches.shape[1]), hit_patches.shape[0])
        map = np.zeros(map.shape + (hit_patches.shape[1],), dtype=bool)
        map[centers_idx[0], centers_idx[1], pattern_idx] = hit_patches.flatten()
        return map

    map = np.zeros_like(map, dtype=bool)
    map[centers_idx[0], centers_idx[1]] = hit_patches
    return map


def index_to_mask(index: np.ndarray, length: int, invert=False):
    """
    Convert a list of indices to a boolean mask.

    Parameters
    ----------
    index :
        The indices to set as True in the mask. (Or False if invert is True.)

    length : int
        The length of the mask.

    invert : bool, optional
        Invert the mask.

    Returns
    -------
    mask : np.ndarray
        The boolean mask.
    """
    # if not isinstance(length, int):
    #     length = len(length)

    mask = np.zeros(length, dtype=bool) if not invert else np.ones(length, dtype=bool)
    mask[index] = not invert
    return mask


def extract_patches(
    map: np.ndarray, mask: np.ndarray, patch_shape: Tuple[int, int], return_coordinates=False
) -> np.ndarray | Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract patches from a map with a mask.

    Args:
        map: The map to extract patches from. Shape: (H, W) where H and W are the map height and width.
        mask: The mask to apply to the map. Shape: (H, W) where H and W are the map height and width.
        patch_shape: The size of the patch to extract.
        return_coordinates: If True, return the coordinates of the center of the extracted patches.
                            (The result of np.where(mask))

    Returns:
        The extracted patches.
        If return_index is True, also return the index of the patches center as a tuple containing two 1d vector
        for the y and x coordinates.
    """
    assert map.shape == mask.shape, (
        f"map and mask must have the same shape " f"(map.shape={map.shape}, mask.shape={mask.shape})"
    )
    assert mask.dtype == bool, f"mask must be of type bool, not {mask.dtype}"

    # Pad map
    map = np.pad(map, tuple((p // 2, p - p // 2) for p in patch_shape), mode="constant", constant_values=0)

    # Compute the coordinates of the pixels inside the patches
    y, x = np.where(mask)
    py, px = patch_shape
    masked_y = (np.repeat(np.arange(py), px)[np.newaxis, :] + y[:, np.newaxis]).flatten()
    masked_x = (np.tile(np.arange(px), py)[np.newaxis, :] + x[:, np.newaxis]).flatten()

    if return_coordinates:
        return map[masked_y, masked_x].reshape((-1, patch_shape[0], patch_shape[1])), (y, x)
    else:
        return map[masked_y, masked_x].reshape((-1, patch_shape[0], patch_shape[1]))


def extract_unravelled_pattern(
    map: np.ndarray, where: np.ndarray | Tuple[np.ndarray, np.ndarray], pattern: np.ndarray, return_coordinates=False
):
    """
    Extract pattern from `map` centered on the pixels in `where`. The patterns are extracted as a 1D arrays.
    Similar to `extract_patches` but for sparse 2D patterns instead of full patches.

    Args:
        map: The map to extract patterns from. Shape: (H, W) where H and W are the map height and width.
        where: Either:
            - The coordinates of the center of the patterns to extract. Shape: (2, N).
            - A boolean mask of the same shape as `map` indicating the pixels to extract patterns from.
        pattern: The pattern to extract. Shape: (h, w) where h and w are the pattern height and width.
        return_coordinates: If True, return the coordinates of the center of the extracted patterns.
                            (The result of np.where(where))

    Returns:
        The extracted patterns. Shape: (N, pattern_size) where pattern_size is the number of true pixel in `pattern`.
        If return_index is True, also return the index of the patterns center as a tuple containing two 1d vector
        for the y and x coordinates.
    """
    where = np.asarray(where)

    assert map.ndim == 2, f"map must be 2D, not {map.ndim}D"
    assert where.ndim == 2, f"where arreay must be 2D, not {where.ndim}D"
    if where.shape == map.shape:
        assert where.dtype == bool, f"where must be of type bool, not {where.dtype}"
    else:
        assert where.shape[0] == 2, (
            f"Invalid value for where shape: {where.shape}.\n"
            f"where must either be a boolean mask of the same shape as map "
            f"or a tuple of 1D arrays providing the coordinates of the centers where "
            f"patterns will be extracted."
        )
        assert where.dtype == np.int64, f"where must be of type np.int64, not {where.dtype}"
    if where.shape == map.shape:
        y, x = np.where(where)
    else:
        y, x = where

    assert pattern.ndim == 2, f"pattern must be 2D, not {pattern.ndim}D"
    pattern_y, pattern_x = np.where(pattern)

    # Pad map
    map = np.pad(map, tuple((p // 2, p - p // 2) for p in pattern.shape), mode="constant", constant_values=0)

    # Compute the coordinates of the pixels to extract
    y_idxs = (pattern_y[np.newaxis, :] + y[:, np.newaxis]).flatten()
    x_idxs = (pattern_x[np.newaxis, :] + x[:, np.newaxis]).flatten()

    # Extract and reshape to (N, pattern_size)
    map = map[y_idxs, x_idxs].reshape((-1, pattern.sum()))

    if return_coordinates:
        return map, (y, x)
    else:
        return map
