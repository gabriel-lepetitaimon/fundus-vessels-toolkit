########################################################################################################################
#   *** VESSELS SKELETONIZATION AND GRAPH CREATION FROM SEGMENTATION ***
#   This modules provides functions to extract the vascular graph from a binary image of retinal vessels.
#   Two implementation are provided: one using numpy array, scipy.ndimage and scikit-image,
#   the other using pytorch and kornia.
#
#   TODO: Implement local maxima rift detection on pytorch to compute the medial axis from the distance transform.
#
#
########################################################################################################################

__all__ = ['skeletonize', 'torch_medial_axis', 'extract_patches', 'extract_graph']

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

from nntemplate.misc.function_tools import LogTimer

# ====================== NUMPY IMPLEMENTATION =============================

# Pre-compute masks for junctions and endpoints detections

_junctions_endpoints_masks_cache = None


def compute_junction_endpoint_masks():
    global _junctions_endpoints_masks_cache
    if _junctions_endpoints_masks_cache is None:
        def create_line_junction_masks(n_lines: int | Tuple[int, ...] = (3, 4)):
            if isinstance(n_lines, int):
                n_lines = (n_lines,)
            masks = []
            if 3 in n_lines:
                masks_3 = [
                    # Y vertical
                    [[1, 0, 1],
                     [0, 1, 0],
                     [2, 1, 2]],
                    [[1, 0, 1],
                     [2, 1, 2],
                     [0, 1, 0]],
                    # Y Diagonal
                    [[0, 1, 0],
                     [2, 1, 1],
                     [1, 2, 0]],
                    [[2, 1, 0],
                     [0, 1, 1],
                     [1, 0, 2]],
                    # T Vertical
                    [[2, 0, 2],
                     [1, 1, 1],
                     [0, 1, 0]],
                    # T Diagonal
                    [[1, 2, 0],
                     [0, 1, 2],
                     [1, 0, 1]]]
                masks_3 = np.asarray(masks_3)
                masks_3 = np.concatenate([np.rot90(masks_3, k=k, axes=(1, 2)) for k in range(4)])
                masks += [masks_3]
            if 4 in n_lines:
                masks_4 = np.asarray([
                    [[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]],
                    [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]
                ], dtype=bool)
                masks += [masks_4]
            masks = np.concatenate(masks)
            return masks == 1, masks == 0

        junction_3lines_masks = create_line_junction_masks(3)
        junction_4lines_masks = create_line_junction_masks(4)

        hollow_cross_mask = np.asarray([[2, 1, 2],
                                        [1, 0, 1],
                                        [2, 1, 2]])
        hollow_cross_mask = hollow_cross_mask == 1, hollow_cross_mask == 0

        def create_endpoint_masks():
            mask = np.asarray([
                [[2, 1, 2],
                 [0, 1, 0],
                 [0, 0, 0]],
                [[0, 2, 1],
                 [0, 1, 2],
                 [0, 0, 0]]
            ])
            mask = np.concatenate([np.rot90(mask, k, axes=(1, 2)) for k in range(4)])
            return mask == 1, mask == 0

        endpoint_masks = create_endpoint_masks()

        _junctions_endpoints_masks_cache = junction_3lines_masks, junction_4lines_masks, hollow_cross_mask, endpoint_masks

    return _junctions_endpoints_masks_cache


def skeletonize(vessel_map: np.ndarray, return_distance=False, fix_hollow=True, remove_small_ends=5,
                branches_label: np.ndarray | None = None):
    """
    Skeletonize a binary image of retinal vessels using the medial axis transform.
    Junctions and endpoints are detected and labeled, and simple cleanup operations are performed on the skeleton.

    Args:
        vessel_map (H, W): Binary image of retinal vessels.
        return_distance: If True, also return the distance transform of the medial axis.
        fix_hollow: If True, fill hollow cross pattern.
        remove_small_ends: Remove small end branches whose size is inferior to this. 1px branches should be removed as
                           they will be ignored in the graph construction.
        branches_label: If provided store the branches label in this array.
    Returns:
        The skeletonized image where each pixel is described by an integer value as:
            - 0: background
            - 1: Vessel endpoint
            - 2: Vessel branch
            - 3: Vessel 3-branch junction
            - 4: Vessel 4-branch junction

        If return_distance is True, also return the distance transform of the medial axis.
    """
    from skimage.morphology import medial_axis, skeletonize
    from skimage.measure import label
    import scipy.ndimage as scimage

    junction_3lines_masks, junction_4lines_masks, hollow_cross_mask, endpoint_masks = compute_junction_endpoint_masks()
    sqr3 = np.ones((3, 3), dtype=bool)

    # === Compute the medial axis ===
    with LogTimer("Compute skeleton"):
        if return_distance:
            bin_skel, skel_dist = medial_axis(vessel_map, return_distance=return_distance, random_state=0)
        else:
            bin_skel = skeletonize(vessel_map)
            skel_dist = None
        skel = bin_skel.astype(np.int8)
        bin_skel_patches = extract_patches(bin_skel, bin_skel, (3, 3), True)

    # === Build the skeleton with junctions and endpoints ===
    with LogTimer("Detect junctions"):
        # Detect junctions
        skel[fast_hit_or_miss(bin_skel, bin_skel_patches, *junction_3lines_masks)] = 2
        skel += fast_hit_or_miss(bin_skel, bin_skel_patches, *junction_4lines_masks) * 2

    # Fix hollow cross junctions
    if fix_hollow:

        with LogTimer("Fix hollow cross") as log:
            # Fix hollow cross junctions interpreted as multiple 3-lines junctions
            junction_hollow_cross = fast_hit_or_miss(bin_skel, bin_skel_patches, *hollow_cross_mask)
            log.print('Hollow cross found')
            skel -= scimage.convolve(junction_hollow_cross.astype(np.int8), sqr3.astype(np.int8))
            skel = skel.clip(0) + junction_hollow_cross

            log.print('Hollow cross quick fix, starting post fix')
            for y, x in zip(*np.where(junction_hollow_cross)):
                neighborhood = skel[y - 1:y + 2, x - 1:x + 2]
                if np.sum(neighborhood) == 1:
                    skel[y, x] = 0
                if any(scimage.binary_hit_or_miss(neighborhood, *m)[1, 1] for m in zip(*junction_4lines_masks)):
                    skel[y, x] = 3
                elif any(scimage.binary_hit_or_miss(neighborhood, *m)[1, 1] for m in zip(*junction_3lines_masks)):
                    skel[y, x] = 2
                else:
                    skel[y, x] = 1

    skel += skel.astype(bool)

    # Detect endpoints
    with LogTimer('Detect endpoints') as log:
        bin_skel = skel > 0
        skel -= fast_hit_or_miss(bin_skel, bin_skel, *endpoint_masks)

    with LogTimer('Remove 1px small end branches'):
        if remove_small_ends > 0:
            # Remove small end branches
            skel = remove_1px_endpoints(skel, endpoint_masks, sqr3)

    # === Create branches labels ===
    branches_label_required = branches_label is not None or remove_small_ends >= 2
    if branches_label is None and branches_label_required:
        branches_label = np.zeros_like(skel, dtype=np.int32)
    if branches_label_required:
        assert branches_label.shape == vessel_map.shape, "branches_label must have the same shape as vessel_map"
        assert branches_label.dtype == np.int32, "branches_label must be of type np.int32"

        with LogTimer('Create branches labels'):
            junctions = skel >= 3
            junction_neighbours = scimage.binary_dilation(junctions, sqr3)
            binskel_no_junctions = (skel >= 1) & ~junction_neighbours
            branches_label[:], nb_branches = label(binskel_no_junctions, return_num=True, connectivity=2)
    else:
        nb_branches = 0
        branches_label = None

    # === Remove small branches ===
    if remove_small_ends >= 2:
        # --- Remove larger than 1 px endpoints branches ---
        with LogTimer('Remove small branches') as log:
            # Select small branches
            branch_sizes = np.bincount(branches_label.ravel())
            small_branches_id = np.where(branch_sizes < remove_small_ends)[0]
            log.print(f'Found {len(small_branches_id)} small branches.')

            # Select branches which including endpoints
            endpoint_branches_id = np.unique(branches_label[skel == 1])
            log.print(f'Found {len(endpoint_branches_id)} branches shich includes endpoints.')

            # Identify small branches labels
            branches_to_remove = np.intersect1d(endpoint_branches_id, small_branches_id, assume_unique=True)
            log.print(f' => {len(branches_to_remove)} branches to be removed.')

            label_lookup = np.zeros(nb_branches + 1, dtype=np.int32)
            label_lookup[branches_to_remove] = 1
            label_lookup = np.arange(nb_branches + 1, dtype=np.int32) - np.cumsum(label_lookup)
            nb_branches = nb_branches - len(branches_to_remove)
            log.print(f'Computed label_lookup')

            # Remove small branches from the skeleton.
            skel[np.isin(branches_label, branches_to_remove)] = 0

            # Remove small branches labels from the label map
            branches_label[:] = label_lookup[branches_label]

            log.print(f'Small branch has been removed')
            # At this point small branches have been reduced to 1px endpoints and must be removed again.
            bin_skel = skel > 0
            skel -= fast_hit_or_miss(bin_skel, bin_skel, *endpoint_masks)
            log.print(f'Endpoints detected again')

            skel = remove_1px_endpoints(skel, endpoint_masks, sqr3)
            log.print(f'1px branch removed')

    if return_distance:
        return skel, skel_dist
    else:
        return skel


def remove_1px_endpoints(skel, endpoint_masks=None, sqr3=None):
    import scipy.ndimage as scimage
    if endpoint_masks is None:
        endpoint_masks = compute_junction_endpoint_masks()[3]
    if sqr3 is None:
        sqr3 = np.ones((3, 3), dtype=np.int8)

    endpoints = skel == 1
    jonctions = skel >= 3
    jonction_neighbours = scimage.binary_dilation(jonctions, sqr3) & (skel == 1)
    branches_1px = endpoints & jonction_neighbours

    # Remove the 1px branches from the skeleton and decrease the junctions degree around them.
    branches_1px = scimage.convolve(branches_1px.astype(np.int8), sqr3.astype(np.int8))
    # If 1 endpoint in neighborhood:
    # background or endpoints -> background; vessels or 3-junction -> vessels; 4-junctions -> 3-junctions
    junction_mapping = np.asarray([0, 0, 2, 2, 3], dtype=skel.dtype)
    b1 = np.where(branches_1px == 1)
    skel[b1] = junction_mapping[skel[b1]]

    # If 2 endpoints in neighborhood:
    # background or endpoints -> background; junctions or vessels -> vessels;
    junction_mapping = np.asarray([0, 0, 2, 2, 2], dtype=skel.dtype)
    b2 = np.where(branches_1px == 2)
    skel[b2] = junction_mapping[skel[b2]]

    # Detect endpoints again.
    # This is required because blindly casting vessels to endpoints in the neighborhood of 1px branches may result in
    # invalid endpoints in the middle of a branch.
    # for m in zip(*endpoint_masks):
    #     skel[scimage.binary_hit_or_miss(skel, *m)] = 1
    bin_skel = skel > 0
    skel[fast_hit_or_miss(bin_skel, bin_skel, *endpoint_masks)] = 1

    return skel

def fast_hit_or_miss(map: np.ndarray, mask: np.ndarray | Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]],
                     positive_patterns: np.ndarray, negative_patterns: np.ndarray | None = None,
                     aggregate_patterns='any') -> np.ndarray:
    """
    Apply a hit or miss filter to a map with a mask.

    Args:
        map: The map to apply the filter to. Shape: (H, W) where H and W are the map height and width.
        mask : The mask to apply to the map. Shape: (H, W) where H and W are the map height and width.
        positive_patterns : The positive pattern to apply. Shape: (p, h, w) where p is the number of patterns, h and w are the pattern height and width.
        negative_patterns : The negative pattern to apply. Shape: (p, h, w) where p is the number of patterns, h and w are the pattern height and width.
        aggregate_patterns : The aggregation method to use. Can be 'any', 'sum' or None.

    Returns:
        The result of the hit-or-miss. Shape: (H, W)
    """
    assert positive_patterns.dtype == bool, "positive_patterns must be of type bool"
    if positive_patterns.ndim == 2:
        positive_patterns = positive_patterns[np.newaxis, ...]
    assert positive_patterns.ndim == 3, "positive_patterns must be of shape (p, h, w): " \
                                        "where p is the number of patterns, h and w are the pattern height and width."
    if negative_patterns is None:
        negative_patterns = ~positive_patterns
    else:
        if negative_patterns.ndim == 2:
            negative_patterns = negative_patterns[np.newaxis, ...]
        assert negative_patterns.dtype == bool, "negative_patterns must be of type bool"
        assert positive_patterns.shape == negative_patterns.shape, "positive_patterns and negative_patterns must have the same shape"

    # Extract patches from the map
    pattern_shape = positive_patterns.shape[1:]
    if isinstance(mask, tuple):
        patches, centers_idx = mask
    else:
        patches, centers_idx = extract_patches(map, mask, pattern_shape, return_coordinates=True)

    # Apply hit-or-miss for all patterns
    positive_patterns = positive_patterns.reshape((positive_patterns.shape[0], -1))
    negative_patterns = negative_patterns.reshape((negative_patterns.shape[0], -1))
    patches = patches.reshape((patches.shape[0], -1))

    # print(positive_patterns)
    hit_patches = binary1d_hit_or_miss(patches, positive_patterns, negative_patterns)

    # Aggregate the response of hit-or-miss for each patterns
    if aggregate_patterns == 'any':
        hit_patches = hit_patches.any(axis=1)
    elif aggregate_patterns == 'sum':
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


def extract_patches(map: np.ndarray, mask: np.ndarray, patch_shape: Tuple[int, int], return_coordinates=False) \
        -> np.ndarray | Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    assert map.shape == mask.shape, f"map and mask must have the same shape " \
                                    f"(map.shape={map.shape}, mask.shape={mask.shape})"
    assert mask.dtype == bool, f"mask must be of type bool, not {mask.dtype}"

    # Pad map
    map = np.pad(map, tuple((p//2, p-p//2) for p in patch_shape), mode='constant', constant_values=0)

    # Compute the coordinates of the pixels inside the patches
    y, x = np.where(mask)
    py, px = patch_shape
    masked_y = (np.repeat(np.arange(py), px)[np.newaxis, :] + y[:, np.newaxis]).flatten()
    masked_x = (np.tile(np.arange(px), py)[np.newaxis, :] + x[:, np.newaxis]).flatten()

    if return_coordinates:
        return map[masked_y, masked_x].reshape((-1, patch_shape[0], patch_shape[1])), (y, x)
    else:
        return map[masked_y, masked_x].reshape((-1, patch_shape[0], patch_shape[1]))


def binary1d_hit_or_miss(samples, positive_patterns, negative_patterns=None):
    """
    Apply a hit or miss filter to a 1D binary array.

    Args:
        samples (S, N): The samples to apply the filter to.
        positive_pattern (P, N): The positive pattern to apply.
        negative_pattern (P, N): The negative pattern to apply.
                                 If None, the negative pattern is the complement of the positive pattern.

    Returns:
        The result of the hit-or-miss for each sample and each pattern. The output shape is (S, P).
    """
    assert samples.dtype == bool, "samples must be of type bool"
    assert positive_patterns.dtype == bool, "positive_patterns must be of type bool"
    assert positive_patterns.ndim == 2 and samples.ndim == 2, "samples and positive_patterns must be 2D"
    assert positive_patterns.shape[1] == samples.shape[1], "samples and positive_patterns must have the same number of columns"
    if negative_patterns is None:
        negative_patterns = ~positive_patterns
    else:
        assert negative_patterns.dtype == bool, "negative_patterns must be of type bool"
        assert negative_patterns.shape == positive_patterns.shape, "negative_patterns must have the same shape as positive_patterns"

    # Reshape patterns to (1, N, P)
    positive_patterns = np.expand_dims(positive_patterns.transpose(), axis=0)
    negative_patterns = np.expand_dims(negative_patterns.transpose(), axis=0)
    samples = np.expand_dims(samples, axis=2)

    # Apply patterns
    return ~np.any((positive_patterns & ~samples) | (negative_patterns & samples), axis=1)


def extract_unravelled_pattern(map: np.ndarray, where: np.ndarray | Tuple[np.ndarray, np.ndarray], pattern: np.ndarray, return_coordinates=False):
    """
    Extract pattern from `map` centered on the pixels in `where`. The patterns are extracted as a 1D arrays.

    Args:
        map: The map to extract patterns from. Shape: (H, W) where H and W are the map height and width.
        where: Either:
            - The coordinates of the center of the patterns to extract. Shape: (2, N) where N is the number of patterns.
            - A boolean mask of the same shape as `map` indicating the pixels to extract patterns from.
        pattern: The pattern to extract. Shape: (h, w) where h and w are the pattern height and width.
        return_coordinates: If True, return the coordinates of the center of the extracted patterns.
                            (The result of np.where(where))

    Returns:
        The extracted patterns.
        If return_index is True, also return the index of the patterns center as a tuple containing two 1d vector
        for the y and x coordinates.
    """
    where = np.asarray(where)

    assert map.ndim == 2, f"map must be 2D, not {map.ndim}D"
    assert where.ndim == 2, f"where arreay must be 2D, not {where.ndim}D"
    if where.shape == map.shape:
        assert where.dtype == bool, f"where must be of type bool, not {where.dtype}"
    else:
        assert where.shape[0] == 2, f"Invalid value for where shape: {where.shape}.\n" \
                                    f"where must either be a boolean mask of the same shape as map " \
                                    f"or a tuple of 1D arrays providing the coordinates of the centers where " \
                                    f"patterns will be extracted."
        assert where.dtype == np.int64, f"where must be of type np.int64, not {where.dtype}"
    if where.shape == map.shape:
        y, x = np.where(where)
    else:
        y, x = where

    assert pattern.ndim == 2, f"pattern must be 2D, not {pattern.ndim}D"
    pattern_y, pattern_x = np.where(pattern)

    # Pad map
    map = np.pad(map, tuple((p // 2, p - p // 2) for p in pattern.shape), mode='constant', constant_values=0)

    # Compute the coordinates of the pixels to extract
    y_idxs = (pattern_y[np.newaxis, :] + y[:, np.newaxis]).flatten()
    x_idxs = (pattern_x[np.newaxis, :] + x[:, np.newaxis]).flatten()

    map = map[y_idxs, x_idxs].reshape((-1, pattern.sum()))
    if return_coordinates:
        return map, (y, x)
    else:
        return map


def extract_graph(vessel_map: np.ndarray, return_label=False, junctions_merge_distance=3):
    import networkx as nx
    import scipy.ndimage as scimage
    import skimage.morphology as skmorph
    from skimage.measure import label
    from skimage.segmentation import expand_labels

    if vessel_map.dtype != np.int8:
        skel = skeletonize(vessel_map, fix_hollow=True, remove_small_ends=10, return_distance=False)
    else:
        skel = vessel_map

    bin_skel = skel > 0
    junctions = skel >= 3
    sqr3 = np.ones((3, 3), dtype=bool)

    # Label branches
    skel_no_junctions = bin_skel & ~scimage.binary_dilation(junctions, sqr3)
    labeled_branches, nb_branches = label(skel_no_junctions, return_num=True)
    labeled_branches = expand_labels(labeled_branches, 2) * bin_skel

    # Label junctions
    jy, jx = np.where(junctions)
    labels_junctions = np.zeros_like(labeled_branches)
    labels_junctions[jy, jx] = np.arange(1, len(jy)+1)
    nb_junctions = len(jy)

    # Identify branches connected to each junction
    # junctions_ring = [(jy - 1, jx + 1), (jy, jx + 1),
    #                  (jy + 1, jx + 1), (jy + 1, jx),
    #                  (jy + 1, jx - 1), (jy, jx - 1),
    #                  (jy - 1, jx - 1), (jy - 1, jx)]
    # junctions_ring = tuple(np.clip(yx, 0, hw-1)
    #                       for yx, hw in zip(np.asarray(junctions_ring).transpose((1, 0, 2)), labeled_branches.shape))
    # junctions_ring = labeled_branches[junctions_ring]

    ring_pattern = np.asarray([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=bool)
    junctions_ring = extract_unravelled_pattern(labeled_branches, (jy, jx), ring_pattern, return_coordinates=False)

    # Build the matrix of connections from branches to junctions
    connections = np.zeros((nb_branches+1, nb_junctions), dtype=bool)
    connections[junctions_ring.flatten(), np.tile(np.arange(nb_junctions), ring_pattern.sum())] = 1
    connections = connections[1:, :]

    # Detect junctions clusters
    cluster_structure = skmorph.disk(junctions_merge_distance) > 0
    junctions_clusters_by_junctions = extract_unravelled_pattern(labels_junctions, (jy, jx), cluster_structure)
    junctions_clusters = np.zeros((nb_junctions+1, nb_junctions), dtype=bool)
    junctions_clusters[junctions_clusters_by_junctions.flatten(),
                       np.tile(np.arange(nb_junctions), np.sum(cluster_structure))] = 1
    junctions_clusters = junctions_clusters[1:, :]

    if return_label:
        return junctions_clusters, labeled_branches, (jy, jx)
    else:
        return connections

# ====================== TORCH IMPLEMENTATION =============================

def torch_medial_axis(vessel_maps: torch.Tensor, keep_distance=True) -> torch.Tensor:
    """Compute the medial axis of a binary image.
    Args:
        vessel_maps (torch.Tensor): the input image with shape :math:`(B?, 1?, H, W)`.
    Returns:
        torch.Tensor: the skeletonized image with shape :math:`(B, 1, H, W)`.
    """
    from kornia.contrib import distance_transform

    shape = vessel_maps.shape
    vessel_maps = _check_vessel_map(vessel_maps)

    # Compute Distance Transform
    dt = distance_transform(vessel_maps.unsqueeze(1).float())

    @torch.jit.script
    def rift_maxima(patch):
        v = patch[4]
        zero = torch.zeros_like(v)
        if torch.all(v == zero):
            return v
        else:
            q = torch.quantile(patch, 7 / 9, interpolation='lower')
            return v if torch.all(zero < q <= v) else zero

    rift_maxima = torch.func.vmap(rift_maxima)

    dt = F.unfold(dt, kernel_size=3, padding=1)
    dt = dt.permute(0, 2, 1).reshape(-1, 9)
    dt = rift_maxima(dt)
    dt = dt.reshape(shape[0], -1, shape[-2], shape[-1])
    return dt if keep_distance else dt > 0

    # Mask pixel with 0 distance
    not_null_image, not_null_row, not_null_col = torch.where(vessel_maps != 0)
    not_null_index = not_null_image * np.prod(vessel_maps.shape[1:]) + not_null_row * vessel_maps.shape[
        2] + not_null_col
    not_null_index = not_null_index.reshape(-1)

    # Cancel out any pixels that do not belong the local maximas rifts
    unfold_dt = F.unfold(dt, kernel_size=3, padding=1)
    unfold_dt = unfold_dt.permute(0, 2, 1).reshape(-1, 9)[not_null_index]
    quantile = torch.quantile(unfold_dt, 7 / 9, dim=1, interpolation='lower')

    print(unfold_dt.shape, quantile.shape)
    not_local_maxima = torch.where((quantile > 0) & (unfold_dt[:, 4] > quantile))[0]
    dt = dt.reshape(-1)[not_null_index]
    dt[not_local_maxima] = 0

    final_shape = (shape[0],) + tuple(shape[-2:])
    skeleton = torch.zeros(np.prod(final_shape), dtype=dt.dtype, device=vessel_maps.device)
    skeleton[not_null_index] = dt
    skeleton = skeleton.reshape(final_shape)
    if not keep_distance:
        return skeleton > 0
    else:
        return skeleton


def _check_vessel_map(vessel_maps):
    if vessel_maps.ndim == 4:
        assert vessel_maps.shape[
                   1] == 1, f"Expected 2D vessels map of shapes (B, 1, H, W), but provided maps has multiple channels."
        vessel_maps = vessel_maps.squeeze(1)
    elif vessel_maps.ndim == 2:
        vessel_maps = vessel_maps.unsqueeze(0)
    else:
        assert vessel_maps.ndim == 3, f"Expected 2D vessels maps of shapes (B?, 1?, H, W), got {vessel_maps.ndim}D tensor."

    if vessel_maps.dtype != torch.bool:
        vessel_maps = vessel_maps > 0.5
    return vessel_maps
