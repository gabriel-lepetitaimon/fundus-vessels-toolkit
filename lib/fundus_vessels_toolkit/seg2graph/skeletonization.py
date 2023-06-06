import numpy as np
from skimage.morphology import medial_axis, remove_small_objects
from skimage.morphology import skeletonize as skimage_skeletonize
import scipy.ndimage as scimage
from typing import Literal, TypeAlias

from .skeleton_utilities import extract_patches, fast_hit_or_miss, compute_junction_endpoint_masks, remove_1px_endpoints
from .graph_utilities_cy import label_skeleton


SkeletonizeMethod: TypeAlias = Literal['medial_axis', 'zhang', 'lee']


def skeletonize(vessel_map: np.ndarray, return_distance=False, fix_hollow=True, max_spurs_length=0,
                branches_label: np.ndarray | None = None, skeletonize_method: SkeletonizeMethod = 'lee') -> np.ndarray:
    """
    Skeletonize a binary image of retinal vessels using the medial axis transform.
    Junctions and endpoints are detected and labeled, and simple cleanup operations are performed on the skeleton.

    Args:
        vessel_map (H, W): Binary image of retinal vessels.
        return_distance: If True, also return the distance transform of the medial axis.
        fix_hollow: If True, fill hollow cross pattern.
        max_spurs_length: Remove small end branches whose size is inferior to this. 1px branches should be removed as
                           they will be ignored in the graph construction.
        branches_label: If provided store the branches label in this array.
        skeletonize_method: Method to use for skeletonization. (One of: 'medial_axis', 'zhang', 'lee'. Default: 'lee'.)
    Returns:
        The skeletonized image where each pixel is described by an integer value as:
            - 0: background
            - 1: Vessel endpoint
            - 2: Vessel branch
            - 3: Vessel 3-branch junction
            - 4: Vessel 4-branch junction

        If return_distance is True, also return the distance transform of the medial axis.
    """

    junction_3lines_masks, junction_4lines_masks, hollow_cross_mask, endpoint_masks = compute_junction_endpoint_masks()
    sqr3 = np.ones((3, 3), dtype=bool)

    # === Compute the medial axis ===
    # with LogTimer("Compute skeleton"):
    if skeletonize_method == 'medial_axis':
        bin_skel, skel_dist = medial_axis(vessel_map, return_distance=return_distance, random_state=0)
    else:
        bin_skel = skimage_skeletonize(vessel_map, method=skeletonize_method) > 0
        skel_dist = scimage.distance_transform_edt(bin_skel) if return_distance else None
    remove_small_objects(bin_skel, min_size=3, connectivity=2, out=bin_skel)
    skel = bin_skel.astype(np.int8)
    bin_skel_patches = extract_patches(bin_skel, bin_skel, (3, 3), True)

    # === Build the skeleton with junctions and endpoints ===
    #with LogTimer("Detect junctions"):
    # Detect junctions
    skel[fast_hit_or_miss(bin_skel, bin_skel_patches, *junction_3lines_masks)] = 2
    skel[fast_hit_or_miss(bin_skel, bin_skel_patches, *junction_4lines_masks)] = 3

    # Fix hollow cross junctions
    if fix_hollow:

        # with LogTimer("Fix hollow cross") as log:
        junction_hollow_cross = scimage.morphology.binary_hit_or_miss(bin_skel, *hollow_cross_mask)
        # log.print('Hollow cross found')
        skel -= scimage.convolve(junction_hollow_cross.astype(np.int8), sqr3.astype(np.int8))
        skel = skel.clip(0) + junction_hollow_cross

        # log.print('Hollow cross quick fix, starting post fix')
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
    # with LogTimer('Detect endpoints') as log:
    bin_skel = skel > 0
    skel -= fast_hit_or_miss(bin_skel, bin_skel, *endpoint_masks)

    # with LogTimer('Remove 1px small end branches'):
    if max_spurs_length > 0:
        # Remove small end branches
        skel = remove_1px_endpoints(skel, endpoint_masks, sqr3)

    # === Create branches labels ===
    branches_label_required = branches_label is not None or max_spurs_length >= 2
    if branches_label is None and branches_label_required:
        branches_label = np.zeros_like(skel, dtype=np.int32)
    if branches_label_required:
        assert branches_label.shape == vessel_map.shape, "branches_label must have the same shape as vessel_map"
        assert branches_label.dtype == np.int32, "branches_label must be of type np.int32"

        # with LogTimer('Create branches labels'):
        # junctions = skel >= 3
        # junction_neighbours = scimage.binary_dilation(junctions, sqr3)
        # binskel_no_junctions = (skel >= 1) & ~junction_neighbours
        branches_label, branch_list, _ = label_skeleton(skel)
        nb_branches = len(branch_list)
    else:
        nb_branches = 0
        branches_label = None

    # === Remove small branches ===
    if max_spurs_length >= 2:
        # --- Remove larger than 1 px endpoints branches ---
        # with LogTimer('Remove small branches') as log:
        # Select small branches
        branch_sizes = np.bincount(branches_label.ravel())
        small_branches_id = np.where(branch_sizes < max_spurs_length)[0]
        # log.print(f'Found {len(small_branches_id)} small branches.')

        # Select branches which including endpoints
        endpoint_branches_id = np.unique(branches_label[skel == 1])
        # log.print(f'Found {len(endpoint_branches_id)} branches shich includes endpoints.')

        # Identify small branches labels
        branches_to_remove = np.intersect1d(endpoint_branches_id, small_branches_id, assume_unique=True)
        # log.print(f' => {len(branches_to_remove)} branches to be removed.')

        label_lookup = np.zeros(nb_branches + 1, dtype=np.int32)
        label_lookup[branches_to_remove] = 1
        label_lookup = np.arange(nb_branches + 1, dtype=np.int32) - np.cumsum(label_lookup)
        nb_branches = nb_branches - len(branches_to_remove)
        # log.print(f'Computed label_lookup')

        # Remove small branches from the skeleton.
        skel[np.isin(branches_label, branches_to_remove)] = 0

        # Remove small branches labels from the label map
        branches_label[:] = label_lookup[branches_label]

        # log.print(f'Small branch has been removed')
        # At this point small branches have been reduced to 1px endpoints and must be removed again.
        bin_skel = skel > 0
        skel -= fast_hit_or_miss(bin_skel, bin_skel, *endpoint_masks)
        # log.print(f'Endpoints detected again')

        skel = remove_1px_endpoints(skel, endpoint_masks, sqr3)
        # log.print(f'1px branch removed')

    if return_distance:
        return skel, skel_dist
    else:
        return skel
