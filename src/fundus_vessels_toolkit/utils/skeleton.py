from typing import Tuple

import numpy as np

from .binary_mask import fast_hit_or_miss

# Cached masks for junctions and endpoints detections
_junctions_endpoints_masks_cache = None


def compute_junction_endpoint_masks():
    global _junctions_endpoints_masks_cache
    if _junctions_endpoints_masks_cache is None:

        def create_line_junction_masks(n_lines: int | Tuple[int, ...] = (3, 4)):
            if isinstance(n_lines, int):
                n_lines = (n_lines,)
            masks = []
            if 3 in n_lines:
                # fmt: off
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
                # fmt: on

                masks_3 = np.asarray(masks_3)
                masks_3 = np.concatenate([np.rot90(masks_3, k=k, axes=(1, 2)) for k in range(4)])
                masks += [masks_3]
            if 4 in n_lines:
                # fmt: off
                masks_4 = np.asarray([
                    [[1, 0, 1],
                     [0, 1, 0],
                     [1, 0, 1]],

                    [[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]],

                    [[2, 2, 2],
                     [2, 1, 1],
                     [2, 1, 1]],
                ])
                # fmt: on
                masks += [masks_4]
            masks = np.concatenate(masks)
            return masks == 1, masks == 0

        junction_3lines_masks = create_line_junction_masks(3)
        junction_4lines_masks = create_line_junction_masks(4)

        hollow_cross_mask = np.asarray([[2, 1, 2], [1, 0, 1], [2, 1, 2]])
        hollow_cross_mask = hollow_cross_mask == 1, hollow_cross_mask == 0

        def create_endpoint_masks():
            # fmt: off
            mask = np.asarray([
                [[2, 1, 2],
                 [0, 1, 0],
                 [0, 0, 0]],
                 
                [[0, 2, 1],
                 [0, 1, 2],
                 [0, 0, 0]],
            ])
            solo_mask = np.asarray([
                    [[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]
            ])
            # fmt: on
            mask = np.concatenate([np.rot90(mask, k, axes=(1, 2)) for k in range(4)] + [solo_mask])
            return mask == 1, mask == 0

        endpoint_masks = create_endpoint_masks()

        _junctions_endpoints_masks_cache = (
            junction_3lines_masks,
            junction_4lines_masks,
            hollow_cross_mask,
            endpoint_masks,
        )

    return _junctions_endpoints_masks_cache


def remove_1px_endpoints(skel, endpoint_masks=None, sqr3=None):
    """
    Remove terminal branches (branches with one extremity not connected to any other branches) that are exactly 1px long.

    Args:
       skel: The skeleton map in standard format (0: background, 1: endpoints, 2: branch, 3 & 4: junctions).

    Returns:
        The skeleton map with the 1px branches removed and their junctions updated accordingly.
    """

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
