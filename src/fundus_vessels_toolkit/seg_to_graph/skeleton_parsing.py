from typing import Tuple

import numpy as np
import scipy.ndimage as scimage
import torch

from ..utils.cpp_extensions.skeleton_parsing_cpp import detect_skeleton_nodes as detect_skeleton_nodes_cpp
from ..utils.cpp_extensions.skeleton_parsing_cpp import parse_skeleton as parse_skeleton_cpp
from ..utils.graph.branch_by_nodes import adjacency_list_to_branch_by_nodes


def parse_skeleton(skeleton_map: np.ndarray | torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Label the skeleton map with the junctions and endpoints.

    Parameters
    ----------
    skeleton_map : np.ndarray | torch.Tensor
        Binary image of the vessel skeleton.

    Returns
    -------
    branch labels : np.ndarray
        Label of each branch in the skeleton.
    adjacency list : np.ndarray
        Array of shape [E, 2] Adjacency list of the skeleton graph.
    node coordinates : np.ndarray
        Array of shape [N, 2]: the (y, x) positions of the nodes in the skeleton.
    """
    if isinstance(skeleton_map, np.ndarray):
        skeleton_map = torch.from_numpy(skeleton_map)
    if skeleton_map.dtype == torch.bool:
        skeleton_map = detect_skeleton_nodes(skeleton_map)

    out = parse_skeleton_cpp(skeleton_map.cpu().int())
    branch_labels, adj_list, nodes_coordinates = tuple(_.numpy() for _ in out[:3])

    branch_by_node = adjacency_list_to_branch_by_nodes(adj_list, n_branches=out[3], branch_labels=branch_labels)

    return branch_by_node, branch_labels, nodes_coordinates


def detect_skeleton_nodes(
    skeleton_map: np.ndarray | torch.Tensor,
    fix_hollow=True,
    remove_endpoint_branches=True,
):
    """
    Parse a skeleton mask to detect junctions and endpoints and remove small branches.

    Parameters
    ----------
    skeleton_map : np.ndarray | torch.Tensor
        Binary image of the vessel skeleton.

    fix_hollow:
        If True (by default), hollow cross pattern are filled and replaced by a 4-branches junction.

    remove_endpoint_branches:
        If True (by default), remove terminal branches that are made of only an endpoint.

    Returns
    -------
        The skeletonized image where each pixel is described by an integer value as:
            - 0: background
            - 1: Vessel branch
            - 2: Vessel endpoint or junction
    """
    cast_numpy = isinstance(skeleton_map, np.ndarray)
    if cast_numpy:
        skeleton_map = torch.from_numpy(skeleton_map)

    skeleton_rank = detect_skeleton_nodes_cpp(skeleton_map.cpu().bool(), fix_hollow, remove_endpoint_branches)
    rank_lookup = torch.tensor([0, 2, 1, 2, 2], dtype=torch.uint8)
    out = rank_lookup[skeleton_rank]

    return out.numpy() if cast_numpy else out


########################################################################################################################
#       LEGACY IMPLEMENTATION
########################################################################################################################
def parse_skeleton_legacy(skeleton_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a skeleton image into a graph of branches and nodes.

    .. warning::
        This function use a python legacy implementation and is slower than the C++ implementation. Use :meth:`label_skeleton` instead.

    Parameters
    ----------

    skeleton_map : np.ndarray
        Binary image of the vessel skeleton.

    Returns
    -------
    branch labels : np.ndarray
        Label of each branch in the skeleton.

    adjacency list : np.ndarray
        Array of shape [E, 2] Adjacency list of the skeleton graph.

    node coordinates : np.ndarray
        Array of shape [N, 2]: the (y, x) positions of the nodes in the skeleton.
    """
    from ..utils.graph.graph_cy import label_skeleton

    if skeleton_map.dtype == bool:
        skeleton_map = detect_skeleton_nodes_legacy(skeleton_map)

    branch_labels, adj_list, node_yx = label_skeleton(skeleton_map)
    branch_by_node = adjacency_list_to_branch_by_nodes(adj_list, branch_labels=branch_labels)

    return branch_by_node, branch_labels, node_yx


def detect_skeleton_nodes_legacy(
    skeleton_map: np.ndarray,
    fix_hollow=True,
    remove_endpoint_branches=True,
) -> np.ndarray:
    """
    Parse a skeleton mask to detect junctions and endpoints and remove small branches.

    .. warning::
        This function use a python legacy implementation and is slower than the C++ implementation. Use :meth:`parse_skeleton_rank` instead.

    Parameters
    ----------
    skeleton_map : np.ndarray | torch.Tensor
        Binary image of the vessel skeleton.

    fix_hollow:
        If True (by default), hollow cross pattern are filled and replaced by a 4-branches junction.

    remove_endpoint_branches:
        If True (by default), remove terminal branches that are made of only an endpoint.

    Returns
    -------
        The skeletonized image where each pixel is described by an integer value as:
            - 0: background
            - 1: Vessel branch
            - 2: Vessel endpoint or junction
    """
    from skimage.morphology import remove_small_objects

    from ..utils.binary_mask import extract_patches, fast_hit_or_miss
    from ..utils.skeleton import (
        compute_junction_endpoint_masks,
        remove_1px_endpoints,
    )

    junction_3lines_masks, junction_4lines_masks, hollow_cross_mask, endpoint_masks = compute_junction_endpoint_masks()
    sqr3 = np.ones((3, 3), dtype=bool)

    # === Compute the medial axis ===
    # with LogTimer("Compute skeleton"):
    bin_skel = skeleton_map.astype(bool)
    remove_small_objects(bin_skel, min_size=3, connectivity=2, out=bin_skel)
    skel = bin_skel.astype(np.int8)
    bin_skel_patches = extract_patches(bin_skel, bin_skel, (3, 3), True)

    # === Build the skeleton with junctions and endpoints ===
    # with LogTimer("Detect junctions"):
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
        for y, x in zip(*np.where(junction_hollow_cross), strict=True):
            neighborhood = skel[y - 1 : y + 2, x - 1 : x + 2]
            if np.sum(neighborhood) == 1:
                skel[y, x] = 0
            if any(
                scimage.binary_hit_or_miss(neighborhood, *m)[1, 1] for m in zip(*junction_4lines_masks, strict=True)
            ):
                skel[y, x] = 3
            elif any(
                scimage.binary_hit_or_miss(neighborhood, *m)[1, 1] for m in zip(*junction_3lines_masks, strict=True)
            ):
                skel[y, x] = 2
            else:
                skel[y, x] = 1

    skel += skel.astype(bool)

    # Detect endpoints
    # with LogTimer('Detect endpoints') as log:
    bin_skel = skel > 0
    skel -= fast_hit_or_miss(bin_skel, bin_skel, *endpoint_masks)

    # with LogTimer('Remove 1px small end branches'):
    if remove_endpoint_branches:
        # Remove small end branches
        skel = remove_1px_endpoints(skel, endpoint_masks, sqr3)

    parsed_lookup = np.array([0, 2, 1, 2, 2], dtype=skel.dtype)
    return parsed_lookup[skel]
