from typing import Optional, Tuple

import numpy as np
import scipy.ndimage as scimage
import torch

from ..utils.cpp_extensions.graph_cpp import detect_skeleton_nodes as detect_skeleton_nodes_cpp
from ..utils.cpp_extensions.graph_cpp import parse_skeleton as parse_skeleton_cpp
from ..utils.cpp_extensions.graph_cpp import parse_skeleton_with_cleanup as parse_skeleton_with_cleanup_cpp
from ..utils.torch import autocast_torch
from ..vascular_data_objects import VBranchGeoData, VGeometricData, VGraph


def skeleton_to_vgraph(
    skeleton_map: torch.Tensor | np.ndarray,
    segmentation_map: Optional[torch.Tensor | np.ndarray] = None,
    fix_hollow=True,
    clean_branches_extremities=20,
    max_spurs_length=0,
    max_spurs_calibre_factor=1,
) -> VGraph:
    """
    Parse a skeleton image into a graph of branches and nodes.

    Parameters
    ----------
    skeleton_map : np.ndarray | torch.Tensor
        Binary image of the vessel skeleton.

    segmentation_map : np.ndarray | torch.Tensor
        Segmentation map of the image. If provided, the function will remove small branches and clean the terminations.

    fix_hollow:
        If True (by default), hollow cross pattern are filled and considered as a 4-branches junction.

    clean_branches_extremities:
        If > 0, clean the skeleton extremities of each branches.
        This step ensure that the branch skeleton actually starts where the branch emerge from the junction/bifurcation and not at the center of the junction/bifurcation. Skeleton inside the junction/bifurcation is often not relevant and may affect the accuracy of tangent and calibre estimation.

        The value of ``clean_branches_extremities`` is the maximum number of pixel that can be removed through this process.

    max_spurs_length:
        If > 0, remove spurs (short terminal branches) that are shorter than this value in pixel.

    max_spurs_calibre_factor:
        If > 0, remove spurs (short terminal branches) that are shorter than ``max_spurs_calibre_factor`` times the calibre of the largest branch adjacent to the spurs.

    Returns
    -------
    A :class:`VGraph` object containing the graph of the skeleton.
    """  # noqa: E501
    if isinstance(skeleton_map, np.ndarray):
        skeleton_map = torch.from_numpy(skeleton_map)
    skeleton_map = skeleton_map.cpu()
    if segmentation_map is not None:
        if isinstance(segmentation_map, np.ndarray):
            segmentation_map = torch.from_numpy(segmentation_map)
        segmentation_map = segmentation_map.cpu().bool()

    if skeleton_map.dtype == torch.bool:
        remove_endpoint_branches = max_spurs_length > 0 or max_spurs_calibre_factor > 0
        skeleton_map = detect_skeleton_nodes(
            skeleton_map, fix_hollow=fix_hollow, remove_endpoint_branches=remove_endpoint_branches
        )

    skeleton_map = skeleton_map.int()

    outs = parse_skeleton(
        skeleton_map,
        segmentation_map=segmentation_map,
        clean_branches_extremities=clean_branches_extremities,
        max_spurs_length=max_spurs_length,
        max_spurs_calibre_factor=max_spurs_calibre_factor,
    )
    branch_list, nodes_yx, branches_curve = outs[:3]
    labels = outs[-1]

    geo_data = VGeometricData(
        nodes_coord=nodes_yx.numpy(),
        branches_curve=[_.numpy() for _ in branches_curve],
        domain=labels.shape,
    )
    if segmentation_map is not None and clean_branches_extremities > 0:
        tangents_calibres = outs[3]
        tangents = [VBranchGeoData.TerminationTangents(t[:, :2].numpy()) for t in tangents_calibres]
        calibres = [VBranchGeoData.TerminationData(t[:, 2].numpy()) for t in tangents_calibres]
        boundaries = [VBranchGeoData.TerminationData(t[:, 3:7].reshape(2, 2, 2).numpy()) for t in tangents_calibres]
        geo_data.set_branch_data(VBranchGeoData.Fields.TERMINATION_TANGENTS, tangents)
        geo_data.set_branch_data(VBranchGeoData.Fields.TERMINATION_CALIBRES, calibres)
        geo_data.set_branch_data(VBranchGeoData.Fields.TERMINATION_BOUNDARIES, boundaries)

    return VGraph(branch_list.numpy(), geo_data)


@autocast_torch
def parse_skeleton(
    skeleton_map: torch.Tensor,
    segmentation_map: Optional[torch.Tensor] = None,
    clean_branches_extremities=0,
    max_spurs_length=0,
    max_spurs_calibre_factor=1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """# noqa: E501
    Label the skeleton map with the junctions and endpoints.

    Parameters
    ----------
    skeleton_map : np.ndarray | torch.Tensor
        Binary image of the vessel skeleton.

    Returns
    -------
    labels map : np.ndarray
        The skeletonized image where each pixel is described by an integer value as:
            - 0: background
            - > 0: Vessel branch (the branch ID is the pixel value -1)
            - < 0: Vessel endpoint or junction (the node ID is the absolute pixel value -1)
    branch list : np.ndarray
        Array of shape [B, 2] Adjacency list of the skeleton graph.
    node coordinates : np.ndarray
        Array of shape [N, 2]: the (y, x) positions of the nodes in the skeleton.
    branches coordinates : list[np.ndarray]
        List of B arrays of shape [l, 2] containing the (y, x) positions of the consecutive pixels in the branch l.
    """
    skeleton_map = skeleton_map.cpu()
    if skeleton_map.dtype == torch.bool:
        skeleton_map = detect_skeleton_nodes(skeleton_map)
    skeleton_map = skeleton_map.int()

    if segmentation_map is not None:
        segmentation_map = segmentation_map.cpu().bool()
        opts = dict(
            clean_terminations=clean_branches_extremities,
            max_spurs_length=max_spurs_length,
            max_spurs_calibre_ratio=max_spurs_calibre_factor,
        )
        out = parse_skeleton_with_cleanup_cpp(skeleton_map, segmentation_map, opts)
    else:
        out = parse_skeleton_cpp(skeleton_map)
    return out + (skeleton_map,)


@autocast_torch
def detect_skeleton_nodes(
    skeleton_map: np.ndarray | torch.Tensor,
    fix_hollow=True,
    remove_endpoint_branches=True,
):
    """
    Parse a skeleton mask to detect junctions and endpoints.

    This function may optionally fix hollow cross patterns and remove terminal branches that are made of only an endpoint.

    Parameters
    ----------
    skeleton_map : np.ndarray | torch.Tensor
        Binary image of the vessel skeleton.

    fix_hollow:
        If True (by default), hollow cross pattern are filled and considered as a 4-branches junction.

    remove_endpoint_branches:
        If True (by default), remove terminal branches that are made of only an endpoint.

    Returns
    -------
        The skeletonize image where each pixel is described by an integer value as:
            - 0: background
            - 1: Vessel branch
            - 2: Vessel endpoint or junction
    """
    cast_numpy = isinstance(skeleton_map, np.ndarray)
    if cast_numpy:
        skeleton_map = torch.from_numpy(skeleton_map)

    skeleton_rank = detect_skeleton_nodes_cpp(skeleton_map.cpu().bool(), fix_hollow, remove_endpoint_branches)
    rank_lookup = torch.tensor([0, 3, 1, 2, 2, 2], dtype=torch.uint8)
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
    branch list : np.ndarray
        Array of shape [B, 2] Adjacency list of the skeleton graph.

    branch labels : np.ndarray
        Label of each branch in the skeleton.

    node coordinates : np.ndarray
        Array of shape [N, 2]: the (y, x) positions of the nodes in the skeleton.
    """  # noqa: E501
    from ..utils.graph.graph_cy import label_skeleton

    if skeleton_map.dtype == bool:
        skeleton_map = detect_skeleton_nodes_legacy(skeleton_map)

    branch_labels, branch_list, node_yx = label_skeleton(skeleton_map)

    return branch_list, branch_labels, node_yx


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
    """  # noqa: E501
    from skimage.morphology import remove_small_objects

    from ..utils.binary_mask import extract_patches, fast_hit_or_miss
    from ..utils.skeleton_legacy import (
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
