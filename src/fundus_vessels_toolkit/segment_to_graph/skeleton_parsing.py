import warnings
from typing import List, Optional, Tuple

import numpy as np
import scipy.ndimage as scimage
import torch

from ..utils.cpp_extensions.graph_cpp import detect_skeleton_nodes as detect_skeleton_nodes_cpp
from ..utils.cpp_extensions.graph_cpp import detect_skeleton_nodes_debug as detect_skeleton_nodes_debug_cpp
from ..utils.cpp_extensions.graph_cpp import parse_skeleton as parse_skeleton_cpp
from ..utils.cpp_extensions.graph_cpp import parse_skeleton_with_cleanup as parse_skeleton_with_cleanup_cpp
from ..utils.lookup_array import create_removal_lookup
from ..utils.torch import autocast_torch
from ..vascular_data_objects import FundusData, VBranchGeoData, VGeometricData, VGraph


def skeleton_to_vgraph(
    skeleton_map: torch.Tensor | np.ndarray,
    vessels: Optional[torch.Tensor | np.ndarray | FundusData] = None,
    fix_hollow=True,
    clean_branches_tips=20,
    clean_terminal_branches_tips=10,
    min_terminal_branch_length=4,
    min_terminal_branch_calibre_ratio=1,
    max_spurs_length=30,
) -> VGraph:
    """
    Parse a skeleton image into a graph of branches and nodes.

    Parameters
    ----------
    skeleton_map : np.ndarray | torch.Tensor
        Binary image of the vessel skeleton.

    segmentation_map : np.ndarray | torch.Tensor | FundusData
        Segmentation map of the image. If provided, the function will remove small branches and clean the branches tips.

    fix_hollow:
        If True (by default), hollow cross pattern are filled and considered as a 4-branches junction.

    clean_branches_tips:
        If > 0, clean the skeleton extremities of each branches.
        This step ensure that the branch skeleton actually starts where the branch emerge from the junction/bifurcation and not at the center of the junction/bifurcation. Skeleton inside the junction/bifurcation is often not relevant and may affect the accuracy of tangent and calibre estimation.

        The value of ``clean_branches_tips`` is the maximum number of pixel that can be removed through this process.

    min_terminal_branch_length :
        If > 0, remove terminal branches that are shorter than this value in pixel.

    min_terminal_branch_calibre_ratio :
        If > 0, remove terminal branches that are shorter than this value times the calibre of its largest adjacent branch.

    max_spurs_length :
        If > 0, prevent the removal of a terminal branch longer than this value.

    Returns
    -------
    A :class:`VGraph` object containing the graph of the skeleton.
    """  # noqa: E501
    if isinstance(skeleton_map, np.ndarray):
        skeleton_map = torch.from_numpy(skeleton_map)
    skeleton_map = skeleton_map.cpu()

    fundus_data = None
    if vessels is not None:
        if isinstance(vessels, FundusData):
            fundus_data = vessels
            vessels = vessels.vessels
        if isinstance(vessels, np.ndarray):
            vessels = torch.from_numpy(vessels)
        vessels = vessels.cpu().bool()

    if skeleton_map.dtype == torch.bool:
        remove_endpoint_branches = min_terminal_branch_length > 0 or min_terminal_branch_calibre_ratio > 0
        skeleton_map = detect_skeleton_nodes(
            skeleton_map, fix_hollow=fix_hollow, remove_endpoint_branches=remove_endpoint_branches
        )

    skeleton_map = skeleton_map.int()

    outs = parse_skeleton(
        skeleton_map,
        segmentation_map=vessels,
        clean_branches_tips=clean_branches_tips,
        clean_terminal_branches_tips=clean_terminal_branches_tips,
        min_terminal_branch_length=min_terminal_branch_length,
        min_terminal_branch_calibre_ratio=min_terminal_branch_calibre_ratio,
        max_spurs_length=max_spurs_length,
    )
    branch_list, nodes_yx, branches_curve = outs[1:4]
    labels = outs[0]

    branch_list = branch_list.numpy()
    nodes_indexes = np.unique(branch_list)
    if len(nodes_indexes) != len(nodes_yx):
        orphan_nodes = np.setdiff1d(np.arange(len(nodes_indexes)), nodes_indexes)
        warnings.warn(
            f"Nodes {orphan_nodes} are not connected to any branch and will be ignored. "
            "This is however an abnormal behavior as they should have been removed already.",
            stacklevel=2,
        )
        del_lookup = create_removal_lookup(orphan_nodes, length=len(nodes_yx))
        branch_list = del_lookup[branch_list]
        nodes_yx = np.delete(nodes_yx, orphan_nodes, axis=0)

    geo_data = VGeometricData(
        nodes_coord=nodes_yx.numpy(),
        branches_curve=[_.numpy() for _ in branches_curve],
        domain=labels.shape,
        fundus_data=fundus_data,
    )
    if vessels is not None and clean_branches_tips > 0:
        tangents, calibres, boundaries = [], [], []
        for t in outs[-1]:
            tangents.append(VBranchGeoData.TipsTangents(t[:, :2].numpy()))
            calibres.append(VBranchGeoData.TipsScalar(t[:, 2].numpy()))
            boundaries.append(VBranchGeoData.Tips2Points(t[:, 3:7].reshape(2, 2, 2).numpy()))
        geo_data.set_branch_data(VBranchGeoData.Fields.TIPS_TANGENT, tangents)
        geo_data.set_branch_data(VBranchGeoData.Fields.TIPS_CALIBRE, calibres)
        geo_data.set_branch_data(VBranchGeoData.Fields.TIPS_BOUNDARIES, boundaries)

    return VGraph(branch_list, geo_data, nodes_count=len(nodes_yx))


@autocast_torch
def parse_skeleton(
    skeleton_map: torch.Tensor,
    segmentation_map: Optional[torch.Tensor] = None,
    clean_branches_tips=20,
    clean_terminal_branches_tips=10,
    min_terminal_branch_length=3,
    min_terminal_branch_calibre_ratio=1,
    max_spurs_length=30,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Label the skeleton map with the junctions and endpoints.

    Parameters
    ----------
    skeleton_map :
        The skeleton of the vessels either as a binary image or an integer map encoded as:
        - 0: background pixel
        - 1: Vessel branch pixel
        - 2: Vessel endpoint or junction

    segmentation_map :
        The segmentation map of the image as a binary image.

    clean_branches_tips :
        If > 0, clean at most this number of pixels at the node extremities of each branches.

    clean_terminal_branches_tips :
        If > 0, clean at most this number of pixels at the extremities of terminal branches.

    min_terminal_branch_length :
        If > 0, remove terminal branches that are shorter than this value in pixel.

    min_terminal_branch_calibre_ratio :
        If > 0, remove terminal branches that are shorter than this value times the calibre of its largest adjacent branch.

    max_spurs_length :
        If > 0, prevent the removal of a terminal branch longer than this value.

    Returns
    -------
    labels map :
        The skeletonized image where each pixel is described by an integer value as:
            - 0: background
            - > 0: Vessel branch (the branch ID is the pixel value -1)
            - < 0: Vessel endpoint or junction (the node ID is the absolute pixel value -1)

    branch list : Array[B, 2]
        Array of shape [B, 2] Adjacency list of the skeleton graph.

    node coordinates : Array[N, 2]
        Array of shape [N, 2]: the (y, x) positions of the nodes in the skeleton.

    branches coordinates : list[Array[, 2]]
        List of B arrays of shape [l, 2] containing the (y, x) positions of the consecutive pixels in the branch l.

    branches tips infos : Array[B, 2, 7]
        Array of shape [B, 2, 7] containing geometrical information about each branch extremities.
        The data are organized as follow:
            - [..., 0:2]: The tangents (ty, tx) at the branches tips.
            - [..., 2]: The calibre at the tips.
            - [..., 3:7]: The branch boundaries (b1y, b1x, b2y, b2x) at the tips.
    """  # noqa: E501
    skeleton_map = skeleton_map.cpu()
    if skeleton_map.dtype == torch.bool:
        skeleton_map = detect_skeleton_nodes(skeleton_map)
    skeleton_map = skeleton_map.int()

    if segmentation_map is not None:
        segmentation_map = segmentation_map.cpu().bool()
        opts = dict(
            clean_branches_tips=clean_branches_tips,
            clean_terminal_branches_tips=clean_terminal_branches_tips,
            min_spurs_length=min_terminal_branch_length,
            spurs_calibre_factor=min_terminal_branch_calibre_ratio,
            max_spurs_length=max_spurs_length,
        )
        out = parse_skeleton_with_cleanup_cpp(skeleton_map, segmentation_map, opts)
    else:
        out = parse_skeleton_cpp(skeleton_map)
    return (skeleton_map,) + out


@autocast_torch
def detect_skeleton_nodes(
    skeleton_map: np.ndarray | torch.Tensor,
    fix_hollow=True,
    remove_endpoint_branches=True,
    *,
    debug=False,
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
    if debug:
        return detect_skeleton_nodes_debug_cpp(skeleton_map)
    skeleton_rank = detect_skeleton_nodes_cpp(skeleton_map.cpu().bool(), fix_hollow, remove_endpoint_branches)
    rank_lookup = torch.tensor([0, 3, 1, 2, 2, 2], dtype=torch.uint8)
    return rank_lookup[skeleton_rank]


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
