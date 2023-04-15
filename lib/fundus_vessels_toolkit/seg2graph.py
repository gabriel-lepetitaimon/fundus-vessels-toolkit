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

__all__ = ['skeletonize', 'extract_patches', 'compute_adjacency_branches_nodes']

import numpy as np
import networkx as nx
from typing import Tuple, Dict

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
                     [0, 1, 0]],
                    [[2, 2, 2],
                     [2, 1, 1],
                     [2, 1, 1]],
                ])
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
                branches_label: np.ndarray | None = None, skeletonize_method=None):
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
    # with LogTimer("Compute skeleton"):
    if skeletonize_method == 'medial_axis':
        bin_skel, skel_dist = medial_axis(vessel_map, return_distance=return_distance, random_state=0)
    else:
        bin_skel = skeletonize(vessel_map, method=skeletonize_method) > 0
        skel_dist = scimage.distance_transform_edt(bin_skel) if return_distance else None
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

        # with LogTimer('Create branches labels'):
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
        # with LogTimer('Remove small branches') as log:
        # Select small branches
        branch_sizes = np.bincount(branches_label.ravel())
        small_branches_id = np.where(branch_sizes < remove_small_ends)[0]
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
    assert positive_patterns.shape[1] == samples.shape[1], "samples and positive_patterns must have the same number of columns " \
                                                           f"positive_patterns.shape = {positive_patterns.shape}, samples.shape = {samples.shape}"
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


def compute_adjacency_branches_nodes(vessel_map: np.ndarray, return_label=False, nodes_merge_distance=True,
                                     merge_small_cycles=0, simplify_topology=True):
    """
    Extract the naive vasculature graph from a vessel map.
    If return label is True, the label map of branches and nodes are also computed and returned.

    Small topological corrections are applied to the graph:
        - nodes too close to each other are merged (max distance=5√2/2 by default)
        - cycles with small perimeter are merged (max size=15% of the image width by default)
        - if simplify_topology is True, the graph is simplified by merging equivalent branches (branches connecting the same junctions)
            and removing nodes with degree 2.

    Args:
        vessel_map: The vessel map. Shape: (H, W) where H and W are the map height and width.
        return_label: If True, return the label map of branches and nodes.
        nodes_merge_distance: If not 0, nodes separated by less than this distance are merged (5√2/2 by default).
        merge_small_cycles: If not 0, cycles with a perimeter smaller than this value are merged (disabled by default).
        simplify_topology: If True, the graph is simplified by merging equivalent branches (branches connecting the same junctions)
            and removing nodes with degree 2.

    Returns:
        The adjacency matrix of the graph.
        Shape: (nBranch, nNode) where nBranch and nNode are the number of branches and nodes (junctions or terminations).

        If return_label is True, also return the label map of branches
            (where each branch of the skeleton is labeled by a unique integer corresponding to its index in the adjacency matrix)
        and the coordinates of the nodes as a tuple of (y, x) where y and x are vectors of length nNode.
    """
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
    nodes = (skel >= 3) | (skel == 1)
    sqr3 = np.ones((3, 3), dtype=bool)

    # Label branches
    skel_no_nodes = bin_skel & ~scimage.binary_dilation(nodes, sqr3)
    labeled_branches, nb_branches = label(skel_no_nodes, return_num=True)
    labeled_branches = expand_labels(labeled_branches, 2) * (bin_skel & ~nodes)

    # Label nodes
    node_y, node_x = np.where(nodes)
    labels_nodes = np.zeros_like(labeled_branches)
    labels_nodes[node_y, node_x] = np.arange(1, len(node_y)+1)
    nb_nodes = len(node_y)

    # Identify branches connected to each junction
    ring_pattern = np.asarray([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=bool)
    nodes_ring = extract_unravelled_pattern(labeled_branches, (node_y, node_x), ring_pattern, return_coordinates=False)

    # Build the matrix of connections from branches to nodes
    branches_by_nodes = np.zeros((nb_branches+1, nb_nodes), dtype=bool)
    branches_by_nodes[nodes_ring.flatten(), np.repeat(np.arange(nb_nodes), ring_pattern.sum())] = 1
    branches_by_nodes = branches_by_nodes[1:, :]

    if nodes_merge_distance is True:
        nodes_merge_distance = 2.5 * np.sqrt(2)
    if nodes_merge_distance > 0:
        # Merge nodes clusters
        # - Generate the structure placed at each junction to detect its neighbor nodes which should be merged
        cluster_structure = skmorph.disk(nodes_merge_distance) > 0
        structure_numel = cluster_structure.sum()
        # - Extract the nodes index in the neighborhood of each junction; shape: [nb_nodes, structure_numel]
        nodes_clusters_by_nodes = extract_unravelled_pattern(labels_nodes, (node_y, node_x), cluster_structure)
        # - Build the proximity connectivity matrix between nodes
        nodes_clusters = np.zeros((nb_nodes+1, nb_nodes), dtype=bool)
        nodes_clusters[nodes_clusters_by_nodes,
                           np.tile(np.arange(nb_nodes)[:, None], (1, structure_numel))] = 1
        nodes_clusters = nodes_clusters[1:, :] - np.eye(nb_nodes)
        # - Identify clusters as a list of lists of nodes indices
        nodes_clusters = [_ for _ in nx.find_cliques(nx.from_numpy_array(nodes_clusters))
                              if len(_) > 1]
        # - Merge nodes
        branches_by_nodes, branch_lookup, node_lookup = merge_junctions_clusters(branches_by_nodes, nodes_clusters)
        if node_lookup is not None:
            node_y, node_x = apply_node_lookup_on_coordinates((node_y, node_x), node_lookup)
            nb_nodes = len(node_y)
    else:
        branch_lookup = None

    # Merge small cycles
    if merge_small_cycles > 0:
        if merge_small_cycles < 1:
            merge_small_cycles = merge_small_cycles * vessel_map.shape[0]
        # - Identify cycles
        node_adjency_matrix = branches_by_nodes.T@branches_by_nodes-np.eye(nb_nodes)
        cycles = [_ for _ in nx.chordless_cycles(nx.from_numpy_array(node_adjency_matrix)) if len(_) > 2]
        # - Select chord less cycles with small perimeter
        cycles_perimeters = [perimeter_from_vertices(np.asarray((node_y, node_x)).T[cycle]) for cycle in cycles]
        cycles = [cycle for cycle, perimeter in zip(cycles, cycles_perimeters)
                   if perimeter < merge_small_cycles]
        # - Merge cycles
        branches_by_nodes, branch_lookup_2, node_lookup = merge_junctions_clusters(branches_by_nodes, cycles,
                                                                                   remove_branches_labels=False)
        # - Apply the lookup tables on nodes and branches
        if node_lookup is not None:
            node_y, node_x = apply_node_lookup_on_coordinates((node_y, node_x), node_lookup)
        if branch_lookup_2 is not None:
            if branch_lookup is not None:
                branch_lookup = branch_lookup_2[branch_lookup]
            else:
                branch_lookup = branch_lookup_2

    if simplify_topology:
        # Merge equivalent branches
        branches_by_nodes, branch_lookup_2, node_to_keep = merge_equivalent_branches(branches_by_nodes)
        # - Apply the lookup tables on nodes and branches
        if node_to_keep is not None:
            node_y, node_x = node_y[node_to_keep], node_x[node_to_keep]
        if branch_lookup_2 is not None:
            if branch_lookup is not None:
                branch_lookup = branch_lookup_2[branch_lookup]
            else:
                branch_lookup = branch_lookup_2

    if return_label:
        if branch_lookup is not None:
            # Update branch labels
            branch_lookup = np.concatenate(([0], branch_lookup+1))
            labeled_branches = branch_lookup[labeled_branches]
        return branches_by_nodes, labeled_branches, (node_y, node_x)
    else:
        return branches_by_nodes


def branches_by_nodes_to_node_graph(branches_by_nodes, node_pos=None):
    branches = np.arange(branches_by_nodes.shape[0]) + 1
    branches_by_nodes = branches_by_nodes.astype(bool)
    node_adjacency = branches_by_nodes.T @ (branches_by_nodes * branches[:, None])
    graph = nx.from_numpy_array((node_adjacency > 0) & (~np.eye(branches_by_nodes.shape[1], dtype=bool)))
    if node_pos is not None:
        node_y, node_x = node_pos
        nx.set_node_attributes(graph, node_y, 'y')
        nx.set_node_attributes(graph, node_x, 'x')
    for edge in graph.edges():
        graph.edges[edge]['branch'] = node_adjacency[edge[0], edge[1]] - 1
    return graph


def merge_junctions_clusters(branches_by_junctions, junctions_clusters, remove_branches_labels=True):
    """
    Merge junctions in the connectivity matrix of branches and junctions.
    """
    node_table = np.arange(branches_by_junctions.shape[1], dtype=np.int64)
    branches_to_remove = np.zeros(branches_by_junctions.shape[0], dtype=bool)
    branches_lookup = np.arange(branches_by_junctions.shape[0], dtype=np.int64)
    node_to_remove = np.zeros(branches_by_junctions.shape[1], dtype=bool)

    for cluster in junctions_clusters:
        cluster = np.asarray(tuple(cluster), dtype=np.int64)
        cluster.sort()
        cluster_branches = np.where(np.sum(branches_by_junctions[:, cluster].astype(bool), axis=1) >= 2)[0]
        cluster_branches.sort()
        branches_to_remove[cluster_branches] = True
        incoming_cluster_branches = np.where(np.sum(branches_by_junctions[:, cluster].astype(bool), axis=1) == 1)[0]
        if len(incoming_cluster_branches):
            apply_lookup_inplace((cluster_branches, branches_lookup[incoming_cluster_branches[0]]), branches_lookup)
        else:
            branches_lookup[cluster_branches] = np.nan
        node_table[cluster] = cluster[0]
        node_to_remove[cluster[1:]] = True

    if branches_to_remove.any():
        branches_by_junctions = branches_by_junctions[~branches_to_remove, :]
        branches_lookup = (np.cumsum(~branches_to_remove) - 1)[branches_lookup]
        if remove_branches_labels:
            branches_lookup[branches_to_remove] = np.nan
    else:
        branches_lookup = None
    nb_branches = branches_by_junctions.shape[0]

    node_lookup = np.cumsum(~node_to_remove) - 1
    nb_nodes = node_lookup[-1] + 1
    node_table = node_lookup[node_table]

    nodes_by_branches = np.zeros_like(branches_by_junctions, shape=(nb_nodes, nb_branches))
    np.add.at(nodes_by_branches, node_table, branches_by_junctions.T)

    return nodes_by_branches.T, branches_lookup, node_table


def merge_equivalent_branches(branches_by_node, remove_2branches_nodes=True):
    branches_by_node, branches_lookup = np.unique(branches_by_node, return_inverse=True, axis=0)
    nodes_to_remove = None
    if remove_2branches_nodes:
        nodes_to_remove = np.sum(branches_by_node.astype(bool), axis=0) == 2
        if np.any(nodes_to_remove):
            branches_by_node, branch_lookup2 = fuse_node(branches_by_node, nodes_to_remove)
            branches_lookup = branch_lookup2[branches_lookup]
    return branches_by_node, branches_lookup, ~nodes_to_remove


def fuse_node(branches_by_nodes, nodes_id):
    """
    Remove a node from the connectivity matrix of branches and nodes.
    """
    assert nodes_id.ndim == 1, "nodes_id must be a 1D array"
    if nodes_id.dtype == bool:
        assert len(nodes_id) == branches_by_nodes.shape[1], "nodes_id must be a boolean array of the same length as the number of nodes," \
                                                            f" got len(nodes_id)={len(nodes_id)} instead of {branches_by_nodes.shape[1]}."
        nodes_id = np.where(nodes_id)[0]
    assert nodes_id.dtype == np.int64, "nodes_id must be a boolean or integer array"

    nb_branches = branches_by_nodes.shape[0]
    branch_lookup = np.arange(nb_branches, dtype=np.int64)

    branches_by_fused_nodes = branches_by_nodes[:, nodes_id]
    invalid_fused_node = np.sum(branches_by_fused_nodes, axis=0) > 2
    if np.any(invalid_fused_node):
        print("Warning: some nodes are connected to more than 2 branches and won't be fused.")
        branches_by_fused_nodes = branches_by_fused_nodes[:, ~invalid_fused_node]
    branch1_ids = np.argmax(branches_by_fused_nodes, axis=0)
    branch2_ids = nb_branches - np.argmax(branches_by_fused_nodes[::-1], axis=0) - 1

    sort_id = np.argsort(branch1_ids)[::-1]
    branch1_ids = branch1_ids[sort_id]
    branch2_ids = branch2_ids[sort_id]

    # Sequential merge is required when a branch appear both in branch1_ids and branch2_ids (because 2 adjacent nodes are fused)
    for b1, b2 in zip(branch1_ids, branch2_ids):
        branches_by_nodes[b1] |= branches_by_nodes[b2]
    for b1, b2 in zip(branch1_ids[::-1], branch2_ids[::-1]):
        branch_lookup[b2] = branch_lookup[b1]
    # Instead of:
    # branch_subids, branch2_nodes_ids = np.where(branches_by_nodes[branch2_ids])
    # branches_by_nodes[branch1_ids[branch_subids], branch2_nodes_ids] = True

    branches_by_nodes = np.delete(branches_by_nodes, branch2_ids, axis=0)
    branches_by_nodes = np.delete(branches_by_nodes, nodes_id, axis=1)

    branch_shift_lookup = np.cumsum(np.isin(np.arange(len(branch_lookup)), branch2_ids, invert=True))-1
    branch_lookup = branch_shift_lookup[branch_lookup]

    # for b1, n in zip(branch1_ids[branch_subids], branch2_nodes_ids):
    #     print(f"redirect branches: {b1} to node: {n}")

    return branches_by_nodes, branch_lookup


def apply_node_lookup_on_coordinates(junctions_coord, junction_lookup):
    """
    Apply a lookup table on a set of coordinates to merge coordinates of junctions and return their barycenter.
    """
    jy, jx = junctions_coord
    coord = np.zeros((np.max(junction_lookup) + 1, 2), dtype=np.float64)
    np.add.at(coord, junction_lookup, np.asarray((jy, jx)).T)
    coord = coord / np.bincount(junction_lookup, minlength=coord.shape[0])[:, np.newaxis]
    return coord.T


def perimeter_from_vertices(coord: np.ndarray):
    coord = np.asarray(coord)
    next_coord = np.roll(coord, 1, axis=0)
    return np.sum(np.linalg.norm(coord - next_coord, axis=1))


def vascular_graph_edit_distance(branch_to_node1, node1_yx, branch_to_node2, node2_yx):
    ny1, nx1 = node1_yx
    ny2, nx2 = node2_yx
    ny1 = ny1[:, None]
    nx1 = nx1[:, None]
    ny2 = ny2[None, :]
    nx2 = nx2[None, :]
    node_dist = np.sqrt((ny1 - ny2)**2 + (nx1 - nx2)**2)
    del ny1, nx1, ny2, nx2

    node_extended_match = node_dist < 10
    node_dist = 1/(node_dist+1e8)
    node_dist[~node_extended_match] = 0
    n1_match = np.where(np.sum(node_extended_match, axis=1))[0]
    n2_match = np.argmax(node_dist[n1_match], axis=0)
    del node_dist, node_extended_match

    lookup_n1_idx = np.concatenate([n1_match, np.isin(np.arange(len(node1_yx[0])), n1_match, invert=True, assume_unique=True)])
    lookup_n2_idx = np.concatenate([n2_match, np.isin(np.arange(len(node2_yx[0])), n2_match, invert=True, assume_unique=True)])

    node_to_branch1 = branch_to_node1.T[lookup_n1_idx]
    node_to_branch2 = branch_to_node2.T[lookup_n2_idx]

    g1 = branches_by_nodes_to_node_graph(node_to_branch1.T)
    g2 = branches_by_nodes_to_node_graph(node_to_branch2.T)

    return 0


def apply_lookup_inplace(mapping: Dict[int, int] | Tuple[np.ndarray, np.ndarray] | np.array, array: np.ndarray) -> np.ndarray:
    lookup = mapping
    if not isinstance(mapping, np.ndarray):
        lookup = np.arange(len(array), dtype=np.int64)
        if isinstance(mapping, dict):
            mapping = tuple(zip(mapping.keys(), mapping.values()))
        search = mapping[0]
        replace = mapping[1]
        if not isinstance(replace, np.ndarray):
            replace = [replace] * len(search)
        for s, r in zip(search, replace):
            lookup[lookup == s] = r
    nan_array = np.isnan(array)
    if np.any(nan_array):
        array[~nan_array] = lookup[array[~nan_array]]
    else:
        array[:] = lookup[array]
    return array
