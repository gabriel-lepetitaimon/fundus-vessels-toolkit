from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Self, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import expand_labels

from fundus_toolkits import FundusData
from fundus_vessels_toolkit.utils.lookup_array import invert_complete_lookup
from fundus_vessels_toolkit.vascular_data_objects.vgraph import VGraph

from ...pipelines.avseg_to_tree import NaiveAVSegToTree
from ...utils.math import sigmoid
from ...vascular_data_objects.vbranch_geodata import VBranchGeoData
from ...vascular_data_objects.vtree import VTree, VTreeBranch
from ..av_map_fixing import rasterize_tree_topology


@dataclass
class DeteriorationOpts:
    drop_p: float = 0.05  # Probability to drop each skeleton point
    drop_calibre_threshold: float = 10  # Maximum calibre to drop skeleton points
    drop_calibre_smooth: float = 2.0  # Spread of the sigmoid to drop skeleton points
    drop_segment_min_length: int = 8  # Minimum size of a dropped segment
    drop_segment_avg_length: int = 32  # Minimum size of a dropped segment
    drop_segment_std_length: int = 16  # Standard deviation of the size of a dropped segment
    av_swap_p: float = 0.01  # Probability to swap the artery/vein label of each branch
    av_swap_min_segment_length: int = 10  # Minimum size of a AV swapped segment


def deteriorate_segmentation(
    fundus: FundusData,
    opts: Optional[DeteriorationOpts] = None,
) -> npt.NDArray[np.uint8]:
    deteriorated_map = np.zeros_like(fundus.av)

    av2tree = NaiveAVSegToTree()
    trees = av2tree(fundus)
    trees = deteriorate_trees(trees, opts)
    for i, tree in enumerate(trees):
        tree_map = rasterize_tree_topology(tree, bridge_gap_smaller_than=25, fill_junctions=True)[0] > 0
        tree_map = binary_dilation(tree_map, disk(1))
        tree_map[fundus.av == 0] = 0
        deteriorated_map += tree_map.astype(np.uint8) * (i + 1)

    return deteriorated_map


def deteriorate_trees(trees: tuple[VTree, VTree], opts: Optional[DeteriorationOpts] = None) -> tuple[VTree, VTree]:
    if opts is None:
        opts = DeteriorationOpts()

    # === AV SWAP ===

    # === DISCONNECTIONS ===
    return deteriorate_tree(trees[0], opts), deteriorate_tree(trees[1], opts)


def deteriorate_tree(tree: VTree, opts: Optional[DeteriorationOpts] = None) -> VTree:
    if opts is None:
        opts = DeteriorationOpts()
    tree = tree.copy()

    for b in tree.branches(dynamic_iterator=True):
        curve = b.curve()
        calibres = b.geodata(VBranchGeoData.Fields.CALIBRES)
        if b.curve is not None and len(curve) > opts.drop_segment_avg_length and calibres is not None:
            calibres = calibres.data

            branch_drop_p = (
                opts.drop_p
                * np.sqrt(len(curve))
                * sigmoid((opts.drop_calibre_threshold - calibres.mean()) / opts.drop_calibre_smooth)
            )
            n_drop = branch_drop_p // np.random.rand()
            if n_drop < 1:
                continue

            n_drop = int(np.log2(n_drop)) + 1

            # === DROP SKELETON POINTS ===
            # Sample points to drop
            drop_centers = np.argsort(np.random.rand(len(curve)) * calibres)[: int(n_drop)]
            drop_mask = np.zeros(len(curve), dtype=bool)
            for c in drop_centers:
                segment_length = max(
                    opts.drop_segment_min_length,
                    int(np.random.normal(opts.drop_segment_avg_length, opts.drop_segment_std_length) // 2),
                )
                start = max(0, c - segment_length)
                end = min(len(curve), c + segment_length)
                drop_mask[start:end] = True

            # Discard dropped skeleton points
            if not drop_mask.any():
                continue
            if drop_mask.all():
                # print(f"Dropped entire branch {b.id}")
                tree.delete_branch(b.id, inplace=True)
                continue
            splits = np.where(np.diff(drop_mask.astype(np.uint8)) != 0)[0].astype(np.int32)
            if splits[0] == 0:
                splits = splits[1:]
                if not len(splits):
                    continue
            if splits[-1] == len(drop_mask) - 1:
                splits = splits[:-1]
                if not len(splits):
                    continue
            _, new_branches = tree.split_branch(b.id, splits, return_branch_ids=True, inplace=True)
            tree.delete_branch(new_branches[::2] if not drop_mask[0] else new_branches[1::2], inplace=True)
            # print(f"Dropped {len(new_branches) // 2} segments of size {drop_mask.sum()} from branch {b.id}")

    return tree
