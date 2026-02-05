from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from fundus_toolkits.utils.geometric import Rect

from ...utils.fundus_projections import AffineProjection, ElasticProjection, FlipProjection
from ...utils.math import sigmoid
from ...vascular_data_objects import VBranchGeoData, VGraph, VTree
from ..graph_simplification import simplify_passing_nodes
from ..vbranch_digraph import VBranchDigraph


@dataclass
class DeteriorationOpts:
    drop_p: float = 0.05  # Probability to drop each skeleton point
    drop_calibre_threshold: float = 10  # Maximum calibre to drop skeleton points
    drop_calibre_smooth: float = 2.0  # Spread of the sigmoid to drop skeleton points
    drop_segment_min_length: int = 8  # Minimum size of a dropped segment
    drop_segment_avg_length: int = 16  # Minimum size of a dropped segment
    drop_segment_std_length: int = 8  # Standard deviation of the size of a dropped segment
    av_swap_p: float = 0.01  # Probability to swap the artery/vein label of each branch
    av_swap_min_segment_length: int = 10  # Minimum size of a AV swapped segment


def deteriorate_trees(trees: tuple[VTree, VTree], opts: Optional[DeteriorationOpts] = None) -> tuple[VTree, VTree]:
    if opts is None:
        opts = DeteriorationOpts()

    # === AV SWAP ===

    # === DISCONNECTIONS ===
    return deteriorate_graph(trees[0], opts), deteriorate_graph(trees[1], opts)


def deteriorate_graph[T: VGraph](graph: T, opts: Optional[DeteriorationOpts] = None) -> T:
    if opts is None:
        opts = DeteriorationOpts()
    graph = graph.copy()

    for b in graph.branches(dynamic_iterator=True):
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
                graph.delete_branch(b.id, inplace=True)
                continue
            splits = np.where(np.diff(drop_mask.astype(np.uint8)) != 0)[0].astype(np.int_)
            if splits[0] == 0:
                splits = splits[1:]
                if not len(splits):
                    continue
            if splits[-1] == len(drop_mask) - 1:
                splits = splits[:-1]
                if not len(splits):
                    continue
            _, new_branches = graph.split_branch(b.id, splits, return_branch_ids=True, inplace=True)
            graph.delete_branch(new_branches[::2] if not drop_mask[0] else new_branches[1::2], inplace=True)
            # print(f"Dropped {len(new_branches) // 2} segments of size {drop_mask.sum()} from branch {b.id}")
    simplify_passing_nodes(graph, min_angle=90, inplace=True)
    return graph


def geometric_augment(
    sample: tuple[VBranchDigraph, npt.NDArray],
    *,
    max_rotation: float = 30.0,
    min_rotation: float = 5.0,
    horizontal_flip: bool = True,
    rnd: Optional[np.random.Generator] = None,
) -> tuple[VBranchDigraph, npt.NDArray]:
    digraph, fundus_img = sample
    if rnd is None:
        rnd = np.random.default_rng()
    center = fundus_img.shape[1] // 2, fundus_img.shape[2] // 2

    # === Rotation ===
    # angle = rnd.uniform(-max_rotation, max_rotation)
    # if abs(angle) > min_rotation:
    #     rotate = AffineProjection.rotate(angle, center)
    #     digraph.graph.transform(rotate, inplace=True)
    #     fundus_img = rotate.warp(fundus_img.transpose(1, 2, 0), warped_domain="same")[0].transpose(2, 0, 1)

    # === Elastic ===
    elastic = ElasticProjection.random(fundus_img.shape[-2:], displacement_std=120, smoothing_size=200)
    digraph.graph.transform(elastic, inplace=True)
    fundus_img = elastic.warp(fundus_img.transpose(1, 2, 0), warped_domain="same")[0].transpose(2, 0, 1)

    # === Horizontal flip ===
    # if horizontal_flip and rnd.random() < 0.5:
    #     flip = FlipProjection(center, horizontal=True)
    #     digraph.graph.transform(flip, inplace=True)
    #     fundus_img = flip.warp(fundus_img.transpose(1, 2, 0), warped_domain="same")[0].transpose(2, 0, 1)

    # Reset domain after augmentation
    digraph.graph.geometric_data()._domain = Rect.from_size((fundus_img.shape[1], fundus_img.shape[2]))

    return digraph, fundus_img
