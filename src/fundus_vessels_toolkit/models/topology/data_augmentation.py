from __future__ import annotations

from typing import Any, Literal, Optional, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from fundus_toolkits.utils.geometric import Rect

from ...segment_to_graph.graph_simplification import remove_orphan_nodes, simplify_passing_nodes
from ...segment_to_graph.vbranch_digraph import VBranchDigraph
from fundus_toolkits.transform import (
    AffineTransform,
    ElasticTransform,
    FlipTransform,
    Transform,
    IdentityTransform,
    TransformComposition,
)
from ...vascular_data_objects import VBranchGeoData, VGraph, VGraphBranch, VTree


class DeteriorationOpts(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    min_holes_count: int = Field(default=50)
    """Minimum number of holes to create disconnections"""

    max_holes_count: int = Field(default=70)
    """Maximum number of holes to create disconnections"""

    w_branch_base: float = Field(default=8.0)
    """Base weight for each branch"""

    w_branch_inv_calibre_f: float = Field(default=1.5)
    """Weighting branch calibre"""

    w_branch_sqrt_length_f: float = Field(default=2.0)
    """Weighting branch length"""

    max_w_spread: float = Field(default=0.5)
    """Scale weighting so the min is max_w_spread * max"""
    whole_branch_p: float = Field(default=0.1)
    """Probability to drop an entire branch"""

    whole_branch_max_calibre: float = Field(default=10.0)
    """Maximum average calibre to consider dropping entire branch"""

    whole_branch_max_length: int = Field(default=50)
    """Maximum length to consider dropping entire branch"""

    tip_hole_p: float = Field(default=0.4)
    """Probability to drop an endpoint branch"""

    hole_avg_length: int = Field(default=10)
    """Average length of dropped segments (sampled from normal distribution)"""

    hole_avg_length_f: float = Field(default=0.2)
    """Factor of the branch length added to the average length of dropped segments"""

    hole_std_length: int = Field(default=20)
    """Standard deviation of the length of dropped segments"""

    hole_min_length: int = Field(default=5)
    """Minimum length of dropped segments"""

    hole_min_length_f: float = Field(default=0.1)
    """Factor of the branch length added to the minimum length of dropped segments"""

    segment_min_length: int = Field(default=5)
    """Minimum length of left segments"""

    segment_min_length_f: float = Field(default=0.2)
    """Factor of the branch length added to the minimum length left segments"""


class ElasticOpts(BaseModel):
    """
    Options for elastic deformation.

    - displacement_std: Standard deviation of the displacement in pixels.
    - smoothing_size: Size of the Gaussian kernel for smoothing the displacement field.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    displacement_std: float = Field(default=80.0)
    """Standard deviation of the displacement in pixels"""

    smoothing_size: float = Field(default=200.0)
    """Size of the Gaussian kernel for smoothing the displacement field"""

    def generate_projection(
        self, shape: tuple[int, int], rng: Optional[np.random.Generator] = None
    ) -> ElasticTransform:
        if rng is None:
            rng = np.random.default_rng()
        return ElasticTransform.random(
            shape, displacement_std=self.displacement_std, smoothing_size=self.smoothing_size
        )


class RotationOpts(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    min_angle: float = Field(default=3.0)
    """Minimum absolute angle in degrees to apply rotation"""

    max_angle: float = Field(default=30.0)
    """Maximum absolute angle in degrees to apply rotation"""

    def generate_projection(self, shape: tuple[int, int], rng: Optional[np.random.Generator] = None) -> AffineTransform:
        if rng is None:
            rng = np.random.default_rng()
        center = shape[0] // 2, shape[1] // 2
        angle = rng.uniform(self.min_angle, self.max_angle)
        if rng.random() < 0.5:
            angle = -angle
        return AffineTransform.rotate(angle, center)


class AugmentationOpts(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True)

    elastic: bool | ElasticOpts = Field(default=True)
    """Whether to apply elastic deformation"""

    rotate: bool | RotationOpts = Field(default=False)
    """Whether to apply rotation"""

    horizontal_flip: bool = Field(default=True)
    """Whether to apply horizontal flip"""

    deteriorate_graph: DeteriorationOpts | bool = Field(default=True)
    """Whether to apply topological deterioration to the graph"""

    @property
    def geometric(self) -> bool:
        return bool(self.horizontal_flip or self.rotate is not False or self.elastic is not False)

    @property
    def rotation_opts(self) -> RotationOpts:
        return self.rotate if isinstance(self.rotate, RotationOpts) else RotationOpts()

    @property
    def elastic_opts(self) -> ElasticOpts:
        return self.elastic if isinstance(self.elastic, ElasticOpts) else ElasticOpts()

    @property
    def deterioration_opts(self) -> DeteriorationOpts:
        return self.deteriorate_graph if isinstance(self.deteriorate_graph, DeteriorationOpts) else DeteriorationOpts()

    def generate_transform(self, shape: tuple[int, int], rng: Optional[np.random.Generator] = None) -> Transform:
        if rng is None:
            rng = np.random.default_rng()
        center = shape[0] // 2, shape[1] // 2

        transforms: list[Transform] = []
        if self.horizontal_flip and rng.random() < 0.5:
            transforms.append(FlipTransform(center=center, horizontal=True))
        if self.rotate:
            transforms.append(self.rotation_opts.generate_projection(shape, rng=rng))
        if self.elastic:
            transforms.append(self.elastic_opts.generate_projection(shape, rng=rng))
        if len(transforms) == 0:
            return IdentityTransform()
        return TransformComposition(*transforms)

    @classmethod
    def parse(cls, data: Self | bool) -> Self:
        if data is True:
            return cls()
        elif data is False:
            return cls(elastic=False, rotate=False, horizontal_flip=False, deteriorate_graph=False)
        else:
            return data


def deteriorate_trees(trees: tuple[VTree, VTree], opts: Optional[DeteriorationOpts] = None) -> tuple[VTree, VTree]:
    if opts is None:
        opts = DeteriorationOpts()

    # === AV SWAP ===

    # === DISCONNECTIONS ===
    return deteriorate_graph(trees[0], opts), deteriorate_graph(trees[1], opts)


def deteriorate_graph[T: VGraph](
    graph: T, opts: Optional[DeteriorationOpts] = None, *, rng=None, debug_info: Optional[dict[str, Any]] = None
) -> T:
    if opts is None:
        opts = DeteriorationOpts()
    if rng is None:
        rng = np.random.default_rng()
    graph = graph.copy()
    geo = graph.geometric_data()
    B = graph.branch_count

    # === Fetch branch calibre and length  ===
    branch_curves = geo.branch_curve()
    branch_length = np.array([len(c) for c in branch_curves], dtype=int)

    def fetch_calibre(b: VGraphBranch) -> float:
        if branch_length[b.id] == 0:
            return 0
        cal = b.geodata(VBranchGeoData.Fields.CALIBRES)
        return 0.0 if cal is None or len(cal.data) == 0 else cal.data.mean()

    branch_calibre = np.array([fetch_calibre(b) for b in graph.branches()])

    # === Sample holes in branches  ===
    branch_weights = np.full((B,), opts.whole_branch_p)
    branch_weights += np.sqrt(branch_length) * opts.w_branch_sqrt_length_f
    branch_weights -= opts.w_branch_inv_calibre_f * branch_calibre
    branch_weights = np.maximum(branch_weights, 0.0)
    branch_weights -= branch_weights.min()
    branch_weights /= branch_weights.max()
    branch_weights = opts.max_w_spread + (1 - opts.max_w_spread) * branch_weights
    branch_weights /= branch_weights.sum()

    if debug_info is not None:
        debug_info["branch_weights"] = branch_weights
        debug_info["branch_length"] = branch_length
        debug_info["branch_calibre"] = branch_calibre

    n_holes = rng.integers(opts.min_holes_count, opts.max_holes_count + 1)
    hole_branches = rng.choice(a=len(branch_weights), size=n_holes, replace=True, p=branch_weights)

    branch_ids, n_holes_per_branch = np.unique(hole_branches, return_counts=True)
    branches = graph.branches(branch_ids, dynamic_iterator=True)

    def rng_hole_length(b_length: int) -> int:
        return max(
            opts.hole_min_length + int(opts.segment_min_length_f * b_length),
            int(round(rng.normal(opts.hole_avg_length + opts.hole_avg_length_f * b_length, opts.hole_std_length))),
        )

    # === Riddle branches with holes ===
    for b_old_idx, b, n_holes in zip(branch_ids, branches, n_holes_per_branch, strict=True):
        b_length = int(branch_length[b_old_idx])
        min_seg_len = int(opts.segment_min_length + opts.segment_min_length_f * b_length)
        b_curve = branch_curves[b_old_idx]

        if debug_info is not None:
            debug_info.setdefault("hole_info", {})[int(b_old_idx)] = {
                "branch_length": b_length,
                "branch_calibre": round(float(branch_calibre[b_old_idx]), 1),
                "n_holes": int(n_holes),
                "min_segment_length": min_seg_len,
                "avg_hole_length": opts.hole_avg_length + opts.hole_avg_length_f * b_length,
                "holes": [],
            }

        assert b_length == len(b.curve()), "Branch length mismatch."

        # --- 1. Check whether to drop the entire branch ---
        if b_length <= min_seg_len or (
            branch_calibre[b_old_idx] < opts.whole_branch_max_calibre
            and b_length < opts.whole_branch_max_length
            and rng.random() < opts.whole_branch_p
        ):
            # print(f"Dropping entire branch {b.id} (length={b_length}, calibre={branch_calibre[b_old_idx]:.2f})")
            graph.delete_branch(b.id, inplace=True)
            continue

        segments: list[tuple[int, int]] = [(0, int(b_length))]

        # --- 2. Check whether to drop branch tips ---
        hole_on_tips = [False, False]
        for _ in range(min(2, n_holes)):
            if rng.random() >= opts.tip_hole_p:
                break
            last_tip = 0 if rng.random() < 0.5 else 1
            if hole_on_tips[last_tip]:
                last_tip = 1 - last_tip
            hole_on_tips[last_tip] = True

            seg_start, seg_end = segments[0]
            hole_length = min(seg_end - seg_start - min_seg_len, rng_hole_length(b_length))
            if last_tip == 0:
                segments[0] = (hole_length, seg_end)
            else:
                segments[0] = (seg_start, seg_end - hole_length)
            n_holes -= 1

            if debug_info is not None:
                if last_tip == 0:
                    debug_info["hole_info"][b_old_idx]["holes"].append(f"0 - {hole_length}")
                else:
                    debug_info["hole_info"][b_old_idx]["holes"].append(
                        f"{b_length - hole_length} - {b_length} - ({hole_length}px)"
                    )

        # --- 3. Sample remaining holes in the middle of the branch ---
        fail_safe_counter = 20
        while n_holes > 0 and fail_safe_counter > 0:
            seg_idx = rng.integers(0, len(segments))
            seg_start, seg_end = segments[seg_idx]
            hole_length = rng_hole_length(b_length)

            if hole_length >= seg_end - seg_start - 2 * min_seg_len:
                fail_safe_counter -= 1
                continue

            half_hole1 = hole_length // 2
            half_hole2 = hole_length - half_hole1

            hole_center = int(rng.integers(seg_start + min_seg_len + half_hole1, seg_end - half_hole2 - min_seg_len))
            hole_start, hole_end = hole_center - half_hole1, hole_center + half_hole2

            segments[seg_idx] = (seg_start, hole_start)
            segments.insert(seg_idx + 1, (hole_end, seg_end))

            n_holes -= 1

            if debug_info is not None:
                debug_info["hole_info"][b_old_idx]["holes"].append(f"{hole_start} - {hole_end} ({hole_length}px)")

        if debug_info is not None:
            debug_info["hole_info"][b_old_idx]["segments"] = [f"{s}-{e} ({e - s}px)" for s, e in segments]

        # --- 4. Split branch according to segments ---
        splits: list[int] = [idx for seg in segments for idx in seg]  # type: ignore
        b_start, b_end = splits.pop(0), splits.pop(-1)

        # Update start and end of the branch curve
        if b_start != 0:
            new_node_id = graph.add_nodes(b_curve[b_start], inplace=True)
            graph.branch_list[b.id][0] = new_node_id[0]
        if b_end != b_length:
            new_node_id = graph.add_nodes(b_curve[b_end - 1], inplace=True)
            graph.branch_list[b.id][1] = new_node_id[0]
        if b_start != 0 or b_end != b_length:
            geo.resample_branch_curve(b.id, np.arange(b_start, b_end, dtype=np.int_))

        if len(splits) == 0:
            continue

        # Splits the branch curve
        splits_ = np.array(splits, dtype=np.int_) - b_start
        _, new_branches = graph.split_branch(b.id, splits_, return_branch_ids=True, inplace=True)

        # Discard new branches corresponding to holes
        # print(f"Deleting segments {new_branches[1::2]} of branch {b.id}")
        graph.delete_branch(new_branches[1::2], inplace=True)

    simplify_passing_nodes(graph, min_angle=90, inplace=True)
    remove_orphan_nodes(graph, inplace=True)
    return graph


def geometric_augment(
    sample: tuple[VBranchDigraph, npt.NDArray, npt.NDArray, npt.NDArray],
    *,
    opts: Optional[AugmentationOpts | Literal[True]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[VBranchDigraph, npt.NDArray, npt.NDArray, npt.NDArray]:
    digraph, fundus_img, od_yx, mac_yx = sample
    assert digraph.graph is not None, "Graph must be initialized to apply geometric augmentations"

    opts = AugmentationOpts() if opts in (True, None) else opts
    if rng is None:
        rng = np.random.default_rng()

    center = fundus_img.shape[0] // 2, fundus_img.shape[1] // 2
    shape = fundus_img.shape[0], fundus_img.shape[1]

    fundus_img = fundus_img.transpose(1, 2, 0)  # C,H,W -> H,W,C

    agg_transform = opts.generate_transform(shape, rng=rng)
    digraph.graph.transform(agg_transform, inplace=True)
    od_yx, mac_yx = agg_transform.transform(np.array([od_yx, mac_yx]))
    fundus_img = agg_transform.warp(fundus_img, warped_domain="same")[0]

    digraph.graph.geometric_data()._domain = Rect.from_size(shape)  # Reset domain after augmentation
    fundus_img = fundus_img.transpose(2, 0, 1)  # H,W,C -> C,H,W

    return digraph, fundus_img, od_yx, mac_yx
