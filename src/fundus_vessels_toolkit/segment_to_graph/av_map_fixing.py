from typing import Any, Dict, List, Literal, Optional, Self, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt
from skimage.segmentation import expand_labels

from fundus_toolkits.utils.geometric import Rect
from fundus_vessels_toolkit.utils.lookup_array import invert_complete_lookup
from fundus_vessels_toolkit.vascular_data_objects.vgraph import VGraph

from ..utils.rasterization import rasterize_topology
from ..vascular_data_objects.vbranch_geodata import VBranchGeoData
from ..vascular_data_objects.vtree import VTree, VTreeBranch


def rasterize_tree_topology(
    tree: VTree,
    *,
    topological_labels: bool = True,
    expand_labels_by: int = 0,
    bezier_interpolate: float = 0.5,
    fill_junctions: bool = True,
    geodata_id=0,
    boundaries_field: VBranchGeoData.Key = VBranchGeoData.Fields.BOUNDARIES,
) -> Tuple[npt.NDArray[np.uint64], npt.NDArray[np.float32]]:
    """
    Rasterize the given vessel tree into a binary mask and a distance map.

    Parameters
    ----------
    tree : VTree
        The vessel tree to rasterize.

    topological_labels : bool, optional
        If True, each branch will be assigned a label based on its position in the topology:
        The first 12 bytes encode the index of its subtree, the next 16 bytes encode its position in the

    geodata_id : int, optional
        The ID of the geometric data to use for rasterization. Default is 0.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the binary mask and the distance map.
    """
    geodata = tree.geometric_data(geodata_id)
    labels_map, topo_map = rasterize_topology(
        branch_list=tree.branch_list,
        root_branches=tree.root_nodes_ids(),
        curves=geodata.branch_curve(),
        boundaries=[
            _.data if _ is not None else np.empty((0, 2, 2), dtype=np.int_)
            for _ in geodata.branch_data(boundaries_field)
        ],
        shape=geodata.domain.shape,
        node_count=tree.node_count,
        bezier_interpolate=bezier_interpolate,
        fill_junctions=fill_junctions,
    )

    if expand_labels_by > 0:
        labels_map = expand_labels(labels_map, distance=expand_labels_by)
        topo_map = expand_labels(topo_map, distance=expand_labels_by)

    if topological_labels:
        topo_labels = branch_topological_mapping(tree)
        labels_map = topo_labels[labels_map]

    return labels_map, topo_map


def fix_av_map(
    av_map: npt.NDArray[np.uint8],
    trees: Tuple[VTree, VTree],
    expand_labels_by: int = 1,
    draw_reconnections: bool = True,
    discard_av: bool = False,
) -> npt.NDArray[np.uint8]:
    """
    Fix the AV classification to match the given trees. The vessel segmentation is not modified, only the AV classification.

    Parameters
    ----------
    av_map : npt.NDArray[np.uint8]
        The AV map to fix.

    trees : Tuple[VTree, VTree]
        The arterioles and venules trees.

    expand_labels_by : int, optional
        The number of pixels to expand the labels by. Default is 2.

    Returns
    -------
    npt.NDArray[np.float32]
        The fixed AV map.
    """  # noqa: E501
    kwargs: Dict[str, Any] = dict(bridge_gap_smaller_than=25, fill_junctions=True)
    a_map = rasterize_tree_topology(trees[0], **kwargs)[0] > 0
    v_map = rasterize_tree_topology(trees[1], **kwargs)[0] > 0

    if expand_labels_by > 0:
        from skimage.morphology import binary_dilation, disk

        a_map = binary_dilation(a_map, disk(expand_labels_by))
        v_map = binary_dilation(v_map, disk(expand_labels_by))

    seg_mask = av_map == 0
    a_map[seg_mask] = False
    v_map[seg_mask] = False

    if draw_reconnections:
        draw_missing_connections(trees[0], out=a_map, fill_value=True)
        draw_missing_connections(trees[1], out=v_map, fill_value=True)

    tree_av_map = a_map.astype(np.uint8) + 2 * v_map.astype(np.uint8)

    if not discard_av:
        mask = tree_av_map == 0
        tree_av_map[mask] = av_map[mask]

    return tree_av_map


def draw_missing_connections(graph: VGraph, out: npt.NDArray, fill_value: int = 1):
    """
    Draw missing connections in the graph by filling in the gaps in the out array.

    Parameters
    ----------
    graph : VGraph
        The vessel graph to draw missing connections for.
    out : npt.NDArray
        The output array to fill with missing connections.
    fill_value :
        The value to fill in the gaps. Default is 1.
    """
    for branch in graph.branches():
        mean_calibre = None
        curve = branch.curve()
        calibres = c.data if (c := branch.geodata(VBranchGeoData.Fields.CALIBRES)) is not None else None
        n1, n2 = [node.coord() for node in branch.nodes()]
        for bezier in branch.bspline().filling_curves(n1, n2, smoothing=0.5):
            if bezier[0] == bezier[-1]:
                continue

            mid_points = bezier.evaluate(np.linspace(0, 1, min(10, int(bezier.arc_length(fast_approximation=True)))))
            mid_points = np.round(mid_points).astype(int)
            if (
                np.any(mid_points < 0)
                or np.any(mid_points >= np.array(out.shape))
                or np.all(out[mid_points[:, 0], mid_points[:, 1]] != 0)
            ):
                continue

            mean_calibre = 2.0
            if calibres is not None:
                tip_calibres = []
                if bezier[0] != n1:
                    p0 = np.all(curve == bezier[0], axis=1)
                    if p0.any():
                        tip_calibres.append(calibres[np.argmax(p0)])
                if bezier[-1] != n2:
                    p1 = np.all(curve == bezier[-1], axis=1)
                    if p1.any():
                        tip_calibres.append(calibres[np.argmax(p1)])
                mean_calibre = max(2.0, float(0.75 * np.mean(tip_calibres))) if len(tip_calibres) > 0 else 2.0
            bezier.rasterize(out, width=mean_calibre, fill_value=fill_value)


def evaluate_topology(
    tree: VTree, topo_labels: npt.NDArray[np.uint64], topo_map: npt.NDArray[np.float32], *, epsilon=1e-5
) -> dict[str, npt.NDArray]:
    """
    Evaluate the topology of the vessel tree against the topological labels.

    Parameters
    ----------
    tree : VTree
        The vessel tree to evaluate.
    topo_labels : npt.NDArray[np.uint64]
        The topological labels of the branches.
    topo_map : npt.NDArray[np.float32]
        The topological map of the vessel tree.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the evaluation metrics.
    """
    tree = tree.flip_branch_to_tree_dir()
    branches = np.arange(tree.branch_count)

    nodes_yx = tree.geometric_data().node_coord()
    B = tree.branch_count

    # === UNKNOWN BRANCHES AND BRANCH DIRECTIONS ===
    unknown_branches = np.zeros_like(branches, dtype=np.bool_)
    branches_dir = np.zeros_like(branches, dtype=np.float32)
    domain = Rect.from_size(topo_map.shape).exclude_bottom_right_edges()  # type: ignore

    tips_coord = domain.clip(nodes_yx[tree.branch_list.flatten()]).reshape(B, 2, 2)  # [B, (tail, head), (y,x)]

    for b in tree.branches():
        curve = b.curve()
        if curve is None:
            t1, t2 = topo_map[*tips_coord[b.id].T]
            if t1 == 0 or t2 == 0:
                unknown_branches[b.id] = True
            elif t2 - t1 > epsilon:
                branches_dir[b.id] = 1
            elif t1 - t2 > epsilon:
                branches_dir[b.id] = -1
            continue
        else:
            curve = curve[domain.contains(curve)]
            topo_values = topo_map[*curve.T]
            null_topo = topo_values == 0
            topo_values = topo_values[~null_topo]
            if np.mean(null_topo) > 2 / 3 or len(topo_values) < 2:
                unknown_branches[b.id] = True
                continue

            diff = np.diff(topo_values)
            forward_diff = diff > epsilon
            backward_diff = diff < -epsilon
            branches_dir[b.id] = np.mean(1 * forward_diff - 1 * backward_diff)

    # === BRANCH BEST PARENT ===
    heads_yx = tips_coord[np.arange(B), (branches_dir >= 0).astype(np.int32)].astype(np.int32)
    heads_label, heads_rank = topo_labels[*heads_yx.T], topo_map[*heads_yx.T]

    tails_yx = tips_coord[np.arange(B), (branches_dir < 0).astype(np.int32)].astype(np.int32)
    tails_label, tails_rank = topo_labels[*tails_yx.T], topo_map[*tails_yx.T]

    best_parent = np.full(tree.branch_count, -1, dtype=np.int32)
    for b_id in branches[~unknown_branches]:
        b = tree.branch(b_id)
        tail_label = TopologicalLabel(tails_label[b_id])
        tail_rank = tails_rank[b_id]

        ancestors = (tail_label == heads_label) | tail_label.is_child_of(heads_label)
        ancestors[b_id] = False  # A branch cannot be its own parent
        ancestors[heads_rank > tail_rank] = False  # Parent must have a lower rank
        ancestors = np.where(ancestors)[0]

        if len(ancestors) == 0:
            continue

        best_ancestors = ancestors[np.argsort(heads_rank[ancestors])[::-1]]  # Prefer the highest rank
        best_parent[b_id] = best_ancestors[0]

    return dict(
        missing=np.array(unknown_branches, dtype=np.int32),
        gt_dir=branches_dir,
        gt_parent=best_parent,
        parent=tree.branch_tree,
    )


def count_disconnection(graph, topological_labels: npt.NDArray[np.uint64]) -> int:
    """
    Count the number of disconnected branches in the graph based on the topological map.

    Parameters
    ----------
    graph : VGraph
        The vessel graph to analyze.
    topological_map : npt.NDArray[np.uint64]
        The topological map of the vessel tree.

    Returns
    -------
    int
        The number of disconnected branches.
    """
    disconnected_count = 0
    for branch in graph.branches():
        branch_label = TopologicalLabel.encode(
            subtree=branch.subtree_index(),
            branching_pattern=branch.branching_pattern(),
        )
        curve = branch.curve()
        rasterized_labels = topological_labels[
            np.clip(curve[:, 0], 0, topological_labels.shape[0] - 1),
            np.clip(curve[:, 1], 0, topological_labels.shape[1] - 1),
        ]
        if not np.any(rasterized_labels == branch_label):
            disconnected_count += 1
    return disconnected_count


def evaluate_branch_direction(tree: VTree, topological_map: npt.NDArray[np.float32], epsilon=1e-5) -> npt.NDArray:
    """

    Parameters
    ----------
    tree : _type_
        _description_
    topological_map : npt.NDArray[np.uint64]
        _description_

    Returns
    -------
    npt.NDArray
        _description_
    """
    direction = np.zeros((tree.branch_count,), dtype=np.float32)

    for b in tree.flip_branch_to_tree_dir().branches():
        curve = b.curve()
        if curve.shape[0] < 2:
            continue
        topo_values = topological_map[curve[:, 0], curve[:, 1]]

        # Discard zero values
        topo_values = topo_values[topo_values != 0]

        if len(topo_values) < 2:
            continue

        # Compute the direction as the mean of the topological values
        diff = np.diff(topo_values)
        forward_diff = diff > epsilon
        backward_diff = diff < -epsilon
        direction[b.id] = np.mean(1 * forward_diff - 1 * backward_diff)

    return direction


class TopologicalLabel(np.uint64):
    """
    A class representing a topological label for a branch in the vessel tree, coded on a 64-bit integer.

    The bits are used as follows:
    - 0-11  : subtree index (12 bits, max: 4096)
    - 12-19 : branching rank (8 bits)
    - 20-63 : binary branching pattern (44 bits)

    Examples
    --------
    >>> label = TopologicalLabel.encode(subtree=7, rank=3, branching_pattern=[True, False, True])
    >>> hex(label)
    '0x70300000000005'
    >>> label.subtree, label.rank, label.branching_pattern
    (7, 3, array([ True, False,  True]))

    """

    SUBTREE_MASK = np.uint64(0xFFF0000000000000)
    RANK_MASK = np.uint64(0x0000FF0000000000)
    BRANCHING_PATTERN_MASK = np.uint64(0x000000FFFFFFFFFF)

    @classmethod
    def encode(cls, subtree: int | np.int_, branching_pattern: Sequence[bool]) -> Self:
        """
        Create a TopologicalLabel from an integer label.
        """
        assert 0 <= subtree < 4095, "Subtree index must be between 0 and 4094."
        assert len(branching_pattern) <= 44, "Branching pattern must be less than 44 bits."
        rank = len(branching_pattern)
        branching_pattern_int = sum((1 << i) for i, b in enumerate(branching_pattern) if b)
        subtree = int(subtree + 1) & 0xFFF  # Ensure subtree is within 12 bits
        rank = int(rank) & 0xFF  # Ensure rank is within 8 bits
        branching_pattern_int = int(branching_pattern_int) & 0xFFFFFFFFFFF  # Ensure pattern is within 44 bits
        return cls((subtree << 52) | (rank << 44) | branching_pattern_int)

    @classmethod
    def decode(cls, map: npt.NDArray[np.uint64]) -> Tuple[npt.NDArray[np.uint32], List[Self]]:
        """
        Decode a topological label map into its components.
        """
        labels, labels_map = np.unique(map, return_inverse=True)
        return labels_map.astype(np.uint32), [cls(label) for label in labels]

    @classmethod
    @overload
    def decode_subtree(cls, label: Self | np.uint64) -> np.uint64: ...
    @classmethod
    @overload
    def decode_subtree(cls, label: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]: ...
    @classmethod
    def decode_subtree(cls, label: Self | np.uint64 | npt.NDArray[np.uint64]) -> np.uint64 | npt.NDArray[np.uint64]:
        """
        Decode the subtree indices from a topological label map.
        """
        return (label & cls.SUBTREE_MASK) >> np.uint64(52)

    @classmethod
    @overload
    def decode_rank(cls, label: Self | np.uint64) -> np.uint8: ...
    @classmethod
    @overload
    def decode_rank(cls, label: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint8]: ...
    @classmethod
    def decode_rank(cls, label: Self | np.uint64 | npt.NDArray[np.uint64]) -> np.uint8 | npt.NDArray[np.uint8]:
        """
        Decode the branching ranks from a topological label map.
        """
        return ((label & cls.RANK_MASK) >> np.uint64(44)).astype(np.uint8)

    @classmethod
    @overload
    def decode_branching_pattern(cls, label: Self | np.uint64, *, max_rank: Optional[int] = None) -> np.uint64: ...
    @classmethod
    @overload
    def decode_branching_pattern(
        cls, label: npt.NDArray[np.uint64], *, max_rank: Optional[int] = None
    ) -> npt.NDArray[np.uint64]: ...
    @classmethod
    def decode_branching_pattern(
        cls, label: Self | np.uint64 | npt.NDArray[np.uint64], *, max_rank: Optional[int] = None
    ) -> np.uint64 | npt.NDArray[np.uint64]:
        """
        Decode the branching patterns from a topological label map.
        """
        if max_rank is None:
            max_rank = int(np.max(cls.decode_rank(label)))
        assert max_rank <= 44, "Max rank must be less than or equal to 44."
        pattern_mask = np.uint64(2**max_rank) - np.uint64(1)
        return (label & pattern_mask).astype(np.uint64)

    @classmethod
    def map_to_rgb(cls, map: npt.NDArray[np.uint64], *, encode_pattern: bool = True) -> npt.NDArray[np.uint8]:
        """Convert a topological label map to an RGB color map for visualisation purposes.

        Parameters
        ----------
        map : npt.NDArray[np.uint64]
            The input topological label map.

        encode_pattern : bool, optional
            Whether to encode the branching pattern in the color. Default is True.

        Returns
        -------
        npt.NDArray[np.uint8]
            The output RGB color map.
        """
        # Decode labels map
        labels_map, labels = cls.decode(map)
        colors = np.stack([label.color(format="rgb", encode_pattern=encode_pattern) for label in labels])
        return colors[labels_map.flatten()].reshape(map.shape + (3,)).transpose(2, 0, 1)

    @property
    def is_background(self) -> bool:
        return (self & self.SUBTREE_MASK) == 0

    @property
    def subtree(self) -> int:
        return (int(self & self.SUBTREE_MASK) >> 52) - 1

    @property
    def rank(self) -> int:
        return int(self & self.RANK_MASK) >> 44

    @property
    def branching_pattern(self) -> npt.NDArray[np.bool_]:
        return np.array([(self & np.uint64(1 << i)) != 0 for i in range(self.rank)], dtype=np.bool_)

    def is_same_subtree(self, other: Self | np.int32) -> bool:
        """
        Check if this label belongs to the same subtree as another label.
        """
        o = TopologicalLabel(other)
        return self.subtree == o.subtree

    @overload
    def is_parent_of(self, other: Self | np.uint64) -> bool: ...
    @overload
    def is_parent_of(self, other: npt.NDArray[np.uint64]) -> npt.NDArray[np.bool_]: ...
    def is_parent_of(self, other: Self | np.uint64 | npt.NDArray[np.uint64]) -> bool | npt.NDArray[np.bool_]:
        """
        Check if this label is a parent of another label.
        """
        if not isinstance(other, np.ndarray):
            o = TopologicalLabel(other)
            return (
                self.subtree == o.subtree
                and self.rank < o.rank
                and bool(np.all(self.branching_pattern == o.branching_pattern[: self.rank]))
            )
        else:
            o_subtrees = (other & self.SUBTREE_MASK >> np.int32(52)) - np.int32(1)
            o_ranks = (other & self.RANK_MASK) >> np.int32(44)
            pattern_mask = np.uint64(2**self.rank) - np.uint64(1)
            o_patterns = other & pattern_mask
            self_pattern = self & pattern_mask
            return (self.subtree == o_subtrees) & (self.rank < o_ranks) & (self_pattern == o_patterns)

    @overload
    def is_child_of(self, other: Self | np.uint64) -> bool: ...
    @overload
    def is_child_of(self, other: npt.NDArray[np.uint64]) -> npt.NDArray[np.bool_]: ...
    def is_child_of(self, other: Self | np.uint64 | npt.NDArray[np.uint64]) -> bool | npt.NDArray[np.bool_]:
        """
        Check if this label is a child of another label.
        """
        if not isinstance(other, np.ndarray):
            o = TopologicalLabel(other)
            return (
                self.subtree == o.subtree
                and self.rank > o.rank
                and bool(np.all(o.branching_pattern == self.branching_pattern[: o.rank]))
            )
        else:
            o_subtrees = self.decode_subtree(other)
            o_ranks = self.decode_rank(other)
            pattern_mask = np.uint64(2**o_ranks) - np.uint64(1)
            o_patterns = other & pattern_mask
            self_pattern = self & pattern_mask
            return (self.subtree == o_subtrees) & (self.rank > o_ranks) & (self_pattern == o_patterns)

    def common_ancestor(self, other: Self | np.int32) -> Self | None:
        """
        Find the common ancestor of this label and another label.
        """
        o = TopologicalLabel(other)
        if self.subtree != o.subtree:
            return None

        common_rank = min(self.rank, o.rank)
        common_pattern = self.branching_pattern[:common_rank] == o.branching_pattern[:common_rank]
        common_rank = np.argmin(np.concatenate([common_pattern, [False]]))
        common_branching_pattern = self.branching_pattern[:common_rank]

        return TopologicalLabel.encode(self.subtree, common_rank, common_branching_pattern)  # type: ignore[arg-type]

    @classmethod
    @overload
    def subtree_color(cls, subtree: int, format: Literal["hex"] = "hex") -> str: ...
    @classmethod
    @overload
    def subtree_color(cls, subtree: int, format: Literal["rgb"]) -> npt.NDArray[np.uint8]: ...
    @classmethod
    def subtree_color(cls, subtree: int, format: Literal["hex", "rgb"] = "hex") -> str | npt.NDArray[np.uint8]:
        """Assign a base color to each subtree."""
        if subtree == 0:
            return "#000000" if format == "hex" else np.zeros(3, dtype=np.uint8)

        catppuccin_latte = [
            "#1e66f5",  # Blue
            "#04a5e5",  # Sky
            "#40a02b",  # Green
            "#fe640b",  # Peach
            "#d20f39",  # Red
            "#ea76cb",  # Pink
            "#dc8a78",  # Rosewater
            "#7287fd",  # Lavender
            "#209fb5",  # Sapphire
            "#179299",  # Teal
            "#df8e1d",  # Yellow
            "#e64553",  # Maroon
            "#8839ef",  # Mauve
            "#dd7878",  # Flamingo
        ]
        color = catppuccin_latte[subtree % len(catppuccin_latte)]  # Cycle through the colors

        if format == "hex":
            return color

        from coloraide import Color

        return (np.array(Color(color).convert("srgb").coords()) * 255).astype(np.uint8)

    @overload
    def color(self, format: Literal["hex"] = "hex", *, encode_pattern: bool = True) -> str: ...
    @overload
    def color(self, format: Literal["rgb"], *, encode_pattern: bool = True) -> npt.NDArray[np.uint8]: ...
    def color(
        self, format: Literal["hex", "rgb"] = "hex", *, encode_pattern: bool = True
    ) -> str | npt.NDArray[np.uint8]:
        """Assign a color to this label base on its subtree and its branching pattern.
        Starting from the subtree color, the saturation increase for each primary branching and decrease for each secondary branching. The lightness increase at each level of the tree.

        Parameters
        ----------
        format : Literal["hex", "rgb"], optional
            The format of the output color. Can be "hex" or "rgb". Default is "hex".
        encode_pattern : bool, optional
            Whether to encode the branching pattern in the color. Default is True.
        """  # noqa: E501
        from coloraide import Color

        if self.is_background:
            return "#000000" if format == "hex" else np.zeros(3, dtype=np.uint8)

        hsv_color = Color(self.subtree_color(self.subtree, format="hex")).convert("hsv")
        if encode_pattern:
            hsv_color[1] = 0.95**self.rank
            hsv_color[2] = sum(
                [0.5] + [0.5 ** (i + 2) * (1 if b else -1) for i, b in enumerate(self.branching_pattern)]
            )
            hsv_color[2] = hsv_color[2] * 0.9 + 0.1

        color = hsv_color.convert("srgb")
        if format == "hex":
            return color.to_string(hex=True)
        return (np.array(color.coords()) * 255).astype(np.uint8)


def branch_topological_mapping(tree: VTree) -> npt.NDArray[np.uint64]:
    """
    Compute the topological mapping of the branches in the vessel tree.

    Parameters
    ----------
    tree : VTree
        The vessel tree to compute the topological mapping for.

    Returns
    -------
    npt.NDArray[np.int32]
        An array of topological labels for each branch in the tree.
    """
    from ..vparameters.bifurcations import reorder_branch_by_bifurcations

    working_tree = tree.copy()
    lookup = reorder_branch_by_bifurcations(working_tree)
    lookup = invert_complete_lookup(lookup) + 1

    labels = np.zeros(tree.branch_count + 1, dtype=np.uint64)

    def recursive_labeling(branch: VTreeBranch, subtree: int, branching_pattern: list[bool]) -> None:
        nonlocal labels

        labels[lookup[branch.id]] = TopologicalLabel.encode(subtree, branching_pattern)

        for i, successor_id in enumerate(sorted(branch.successors_ids)):
            if i != branch.n_successors - 1:
                branching_pattern = branching_pattern + [True]
            recursive_labeling(working_tree.branch(successor_id), subtree, branching_pattern)
            branching_pattern = branching_pattern[:-1] + [False]

    for subtree, branch in enumerate(working_tree.root_branches()):
        recursive_labeling(branch, subtree, [])

    return labels
