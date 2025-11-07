from typing import Any, Dict, List, Literal, Self, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt
from skimage.segmentation import expand_labels

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
    bridge_gap_smaller_than: float = 20,
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
        bridge_gap_smaller_than=bridge_gap_smaller_than,
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
    def map_to_rgb(cls, map: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint8]:
        """Convert a topological label map to an RGB color map for visualisation purposes.

        Parameters
        ----------
        map : npt.NDArray[np.uint64]
            The input topological label map.

        Returns
        -------
        npt.NDArray[np.uint8]
            The output RGB color map.
        """
        # Decode labels map
        labels_map, labels = cls.decode(map)
        colors = np.stack([label.color(format="rgb") for label in labels])
        return colors[labels_map.flatten()].reshape(map.shape + (3,))

    @classmethod
    def to_subtree(cls, label: Self | np.uint64 | npt.NDArray[np.uint64]) -> np.uint64 | npt.NDArray[np.uint64]:
        """
        Get the subtree index from a topological label.
        """
        return ((label >> np.uint64(52)) & np.uint64(0xFFF)) - np.uint64(1)

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

    def is_parent_of(self, other: Self | np.int32) -> bool:
        """
        Check if this label is a parent of another label.
        """
        o = TopologicalLabel(other)
        return (
            self.subtree == o.subtree
            and self.rank < o.rank
            and bool(np.all(self.branching_pattern == o.branching_pattern[: self.rank]))
        )

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

    def subtree_color(self) -> str:
        """Assign a base color to each subtree."""
        if self.is_background:
            return "#000000"

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
        return catppuccin_latte[self.subtree % len(catppuccin_latte)]  # Cycle through the colors

    @overload
    def color(self, format: Literal["hex"] = "hex") -> str: ...
    @overload
    def color(self, format: Literal["rgb"]) -> npt.NDArray[np.uint8]: ...
    def color(self, format: Literal["hex", "rgb"] = "hex") -> str | npt.NDArray[np.uint8]:
        """Assign a color to this label base on its subtree and its branching pattern.
        Starting from the subtree color, the saturation increase for each primary branching and decrease for each secondary branching. The lightness increase at each level of the tree.
        """  # noqa: E501
        from coloraide import Color

        if self.is_background:
            return "#000000" if format == "hex" else np.zeros(3, dtype=np.uint8)

        hsv_color = Color(self.subtree_color()).convert("hsv")
        hsv_color[1] = 0.95**self.rank
        hsv_color[2] = sum([0.5] + [0.5 ** (i + 2) * (1 if b else -1) for i, b in enumerate(self.branching_pattern)])
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
