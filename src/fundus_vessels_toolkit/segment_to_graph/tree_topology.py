from __future__ import annotations

from typing import List, Literal, Optional, Self, Sequence, Tuple, overload

import numpy as np
import numpy.typing as npt
from skimage.segmentation import expand_labels

from fundus_toolkits.utils.geometric import Rect
from fundus_vessels_toolkit.utils.cluster import reduce_clusters

from ..utils.lookup_array import invert_complete_lookup
from ..utils.math import gaussian_kernel2d
from ..utils.numpy import binary_sparse_conv2d, bit_invert
from ..utils.rasterization import rasterize_line, rasterize_topology
from ..utils.typing import Bool1DArray, Float1DArray, Float2DArray, Int1DArray
from ..vascular_data_objects.vbranch_geodata import VBranchGeoData
from ..vascular_data_objects.vgraph import VGraph
from ..vascular_data_objects.vtree import VTree, VTreeBranch


########################################################################################################################
#       === TreeTopology CLASS ===
########################################################################################################################
class TreeTopology:
    def __init__(
        self,
        branch_map: npt.NDArray[TopologicalLabel],
        rank_map: npt.NDArray[np.float32],
        fuzzy_skeleton_map: npt.NDArray[np.float32],
        tree: Optional[VTree] = None,
        branch_mapping: Optional[npt.NDArray[TopologicalLabel]] = None,
    ) -> None:
        """Store the tree topology information including the branch map and topological distance map.

        Parameters
        ----------
        branch_map : npt.NDArray[TopologicalLabel]
            A 2D array where each pixel is labeled with the topological label of the branch it belongs to.
        rank_map : npt.NDArray[np.float32]
            A 2D array where each pixel contains a continuous topological rank monotonously increasing from the tree root.
        tree : Optional[VTree], optional
            The vessel tree used to generate the topology, by default None
        branch_mapping : Optional[npt.NDArray[TopologicalLabel]], optional
            An array mapping rasterized branch labels to branch IDs in the tree, by default None
        """  # noqa: E501
        self.branch_map = branch_map.astype(TopologicalLabel)
        self.rank_map = rank_map
        self.fuzzy_skeleton_map = fuzzy_skeleton_map
        assert branch_map.shape == rank_map.shape, "Branch map and topo distance map must have the same shape."

        self._tree = tree
        self._branch_mapping = branch_mapping

    @classmethod
    def from_tree(
        cls,
        tree: VTree,
        *,
        expand_labels_by: int = 0,
        bezier_interpolate: float = 0.5,
        fill_junctions: bool = True,
        boundaries_field: VBranchGeoData.Key = VBranchGeoData.Fields.BOUNDARIES,
    ) -> Self:
        """
        Rasterize the given vessel tree into a binary mask and a distance map.

        Parameters
        ----------
        tree : VTree
            The vessel tree to rasterize.

        topological_labels : bool, optional
            If True, each branch will be assigned a label based on its position in the topology:
            The first 12 bytes encode the index of its subtree, the next 16 bytes encode its position in the

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the binary mask and the distance map.
        """
        geodata = tree.geometric_data()

        labels_map, topo_map = rasterize_topology(
            branch_list=tree.branch_list,
            branch_tree=tree.branch_tree,
            branch_dirs=tree.branch_dirs(),
            curves=geodata.branch_curve(),
            boundaries=[
                _.data if _ is not None else np.empty((0, 2, 2), dtype=np.int_)
                for _ in geodata.branch_data(boundaries_field)
            ],
            nodes_yx=geodata.node_coord(),
            shape=geodata.domain.shape,
            bezier_interpolate=bezier_interpolate,
            fill_junctions=fill_junctions,
        )

        if expand_labels_by > 0:
            labels_map = expand_labels(labels_map, distance=expand_labels_by)
            topo_map = expand_labels(topo_map, distance=expand_labels_by)

        skeleton = tree.geometric_data().skeleton_label_map(connect_nodes=True, interpolate=True) > 0
        gaussian_kernel = gaussian_kernel2d(expand_labels_by / 2 + 4)
        gaussian_kernel /= gaussian_kernel[gaussian_kernel.shape[0] // 2].sum()  # Normalize so lines sum to 1
        fuzzy_skeleton_map = binary_sparse_conv2d(skeleton, gaussian_kernel) * (labels_map > 0)

        branch_mapping = branch_topological_mapping(tree)
        labels_map = branch_mapping[labels_map]

        return cls(
            labels_map.astype(TopologicalLabel), topo_map, fuzzy_skeleton_map, tree=tree, branch_mapping=branch_mapping
        )

    def has_tree(self) -> bool:
        return self._tree is not None

    @property
    def tree(self) -> VTree:
        if self._tree is None:
            raise AttributeError("TreeTopology was not initialized with a tree.")
        return self._tree

    def has_branch_mapping(self) -> bool:
        return self._branch_mapping is not None

    @property
    def branch_mapping(self) -> npt.NDArray[TopologicalLabel]:
        if self._branch_mapping is None:
            raise AttributeError("TreeTopology was not initialized with a branch mapping.")
        return self._branch_mapping

    @property
    def shape(self) -> Tuple[int, int]:
        return self.branch_map.shape  # type: ignore


########################################################################################################################
#       === TOPOLOGICAL METRICS UTILS ===
########################################################################################################################
def read_branch_topology(
    graph: VGraph,
    topology: TreeTopology,
    *,
    min_rank_threshold: float = 0.1,
    max_rank_tolerance: float = 0.25,
) -> tuple[npt.NDArray[TopologicalLabel], Float1DArray, Float1DArray, npt.NDArray[TopologicalLabel], Float2DArray]:
    """
    Read the topological labels and distances for each branch in the graph based on the given tree topology.

    Parameters
    ----------
    graph : VGraph
        The vessel graph including B branches.

    topology : TreeTopology
        The tree topology ground truth.

    min_rank_threshold: float, optional
        If set, the branch tail points are ignored until their rank exceed floor(min_rank) + min_rank_threshold, namely the transition points between the current branch and its parent.
        Default is 0.1.

    max_rank_tolerance: float, optional
        If set, extend the main label of the branch to its head points if their rank floating part is below max_rank_tolerance, namely the transition points between the current branch and its children.
        Default is 0.25.

    Returns
    -------
    branch_label: npt.NDArray[TopologicalLabel]
        A 1D array of size (B,) indicating, for each branch, its topological label. Zero indicates that the branch was not found in the topology ground truth.

    branch_dir: npt.NDArray[np.float32]
        A 1D array of size (B,) indicating, for each branch, its direction. Positive value indicates to keep the branch original direction in ``graph``, negative value indicates to flip it. Zero indicates unknown direction.

    branch_plausibility: Float1DArray
        A 1D array of size (B,) indicating, the mean of the fuzzy_skeleton_map values under each branch's skeleton.

    tips_label: npt.NDArray[TopologicalLabel]
        A 2D array of size (B, 2) indicating, for each branch, the topological labels of its two tips (tail, head).

    tips_rank: npt.NDArray[np.float32]
        A 2D array of size (B, 2) indicating, for each branch, the topological distances of its two tips (tail, head).
    """  # noqa: E501
    domain = Rect.from_size(topology.shape).exclude_bottom_right_edges()  # type: ignore

    B = graph.branch_count
    branch_dir = np.zeros(B, dtype=float)
    branch_label = np.zeros(B, dtype=TopologicalLabel)
    branch_plausibility = np.zeros(B, dtype=float)

    tips_label = np.zeros((B, 2), dtype=TopologicalLabel)
    tips_rank = np.zeros((B, 2), dtype=float)

    for branch in graph.branches():
        curve = branch.curve()
        if curve is None or len(curve) < 3:
            p0, p1 = branch.tip_coord()
            curve = rasterize_line(p0.to_int_pair(), p1.to_int_pair())
        curve = curve[domain.contains(curve)]
        if len(curve) < 3:
            continue

        curve_label = topology.branch_map[*curve.T]
        # → Check that at least half the branch is inside the gt tree topology
        known_label = curve_label != 0
        if known_label.mean() < 0.5:
            continue

        # → Only consider points descendant of the main ancestor ...
        labels_occurence = TopologicalLabel.descendance_count(curve_label[known_label])
        main_ancestor: TopologicalLabel = max(labels_occurence, key=lambda x: labels_occurence[x][1])
        valid_label = main_ancestor.is_parent_of(curve_label, or_self=True)
        # ... and check that sufficient points are kept
        if valid_label.sum() < 3 or valid_label[known_label].mean() < 0.75:
            continue
        curve = curve[valid_label]
        curve_label = curve_label[valid_label]

        # → Get the direction of the branch based on the topological distance map
        curve_rank = topology.rank_map[*curve.T]
        curve_diff = np.diff(curve_rank)
        dir = np.mean(curve_diff > 0) - np.mean(curve_diff < 0)
        branch_dir[branch.id] = dir

        # → Exclude starting curve points which are part of the transition between labels
        if min_rank_threshold > 0:
            min_rank = np.floor(curve_rank.min())
            ignore_mask = curve_rank < min_rank + min_rank_threshold
            if ignore_mask.any() and (~ignore_mask).sum() > 3:
                curve = curve[~ignore_mask]
                curve_label = curve_label[~ignore_mask]
                curve_rank = curve_rank[~ignore_mask]

        # → Extend previous label for points that are part of the transition at the end of the branch
        if max_rank_tolerance > 0 and (max_rank := curve_rank.max()) % 1 < max_rank_tolerance and max_rank >= 1:
            max_rank = np.floor(max_rank)  # Clip max_rank to nearest lower integer
            extend_mask = curve_rank >= max_rank
            if not np.all(extend_mask):
                curve_label[extend_mask] = TopologicalLabel(curve_label[curve_rank.argmax()]).parent
                curve_rank[extend_mask] = max_rank

        # → Assign the most occurring label to the branch
        unique_labels, labels_count = np.unique(curve_label, return_counts=True)
        branch_label[branch.id] = unique_labels[labels_count.argmax()]

        # → Get the plausibility of the branch based on the fuzzy_skeleton_map
        branch_plausibility[branch.id] = topology.fuzzy_skeleton_map[*curve.T].mean()

        # → Get the tip labels and distances
        tips_label[branch.id, 0] = curve_label[0]
        tips_label[branch.id, 1] = curve_label[-1]
        tips_rank[branch.id, 0] = curve_rank[0]
        tips_rank[branch.id, 1] = curve_rank[-1]

    # → Filter branches that overlap on gt to keep only the most plausible one
    branch_first_tip = np.where(branch_dir >= 0, 0, 1)
    B_idx = np.arange(B)
    l0, l1 = tips_label[B_idx, branch_first_tip], tips_label[B_idx, 1 - branch_first_tip]
    d0, d1 = tips_rank[B_idx, branch_first_tip], tips_rank[B_idx, 1 - branch_first_tip]
    inters = dict(start=l0, end=l1, strict=False, start_d=d0, end_d=d1, strict_d=True)
    tip0_overlap = TopologicalLabel.is_between(l0, point_d=d0, **inters)  # type: ignore
    tip1_overlap = TopologicalLabel.is_between(l1, point_d=d1, **inters)  # type: ignore

    for cluster in reduce_clusters(np.argwhere(tip0_overlap | tip1_overlap)):
        if len(cluster) <= 1:
            continue
        cluster = np.array(cluster)
        cluster_plausibility = branch_plausibility[cluster]
        ignored_branch = np.ones(len(cluster), dtype=bool)
        ignored_branch[np.argmax(cluster_plausibility)] = False
        branch_label[cluster[ignored_branch]] = TopologicalLabel(0)
        branch_plausibility[cluster[ignored_branch]] = 0.0

    return branch_label, branch_dir, branch_plausibility, tips_label, tips_rank


def optimal_lines(
    graph: VGraph, topology: TreeTopology, lines: npt.NDArray[np.int_]
) -> tuple[npt.NDArray[np.bool_], Float1DArray, Float1DArray]:
    """
    Determine which lines between branch tips are valid based on the topological labels.

    Parameters
    ----------
    graph : VGraph
        The vessel graph including B branches.

    topology : TreeTopology
        The tree topology ground truth.

    lines : npt.NDArray[np.int_]
        A 2D array of size (N, 4) indicating lines between branch tips. Each row is of the form (b0, b0_tip, b1, b1_tip), where b0 and b1 are branch indices, and b0_tip and b1_tip are tip indices (0 for beginning, 1 for end of the curve).
        This array should not contains any duplicate lines.

    Returns
    -------
    optimal_lines: npt.NDArray[np.bool_]
        A 1D boolean array of size (N,) indicating which lines are optimal (True).

    branch_dir: Float1DArray
        A 1D array of size (B,) indicating, for each branch, its direction. Positive value indicates to keep the branch original direction in ``graph`, negative value indicates to flip it. Zero indicates unknown direction.

    branch_plausibility: npt.NDArray[np.bool_]
        A 1D boolean array of size (B,) indicating which branches were found (True) in the topology gt.
    """  # noqa: E501
    B = graph.branch_count

    branch_label, branch_dir, branch_plausibility, tips_label, tips_rank = read_branch_topology(graph, topology)
    missing = branch_label == 0

    B_idx = np.arange(B)
    B_dir = np.where(branch_dir >= 0, 1, 0)
    heads_label, heads_rank = tips_label[B_idx, B_dir], tips_rank[B_idx, B_dir]
    tails_label, tails_rank = tips_label[B_idx, 1 - B_dir], tips_rank[B_idx, 1 - B_dir]

    # Separate root lines
    root_lines_mask = lines[:, 0] == -1
    root_lines = lines[root_lines_mask][:, 2:]
    lines = lines[~root_lines_mask]

    # Remove not relevant root lines based on unknown branches and branches direction
    valid_root_lines_mask = ~missing[root_lines[:, 0]] & (root_lines[:, 1] == 1 - B_dir[root_lines[:, 0]])
    valid_root_lines = np.full((B,), -1, dtype=np.int_)
    valid_root_lines[root_lines[valid_root_lines_mask, 0]] = np.arange(len(root_lines))[valid_root_lines_mask]

    # Remove not relevant lines based on unknown branches
    valid_lines = ~missing[lines[:, 0]] & ~missing[lines[:, 2]]
    # Remove not relevant lines based on branches direction
    valid_lines &= lines[:, 1] == B_dir[lines[:, 0]]  # Tail tip should be 0 if dir>0 else 1
    valid_lines &= lines[:, 3] == 1 - B_dir[lines[:, 2]]  # Head tip should be 1 if dir>0 else 0

    optimal_lines = np.zeros(lines.shape[0], dtype=np.bool_)
    optimal_roots = np.zeros(root_lines.shape[0], dtype=np.bool_)
    for b_id in np.where(~missing)[0]:
        tail_label = TopologicalLabel(tails_label[b_id])
        tail_rank = tails_rank[b_id]

        concerned_lines = np.argwhere(valid_lines & (lines[:, 2] == b_id)).flatten()
        lines_parent = lines[concerned_lines, 0]
        lines_parent_label = heads_label[lines_parent]
        lines_parent_rank = heads_rank[lines_parent]

        possible_ancestors = tail_label.is_child_of(lines_parent_label, or_self=True)
        possible_ancestors[lines_parent_rank > tail_rank] = False  # Parent must have a lower rank

        if possible_ancestors.sum() == 0:
            root_id = valid_root_lines[b_id]
            if root_id >= 0:
                optimal_roots[root_id] = True
            continue
        concerned_lines = concerned_lines[possible_ancestors]
        optimal_ancestor = np.argmax(lines_parent_rank[possible_ancestors])  # Prefer the nearest (i.e. highest rank)
        optimal_lines[concerned_lines[optimal_ancestor]] = True

    all_optimal_lines = np.zeros(lines.shape[0] + root_lines.shape[0], dtype=np.bool_)
    all_optimal_lines[~root_lines_mask] = optimal_lines
    all_optimal_lines[root_lines_mask] = optimal_roots

    return all_optimal_lines, branch_dir, branch_plausibility


def optimal_branch_tree(graph: VGraph, topology: TreeTopology) -> tuple[Int1DArray, Float1DArray, Bool1DArray]:
    """
    Compute the optimal arborescence of branches for the given graph based on the topological labels.

    Parameters
    ----------
    graph : VGraph
        The vessel graph including B branches.

    topology : TreeTopology
        The tree topology ground truth.

    restrict_lines : Optional[npt.NDArray[np.int_]], optional
        An optional 2D array of size (N, 4) indicating which branch tips can be connected together. Each row is of the form (b0, b0_tip, b1, b1_tip), where b0 and b1 are branch indices, and b0_tip and b1_tip are tip indices (0 for beginning, 1 for end of the curve).

        If None (by default), all tips are allowed to connect.

    Returns
    -------
    branch_tree: Int1DArray
        A 1D array of size (B,) indicating, for each branch, the index of its parent branch in the optimal arborescence. -1 indicates no parent (root branch).

    branch_dir: Float1DArray
        A 1D array of size (B,) indicating, for each branch, its direction. Positive value indicates to keep the branch original direction in ``graph``, negative value indicates to flip it. Zero indicates unknown direction.

    missing: Bool1DArray
        A 1D array of size (B,) indicating if each branch is missing (True) in the topology gt or not (False).
    """  # noqa: E501
    B = graph.branch_count
    branch_tree = np.full(B, -1, dtype=np.int_)

    branch_label, branch_dir, _, tips_label, tips_rank = read_branch_topology(graph, topology)
    missing = branch_label == 0

    B_idx = np.arange(B)
    B_dir = np.where(branch_dir >= 0, 1, 0)
    heads_label, heads_rank = tips_label[B_idx, B_dir], tips_rank[B_idx, B_dir]
    tails_label, tails_rank = tips_label[B_idx, 1 - B_dir], tips_rank[B_idx, 1 - B_dir]

    for b_id in np.where(~missing)[0]:
        tail_label = TopologicalLabel(tails_label[b_id])
        tail_rank = tails_rank[b_id]

        ancestors = tail_label.is_child_of(heads_label, or_self=True)
        ancestors[b_id] = False  # A branch cannot be its own parent
        ancestors[heads_rank > tail_rank] = False  # Parent must have a lower distance
        ancestors = np.where(ancestors)[0]

        if len(ancestors) == 0:
            continue

        best_ancestor = ancestors[np.argmax(heads_rank[ancestors])]  # Prefer the nearest (i.e. highest rank)
        branch_tree[b_id] = best_ancestor

    return branch_tree, branch_dir, missing


def evaluate_topology(
    tree: VTree, topo_labels: npt.NDArray[TopologicalLabel], topo_map: npt.NDArray[np.float32], *, epsilon=1e-5
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
    ...


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


########################################################################################################################
#       === TOPOLOGICAL GRAPHICAL REPRESENTATION UTILS ===
########################################################################################################################
class TopologicalLabel(np.uint64):
    """
    A class representing a topological label for a branch in the vessel tree, coded on a 64-bit integer.

    The bits are used as follows:
    - 0-11  : subtree index (12 bits, max: 4096)
    - 12-19 : branching rank (8 bits)
    - 20-63 : binary branching pattern (44 bits)

    Examples
    --------
    >>> label = TopologicalLabel.encode(subtree=7, branching_pattern=[True, False, True, True])
    >>> label
    TopologicalLabel(0x8b000000000004)
    >>> label.subtree, label.rank, label.branching_pattern
    (7, 4, array([ True, False,  True,  True]))

    """

    SUBTREE_MASK = np.uint64(0xFFF0000000000000)
    RANK_MASK = np.uint64(0x00000000000000FF)
    BRANCHING_PATTERN_MASK = np.uint64(0x000FFFFFFFFFFF00)

    @classmethod
    def encode(cls, subtree: int | np.int_, branching_pattern: Sequence[bool]) -> Self:
        """
        Create a TopologicalLabel from an integer label.
        """
        assert 0 <= subtree < 4095, "Subtree index must be between 0 and 4094."
        rank = len(branching_pattern)
        assert rank <= 44, "Branching pattern must be less than 44 bits."

        branching_pattern_int = np.uint64(sum((1 << (43 - i)) for i, b in enumerate(branching_pattern) if b))
        return cls((np.uint64(subtree + 1) << 52) | (branching_pattern_int << 8) | np.uint64(rank))

    @classmethod
    def decode(cls, map: npt.NDArray[np.uint64]) -> Tuple[npt.NDArray[np.uint32], List[Self]]:
        """
        Decode a topological label map into its components.
        """
        labels, labels_map = np.unique(map, return_inverse=True)
        return labels_map.astype(np.uint32), [cls(label) for label in labels]

    @classmethod
    @overload
    def decode_subtree(cls, label: Self | np.uint64) -> np.int32: ...
    @classmethod
    @overload
    def decode_subtree(cls, label: npt.NDArray[np.uint64]) -> npt.NDArray[np.int32]: ...
    @classmethod
    def decode_subtree(cls, label: Self | np.uint64 | npt.NDArray[np.uint64]) -> np.int32 | npt.NDArray[np.int32]:
        """
        Decode the subtree indices from a topological label map.
        """
        return (np.uint64(label & cls.SUBTREE_MASK) >> np.uint64(52)).astype(np.int32) - 1

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
        return (label & cls.RANK_MASK).astype(np.uint8)

    @classmethod
    @overload
    def subtree_branching_bit_mask(cls, max_rank: np.uint8 | int | None) -> np.uint64: ...
    @classmethod
    @overload
    def subtree_branching_bit_mask(cls, max_rank: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint64]: ...
    @classmethod
    def subtree_branching_bit_mask(
        cls, max_rank: np.uint8 | int | None | npt.NDArray[np.uint8]
    ) -> np.uint64 | npt.NDArray[np.uint64]:
        """
        Create a mask to extract the branching pattern up to a given rank.
        """
        if max_rank is None:
            return np.uint64(0xFFFFFFFFFFFFFF00)  # cls.SUBTREE_MASK | cls.BRANCHING_PATTERN_MASK
        assert np.all(max_rank <= 44), "Max rank must be less than or equal to 44."
        max_rank_ = max_rank.astype(np.uint64) if isinstance(max_rank, np.ndarray) else np.uint64(max_rank)
        return bit_invert(np.uint64(2) ** (np.uint64(52) - max_rank_) - np.uint64(1))  # type: ignore

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
        pattern = (label & cls.BRANCHING_PATTERN_MASK) >> np.uint64(8)
        if max_rank is not None:
            assert np.all(max_rank <= 44), "Max rank must be less than or equal to 44."
            mask = bit_invert(np.uint64(2) ** (np.uint64(44) - max_rank) - np.uint(1)) & np.uint64(0x00000FFFFFFFFFFF)
            pattern &= mask
        return pattern

    @classmethod
    @overload
    def get_parent(cls, label: Self | np.uint64, *, return_self_if_no_parent: bool = False) -> TopologicalLabel: ...
    @classmethod
    @overload
    def get_parent(
        cls, label: npt.NDArray[np.uint64], *, return_self_if_no_parent: bool = False
    ) -> npt.NDArray[np.uint64]: ...
    @classmethod
    def get_parent(
        cls, label: npt.NDArray[np.uint64] | Self | np.uint64, *, return_self_if_no_parent: bool = False
    ) -> npt.NDArray[np.uint64] | TopologicalLabel:
        """
        Get the parent labels of the given topological labels.

        Parameters
        ----------
        label : npt.NDArray[np.uint64] | Self | np.uint64
            The input topological labels.

        return_self_if_no_parent : bool, optional
            Whether to return the same label if it has no parent. Default is False.

        Returns
        -------
        npt.NDArray[np.uint64]
            The parent topological labels.

        Examples
        --------
        >>> TL = TopologicalLabel
        >>> label = TL.encode(subtree=3, branching_pattern=[True, False, True])
        >>> parent = TL.get_parent(label)
        >>> str(parent)
        '(subtree=3, branching=[True, False])'

        >>> bool(TL.encode(subtree=3, branching_pattern=[True, False]) == parent)
        True

        >>> str(TL.get_parent(TL.encode(subtree=0, branching_pattern=[])))
        '(no label)'
        >>> str(TL.get_parent(TL.encode(subtree=0, branching_pattern=[]), return_self_if_no_parent=True))
        '(subtree=0, branching=[])'

        """
        is_single = np.isscalar(label)
        if is_single:
            label = np.array([label], dtype=np.uint64)
            out = label
        else:
            out: npt.NDArray[np.uint64] = label.copy()  # type: ignore

        ranks = cls.decode_rank(out)
        has_parents = ranks > 0
        out[has_parents] -= 1  # Decrease rank by 1
        out[has_parents] &= cls.subtree_branching_bit_mask(ranks[has_parents] - 1) | cls.RANK_MASK  # Erase last bit
        if not return_self_if_no_parent:
            out[~has_parents] = 0
        return TopologicalLabel(out[0]) if is_single else out

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
        return int(self.decode_subtree(self))

    @property
    def rank(self) -> int:
        return int(self.decode_rank(self))

    @property
    def branching_pattern_bits(self) -> np.uint64:
        return self.decode_branching_pattern(self)

    @property
    def branching_pattern(self) -> npt.NDArray[np.bool_]:
        pattern = self.decode_branching_pattern(self)
        return np.array(
            [(pattern & np.uint64(1 << (43 - i))) != 0 for i in range(self.rank)],
            dtype=np.bool_,
        )

    @property
    def parent(self) -> TopologicalLabel:
        return self.get_parent(self)

    def __repr__(self) -> str:
        return f"TopologicalLabel({hex(int(self))})"

    def __str__(self) -> str:
        if self.is_background:
            return "(no label)"
        return (
            f"(subtree={self.subtree}, "
            f"branching=[{', '.join(['True' if b else 'False' for b in self.branching_pattern])}])"
        )

    def is_same_subtree(self, other: Self | np.int32) -> bool:
        """
        Check if this label belongs to the same subtree as another label.
        """
        o = TopologicalLabel(other)
        return self.subtree == o.subtree

    @classmethod
    def ancestor_check(
        cls, parent: npt.NDArray[np.uint64], child: npt.NDArray[np.uint64], strict: bool = True
    ) -> npt.NDArray[np.bool_]:
        """
        Check if each label in `parent` is an ancestor of the corresponding label in `child`.

        Parameters
        ----------
        parent : npt.NDArray[np.uint64]
            Array of N parent labels.
        child : npt.NDArray[np.uint64]
            Array of M child labels.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean 2D array of shape (N, M) indicating ancestor relationships.

        Examples
        --------
        >>> TL = TopologicalLabel
        >>> parent = np.array([TL.encode(subtree=1, branching_pattern=[True]),
        ...                    TL.encode(subtree=1, branching_pattern=[True, False])])
        >>> child = np.array([TL.encode(subtree=1, branching_pattern=[True, False, True]),
        ...                   TL.encode(subtree=1, branching_pattern=[True, True]),
        ...                   TL.encode(subtree=1, branching_pattern=[False, True])])
        >>> TL.ancestor_check(parent, child)
        array([[ True, True, False],
               [ True, False, False]])
        """
        assert parent.ndim == 1 and child.ndim == 1, "Input arrays must be 1D."
        parent, child = np.asarray(parent)[:, None], np.asarray(child)[None, :]
        parent_rank_masks = cls.subtree_branching_bit_mask(cls.decode_rank(parent))
        is_higher_rank = parent < child if strict else parent <= child
        return is_higher_rank & ((parent & parent_rank_masks) == (child & parent_rank_masks))

    @classmethod
    def is_between(
        cls,
        point: npt.NDArray[np.uint64],
        start: npt.NDArray[np.uint64],
        end: npt.NDArray[np.uint64],
        point_d: npt.NDArray[np.float64] | None = None,
        start_d: npt.NDArray[np.float64] | None = None,
        end_d: npt.NDArray[np.float64] | None = None,
        *,
        strict: bool = True,
        strict_d: Optional[bool] = None,
    ) -> npt.NDArray[np.bool_]:
        """
        Check if each label in `point` is between the corresponding labels in `start` and `end`.

        Parameters
        ----------
        point : npt.NDArray[np.uint64]
            1D array of N point labels.
        start : npt.NDArray[np.uint64]
            1D array of M start labels.
        end : npt.NDArray[np.uint64]
            1D array of M end labels.
        point_d : npt.NDArray[np.float64] | None, optional
            1D array storing the topological distances of points, by default None
        start_d : npt.NDArray[np.float64] | None, optional
            1D array storing the topological distances of start points, by default None
        end_d : npt.NDArray[np.float64] | None, optional
            1D array storing the topological distances of end points, by default None
        strict : bool, optional
            Whether to use strict inequalities, by default True

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean 2D array indicating if each point is between the corresponding start and end labels.
        """
        assert point.ndim == 1 and start.ndim == 1 and end.ndim == 1, "Input arrays must be 1D."
        assert start.shape == end.shape, "start and end must have the same shape."

        point, start, end = np.asarray(point)[:, None], np.asarray(start)[None, :], np.asarray(end)[None, :]

        start_rank_masks = cls.subtree_branching_bit_mask(cls.decode_rank(start))
        is_descendant = (start < point) if strict else (start <= point)
        is_descendant &= (point & start_rank_masks) == (start & start_rank_masks)

        point_rank_masks = cls.subtree_branching_bit_mask(cls.decode_rank(point))
        is_ancestor = (point < end) if strict else (point <= end)
        is_ancestor &= (point & point_rank_masks) == (end & point_rank_masks)

        between_mask = is_descendant & is_ancestor

        if point_d is not None:
            assert start_d is not None and end_d is not None, (
                "start_d and end_d must be provided if point_d is provided."
            )
            assert point_d.ndim == 1 and start_d.ndim == 1 and end_d.ndim == 1, "Input distance arrays must be 1D."
            assert point_d.shape[0] == point.shape[0], "point_d must have the same length as point."
            assert start_d.shape[0] == start.shape[1], "start_d must have the same length as start."
            assert end_d.shape[0] == end.shape[1], "end_d must have the same length as end."

            point_d, start_d, end_d = point_d[:, None], start_d[None, :], end_d[None, :]
            if strict_d is True or strict_d is None and strict is True:
                between_mask &= (start_d < point_d) & (point_d < end_d)
            else:
                between_mask &= (start_d <= point_d) & (point_d <= end_d)

        return between_mask

    @overload
    def is_parent_of(self, other: Self | np.uint64, or_self: bool = False) -> bool: ...
    @overload
    def is_parent_of(self, other: npt.NDArray[np.uint64], or_self: bool = False) -> npt.NDArray[np.bool_]: ...
    def is_parent_of(
        self, other: Self | np.uint64 | npt.NDArray[np.uint64], or_self: bool = False
    ) -> bool | npt.NDArray[np.bool_]:
        """
        Check if this label is a parent of another label.

        Parameters
        ----------
        other : Self | np.uint64 | npt.NDArray[np.uint64]
            The other label(s) to compare with.
        or_self : bool, optional
            If True, consider the label as a parent of itself, by default False.

        Examples
        --------
        >>> TL = TopologicalLabel
        >>> parent = TL.encode(subtree=1, branching_pattern=[True])
        >>> child = TL.encode(subtree=1, branching_pattern=[True, False, True])
        >>> parent.is_parent_of(child)
        True

        >>> parent.is_parent_of(parent)
        False

        >>> parent.is_parent_of(parent, or_self=True)
        True

        >>> other_child = TL.encode(subtree=1, branching_pattern=[True, True])
        >>> other_child.is_parent_of(child)
        False

        >>> other_parent = TL.encode(subtree=0, branching_pattern=[True, True])
        >>> parent.is_parent_of(np.array([parent, other_parent, child, other_child]))
        array([False, False, True, True])
        """
        self_rank_mask = self.subtree_branching_bit_mask(self.rank)
        if not isinstance(other, np.ndarray):
            o = TopologicalLabel(other)
            if self & self_rank_mask != other & self_rank_mask:
                return False
            return self.rank <= o.rank if or_self else self.rank < o.rank
        else:
            is_higher_rank = self <= other if or_self else self < other
            return is_higher_rank & (self & self_rank_mask == other & self_rank_mask)

    @overload
    def is_child_of(self, other: Self | np.uint64, or_self: bool = False) -> bool: ...
    @overload
    def is_child_of(self, other: npt.NDArray[np.uint64], or_self: bool = False) -> npt.NDArray[np.bool_]: ...
    def is_child_of(
        self, other: Self | np.uint64 | npt.NDArray[np.uint64], or_self: bool = False
    ) -> bool | npt.NDArray[np.bool_]:
        """
        Check if this label is a child of another label.

        Parameters
        ----------
        other : Self | np.uint64 | npt.NDArray[np.uint64]
            The other label(s) to compare with.
        or_self : bool, optional
            If True, consider the label as a child of itself, by default False.

        Examples
        --------
        >>> parent = TopologicalLabel.encode(subtree=1, branching_pattern=[True])
        >>> child = TopologicalLabel.encode(subtree=1, branching_pattern=[True, False, True])
        >>> child.is_child_of(parent)
        True

        >>> child.is_child_of(child)
        False

        >>> child.is_child_of(child, or_self=True)
        True

        >>> other_parent = TopologicalLabel.encode(subtree=1, branching_pattern=[False])
        >>> child.is_child_of(other_parent)
        False

        >>> child.is_child_of(np.array([parent, other_parent, child]))
        array([ True, False, False])
        """
        other_rank_masks = self.subtree_branching_bit_mask(self.decode_rank(other))
        if not isinstance(other, np.ndarray):
            o = TopologicalLabel(other)
            if self & other_rank_masks != other & other_rank_masks:
                return False
            return self.rank >= o.rank if or_self else self.rank > o.rank
        else:
            is_lower_rank = self >= other if or_self else self > other
            return is_lower_rank & (self & other_rank_masks == other & other_rank_masks)

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
    def descendance_count(cls, array: npt.NDArray[TopologicalLabel]) -> dict[TopologicalLabel, tuple[int, int]]:
        """Count the occurrence of each label in array similarly to np.unique(array, return_counts=True). However child label also increment their ancestor labels.

        Parameters
        ----------
        array : npt.NDArray[TopologicalLabel]
            Array of labels.

        Returns
        -------
        dict[TopologicalLabel, tuple[int, int]]
            A dictionary whose keys are unique labels and values are a tuple counting the occurrences without and with their descendants.

        Example
        -------
        >>> parent = TopologicalLabel.encode(subtree=1, branching_pattern=[True])
        >>> child_a1 = TopologicalLabel.encode(subtree=1, branching_pattern=[True, False])
        >>> child_a2 = TopologicalLabel.encode(subtree=1, branching_pattern=[True, False, True])
        >>> child_b = TopologicalLabel.encode(subtree=1, branching_pattern=[True, True])
        >>> TopologicalLabel.descendance_count(np.array([parent, child_a1, child_a1, child_a2, child_b]))
        {TopologicalLabel(subtree=1, rank=1, branching_pattern=[True]): (1, 5),
         TopologicalLabel(subtree=1, rank=2, branching_pattern=[True, False]): (2, 3),
         TopologicalLabel(subtree=1, rank=3, branching_pattern=[True, False, True]): (1, 1),
         TopologicalLabel(subtree=1, rank=2, branching_pattern=[True, True]): (1, 1)}
        """  # noqa: E501
        labels, counts = np.unique(array, return_counts=True)
        hier_counts = {}
        for label, count in zip(labels, counts, strict=True):
            label = TopologicalLabel(label)
            # Increment ancestor labels
            for anc in hier_counts.keys():
                if label.is_child_of(anc):
                    hier_counts[anc] = (hier_counts[anc][0], hier_counts[anc][1] + int(count))
            hier_counts[label] = (int(count), int(count))
        return hier_counts

    @classmethod
    @overload
    def subtree_color(cls, subtree: int, format: Literal["hex"] = "hex") -> str: ...
    @classmethod
    @overload
    def subtree_color(cls, subtree: int, format: Literal["rgb"]) -> npt.NDArray[np.uint8]: ...
    @classmethod
    def subtree_color(cls, subtree: int, format: Literal["hex", "rgb"] = "hex") -> str | npt.NDArray[np.uint8]:
        """Assign a base color to each subtree."""
        if subtree < 0:
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


def branch_topological_mapping(tree: VTree) -> npt.NDArray[TopologicalLabel]:
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

    labels = np.zeros(tree.branch_count + 1, dtype=TopologicalLabel)

    def recursive_labeling(branch: VTreeBranch, subtree: int, branching_pattern: list[bool]) -> None:
        nonlocal labels

        labels[lookup[branch.id]] = TopologicalLabel.encode(subtree, branching_pattern)

        for i, successor_id in enumerate(sorted(branch.successors_ids)):
            if i != branch.n_successors - 1 or branch.n_successors == 1:
                branching_pattern = branching_pattern + [True]
            recursive_labeling(working_tree.branch(successor_id), subtree, branching_pattern)
            branching_pattern = branching_pattern[:-1] + [False]

    for subtree, branch in enumerate(working_tree.root_branches()):
        recursive_labeling(branch, subtree, [])

    return labels
