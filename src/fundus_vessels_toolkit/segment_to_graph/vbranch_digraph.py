from __future__ import annotations

import warnings
from functools import cached_property
from itertools import pairwise
from typing import Literal, Optional, Protocol, Self, Sequence, TypeGuard, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from networkx import maximum_branching

from ..utils.cluster import cluster_by_distance
from ..utils.lookup_array import create_removal_lookup
from ..utils.math import gaussian, sigmoid, softmax
from ..utils.numpy import np_first_true, np_group_by, np_groupby_mean
from ..utils.tree import (
    accessible_from_root,
    find_cycles,
    has_cycle,
    tree_connected_components,
    tree_distance,
    tree_node_rank,
)
from ..utils.typing import Bool1DArray, Float1DArray, Indices, IndicesLike, Int1DArray, Int1DArrayLike, Int2DArrayLike
from ..vascular_data_objects import VBranchGeoData, VGraph
from ..vascular_data_objects.fundus_data import AVLabel
from ..vascular_data_objects.vtree import VTree
from .geometry_parsing import derive_tips_geometry_from_curve_geometry
from .graph_simplification import find_facing_tips
from .tree_topology import TreeTopology, highest_topo_plausibility


########################################################################################################################
#       === VBranchDigraph CLASS ===
########################################################################################################################
class LineDigraph:
    def __init__(
        self,
        line_list: npt.NDArray[np.int_],
    ):
        """A directed graph representing possible reconnections between line segments.

        Parameters
        ----------
        line_list : npt.NDArray[np.int_]
            An array of shape (N, 4) representing the directed edges connecting the line segment l0 to the line segment l1. Each row is in the format ``(l0, l0_tip, l1, l1_tip)``, where ``l0_tip`` and ``l1_tip`` are in {0, 1} indicates if the line segments are connected through their first (0) or second (1) node.
            (Namely: ``line_list[l0,l0_tip]`` and ``line_list[l1,l1_tip]``).
        """  # noqa: E501
        assert line_list.ndim == 2 and line_list.shape[1] == 4, "line_list must be of shape (N, 4)"
        self.line_list = line_list

    def line_subset(self, idx: Bool1DArray | Indices) -> Self:
        """Get a subgraph of the directed graph containing only the lines selected by the line_mask.

        Parameters
        ----------
        idx : Bool1DArray | Indices
            An boolean array of shape (N,) or an array of indices indicating which lines to keep in the subgraph.

        Returns
        -------
        Self
            A new LineDigraph instance containing only the selected lines.
        """  # noqa: E501
        return self.__class__(line_list=self.line_list[idx])

    def __getitem__(self, line_mask: Bool1DArray | IndicesLike) -> Self:
        """Get a subgraph of the directed graph containing only the lines selected by the line_mask.

        Parameters
        ----------
        line_mask : Bool1DArray
            An array of shape (N,) indicating which lines to keep in the subgraph.

        Returns
        -------
        Self
            A new LineDigraph instance containing only the selected lines.
        """
        return self.line_subset(np.asarray(line_mask))

    def not_root_lines(self) -> Self:
        """Get a subgraph of the directed graph containing only the non-root lines.

        Returns
        -------
        Self
            A new LineDigraph instance containing only the non-root lines.
        """
        return self.line_subset(self.line_list[:, 0] != -1)

    @property
    def b0(self) -> npt.NDArray[np.int_]:
        return self.line_list[:, 0]

    @property
    def b0_tip(self) -> npt.NDArray[np.int_]:
        return self.line_list[:, 1]

    @property
    def b0_dir(self) -> npt.NDArray[np.bool_]:
        return self.line_list[:, 1] == 1

    @property
    def b1(self) -> npt.NDArray[np.int_]:
        return self.line_list[:, 2]

    @property
    def b1_tip(self) -> npt.NDArray[np.int_]:
        return self.line_list[:, 3]

    @property
    def b1_dir(self) -> npt.NDArray[np.bool_]:
        return self.line_list[:, 3] == 0

    @property
    def b0b1(self) -> npt.NDArray[np.int_]:
        return self.line_list[:, [0, 2]]

    @property
    def b0tip_b1tip(self) -> npt.NDArray[np.int_]:
        return self.line_list[:, [1, 3]]

    @property
    def b0b1_dir(self) -> npt.NDArray[np.bool_]:
        return np.stack([self.b0_dir, self.b1_dir], axis=1)

    @cached_property
    def root_mask(self) -> npt.NDArray[np.bool_]:
        return self.line_list[:, 0] == -1

    def search_lines(self, searched_lines: Int2DArrayLike) -> npt.NDArray[np.int_]:
        """Search for lines in the directed graph.

        Parameters
        ----------
        searched_lines : Int2DArrayLike
            A list of M lines to search for in the format (l0, l0_tip, l1, l1_tip).

        Returns
        -------
        npt.NDArray[np.int_]
            An array of shape (M,) representing the first indices of the searched lines in the directed graph.
        """
        searched_lines = np.asarray(searched_lines, dtype=np.int_)
        if searched_lines.ndim == 1:
            searched_lines = searched_lines[None, :]

        lines_ids = np.full(len(searched_lines), -1, dtype=np.int_)
        for i, line in enumerate(searched_lines):
            match = np.argwhere(np.all(self.line_list == line, axis=1)).flatten()
            if len(match) > 0:
                lines_ids[i] = match[0]

        return lines_ids


class VBranchDigraph(LineDigraph):
    def __init__(
        self,
        line_list: npt.NDArray[np.int_],
        line_p: Optional[Float1DArray] = None,
        branch_dir_p: Optional[Float1DArray] = None,
        branch_fp_p: Optional[Float1DArray] = None,
        branch_av_p: Optional[Float1DArray] = None,
        *,
        graph: Optional[VGraph] = None,
        branch_count: Optional[int] = None,
        branch_dir_logit: Optional[Float1DArray] = None,
        branch_fp_logit: Optional[Float1DArray] = None,
        branch_av_logit: Optional[Float1DArray] = None,
    ):
        """A directed graph representing possible reconnections between branches.

        Parameters
        ----------
        graph : VGraph
            The vascular graph.
        line_list : npt.NDArray[np.int_]
            An array of shape (N, 4) representing the directed edges connecting the branch b0 to the branch b1. Each row is in the format ``(b0, b0_tip, b1, b1_tip)``, where ``b0_tip`` and ``b1_tip`` are in {0, 1} indicates if the branches are connected through their first (0) or second (1) node.
            (Namely: ``graph.branch_list[b0,b0_tip]`` and ``graph.branch_list[b1,b1_tip]``).
        line_p : Optional[npt.NDArray[np.float_]], optional
            An array of shape (N,) representing the probabilities of each edge, by default None.
        branch_dir_p : Optional[npt.NDArray[np.float_]], optional
            An array of shape (B,) representing the direction probabilities of each branch, by default None
        branch_fp_p : Optional[npt.NDArray[np.float_]], optional
            An array of shape (B,) representing the false positive probabilities of each branch, by default None
        branch_av_p : Optional[npt.NDArray[np.float_]], optional
            An array of shape (B,) representing the artery probabilities of each branch, by default None.
        """  # noqa: E501
        super().__init__(line_list=line_list)
        self.graph = graph
        self.line_p = line_p
        self._branch_dir_p = branch_dir_p
        self._branch_fp_p = branch_fp_p
        self._branch_fp_logit = branch_fp_logit
        self._branch_av_p = branch_av_p
        self._branch_av_logit = branch_av_logit
        self._branch_dir_logit = branch_dir_logit

        if branch_count is None:
            # Infer branch count from graph or branch attributes
            if graph is not None:
                branch_count = graph.branch_count
            else:
                for branch_attr in (
                    branch_dir_p,
                    branch_dir_logit,
                    branch_fp_p,
                    branch_fp_logit,
                    branch_av_p,
                    branch_av_logit,
                ):
                    if branch_attr is not None:
                        branch_count = len(branch_attr)
                        break
                if branch_count is None:
                    raise ValueError("branch_count was not provided and can't be inferred.")
        self.branch_count = branch_count

    def line_subset(self, idx: Bool1DArray | Indices) -> Self:
        return self.__class__(
            line_list=self.line_list[idx],
            line_p=self.line_p[idx] if self.line_p is not None else None,
            branch_dir_p=self._branch_dir_p,
            branch_fp_p=self._branch_fp_p,
            branch_av_p=self._branch_av_p,
            branch_dir_logit=self._branch_dir_logit,
            branch_fp_logit=self._branch_fp_logit,
            branch_av_logit=self._branch_av_logit,
            graph=self.graph,
            branch_count=self.branch_count,
        )

    # === BRANCH PROPERTIES ===
    @property
    def branch_dir_p(self) -> Float1DArray | None:
        """Get the direction probability of each branch.

        Returns
        -------
        npt.NDArray[np.float_]
            An array of shape (B,) representing the direction probabilities of each branch.
        """
        if self._branch_dir_p is None:
            if self._branch_dir_logit is None:
                return None
            dir_logit = self._branch_dir_logit
            self._branch_dir_p = sigmoid(dir_logit)
        return self._branch_dir_p

    @property
    def branch_dir_logit(self) -> Float1DArray | None:
        """Get the direction logit of each branch.

        Returns
        -------
        npt.NDArray[np.float_]
            An array of shape (B,) representing the direction logits of each branch.
        """
        if self._branch_dir_logit is None:
            if self._branch_dir_p is None:
                return None
            dir_p = self._branch_dir_p + 1e-6  # avoid log(0)
            self._branch_dir_logit = np.log(dir_p) - np.log(1 - dir_p)
        return self._branch_dir_logit

    @property
    def branch_dir(self) -> Bool1DArray | None:
        """Get the direction of each branch.

        Returns
        -------
        npt.NDArray[np.bool_]
            An array of shape (B,) representing the direction of each branch: True if the branch is directed from its first tip to its second tip, False otherwise.
        """  # noqa: E501
        if (dir_p := self._branch_dir_p) is not None:
            return dir_p > 0.5
        if (dir_logit := self._branch_dir_logit) is not None:
            return dir_logit > 0
        return None

    def line_dir_p(self) -> Float1DArray | None:
        """Get the probability of each line based on the direction probabilities of its branches.

        Returns
        -------
        npt.NDArray[np.float32]
            An array of shape (N,) representing the direction probabilities of each line, or None if branch_dir_p is not provided.
        """  # noqa: E501
        if (dir_p := self.branch_dir_p) is None:
            return None
        b0_dir_p, b1_dir_p = dir_p[self.b0], dir_p[self.b1]
        b0_dir_p = np.where(self.b0_dir, b0_dir_p, 1 - b0_dir_p)
        b1_dir_p = np.where(self.b1_dir, b1_dir_p, 1 - b1_dir_p)
        return np.where(self.b0 != -1, 0.5 * (b0_dir_p + b1_dir_p), b1_dir_p)

    @property
    def branch_fp_p(self) -> Float1DArray | None:
        """Get the false positive probability of each branch.

        Returns
        -------
        npt.NDArray[np.float32]
            An array of shape (B,) representing the false positive probabilities of each branch.
        """
        if self._branch_fp_p is None:
            if (fp_logit := self._branch_fp_logit) is None:
                return None
            self._branch_fp_p = sigmoid(fp_logit)
        return self._branch_fp_p

    @property
    def branch_fp_logit(self) -> Float1DArray | None:
        """Get the false positive logit of each branch.

        Returns
        -------
        npt.NDArray[np.float32]
            An array of shape (B,) representing the false positive logits of each branch.
        """
        if self._branch_fp_logit is None:
            if (fp_p := self._branch_fp_p) is None:
                return None
            fp_p += 1e-6  # avoid log(0)
            self._branch_fp_logit = np.log(fp_p) - np.log(1 - fp_p)
        return self._branch_fp_logit

    def branch_fp(self) -> npt.NDArray[np.bool_]:
        """Get a mask of invalid branches.

        Returns
        -------
        npt.NDArray[np.bool_]
            An array of shape (B,) representing whether each branch is invalid.
        """
        if (fp_p := self._branch_fp_p) is not None:
            return fp_p > 0.5
        elif (fp_logit := self._branch_fp_logit) is not None:
            return fp_logit > 0
        else:
            return np.zeros(self.branch_count, dtype=bool)

    @property
    def branch_av_p(self) -> Float1DArray | None:
        """Get the artery probability of each branch.

        Returns
        -------
        npt.NDArray[np.float32]
            An array of shape (B,) representing the artery probabilities of each branch.
            An array of shape (B,) representing the artery probabilities of each branch, or None if not provided.
        """
        if self._branch_av_p is None:
            if (av_logit := self._branch_av_logit) is None:
                return None
            self._branch_av_p = sigmoid(av_logit)
        return self._branch_av_p

    @property
    def branch_av_logit(self) -> Float1DArray | None:
        """Get the artery logit of each branch.

        Returns
        -------
        npt.NDArray[np.float32]
            An array of shape (B,) representing the artery logits of each branch.
        """
        if self._branch_av_logit is None:
            if (av_p := self._branch_av_p) is None:
                return None
            av_p += 1e-6  # avoid log(0)
            self._branch_av_logit = np.log(av_p) - np.log(1 - av_p)
        return self._branch_av_logit

    def branch_av_class(self) -> Int1DArray | None:
        """Get the artery/vein class of each branch.

        Returns
        -------
        npt.NDArray[np.int_]
            An array of shape (B,) representing the class of each branch: 0 for invalid, 1 for artery, 2 for vein.
        """  # noqa: E501
        av_class = np.zeros((self.branch_count,), dtype=np.int_)
        if self._branch_fp_p is not None:
            tp = self._branch_fp_p < 0.5
        elif self._branch_fp_logit is not None:
            tp = self._branch_fp_logit < 0
        else:
            return None

        if self._branch_av_p is not None:
            av_class[tp] = np.where(self._branch_av_p[tp] > 0.5, 1, 2)  # artery if art_p > 0.5, vein otherwise
        elif self._branch_av_logit is not None:
            av_class[tp] = np.where(self._branch_av_logit[tp] > 0, 1, 2)  # artery if art_logit > 0, vein otherwise
        else:
            return None
        return av_class

    def line_av_p(self) -> Float1DArray | None:
        """Get the probability of each line based on the similarity of the av logits of its branches.

        Returns
        -------
        npt.NDArray[np.float32]
            An array of shape (N,) representing the artery probabilities of each line, or None if branch_av_logit is not provided.
        """  # noqa: E501
        if (av_logit := self.branch_av_logit) is None:
            return None
        return sigmoid(av_logit[self.b0]) * sigmoid(av_logit[self.b1])

    @classmethod
    def has_fp_av_p(cls, instance: Self) -> TypeGuard[_VBranchDigraphWithAVProba]:
        """Check if the digraph has artery/vein class information (i.e. if either branch_av_p or branch_av_logit is not None)."""  # noqa: E501
        return _VBranchDigraphWithAVProba.check(instance)

    @classmethod
    def has_all_p(cls, instance: Self) -> TypeGuard[_VBranchDigraphWithAllProba]:
        """Check if the digraph has artery/vein class, false positive and direction information (i.e. if branch_av_p or branch_av_logit is not None, and if branch_fp_p or branch_fp_logit is not None, and if branch_dir_p or branch_dir_logit is not None)."""  # noqa: E501
        return _VBranchDigraphWithAllProba.check(instance)

    # === DIGRAPH BUILDING ===
    @classmethod
    def from_graph(
        cls,
        graph: VGraph,
        *,
        max_distance=200,
        max_angle=30,
        tan_max_angle=110,
        tan_to_hyp_max_angle=90,
        pos_tolerance=25,
        check: bool = True,
        split_for_reconnections: bool = True,
    ) -> Self:
        """Create a BranchDigraph from a VGraph.

        Parameters
        ----------
        graph : VGraph
            The vascular graph.

        Returns
        -------
        Self
            The BranchDigraph instance.
        """
        graph = graph.copy()
        if split_for_reconnections:
            _, candidates = prepare_graph_for_reconnections(
                graph, max_distance=max_distance, max_angle=max_angle, av_attr="av", inplace=True
            )
            derive_tips_geometry_from_curve_geometry(graph, tangent=True, inplace=True)
        else:
            candidates = np.empty((0, 4), dtype=int)

        facing_tips = find_facing_tips(
            graph=graph,
            max_distance=max_distance,
            max_angle=max_angle,
            tan_max_angle=tan_max_angle,
            tan_to_hyp_max_angle=tan_to_hyp_max_angle,
            pos_tolerance=pos_tolerance,
            as_mask=True,
        )

        for b0, tip0, b1, tip1 in candidates:
            facing_tips[b0, tip0, b1, tip1] = True
            facing_tips[b1, tip1, b0, tip0] = True

        graph.branch_tips_connectivity_matrix(facing_tips, erase_opposite_tips=True)

        B = graph.branch_count
        Bidx = np.arange(B)
        facing_tips[Bidx, 0, Bidx, 0] = False
        facing_tips[Bidx, 1, Bidx, 1] = False

        line_list = np.argwhere(facing_tips)
        line_list = [
            line_list,
            np.stack([np.full(B, -1), np.zeros(B), np.arange(B), np.zeros(B)], axis=-1).astype(np.int_),
            np.stack([np.full(B, -1), np.zeros(B), np.arange(B), np.ones(B)], axis=-1).astype(np.int_),
        ]
        line_list = np.vstack(line_list)

        digraph = cls(graph=graph, line_list=line_list)
        if check:
            digraph.check_lines(on_invalid="warn")
        return digraph

    def compute_p_from_gt(
        self,
        art_topology: TreeTopology,
        vei_topology: TreeTopology,
        *,
        check=True,
        smooth_p=0.0,
        smooth_std=1.2,
    ) -> npt.NDArray[np.bool_]:
        """Compute the probabilities of each edge in the directed graph from a ground truth tree topology.

        Parameters
        ----------
        art_topology : TreeTopology
            The ground truth arterial tree topology.
        vei_topology : TreeTopology
            The ground truth venous tree topology.

        check : bool, optional
            If True (by default), check the consistency of the computed probabilities with the ground truth topologies, and raise a warning if inconsistencies are found.

        smooth_p : float, optional
            If greater than 0, smooth the computed probabilities by propagating them to neighboring branches in the graph. The probability of the optimal line is decreased by smooth_p and the total of probabilities from near branches sums to smooth_p.

            By default, no smoothing is applied (smooth_p=0).

        smooth_std : float, optional
            The standard deviation of the Gaussian kernel used to propagate probabilities to neighboring branches when smooth_p > 0. The distance between branches is computed as the distance of the shortest path between them in the graph.

        Returns
        -------
        npt.NDArray[np.bool_]
            An array of shape (N,) representing the optimal edges in the directed graph.
        """  # noqa: E501
        from .tree_topology import optimal_lines

        # === Read branch topologies ===
        assert self.graph is not None, (
            "The graph attribute must be set to compute probabilities from ground truth topologies"
        )
        branch_topo_a = art_topology.read_branch_topo(self.graph)
        branch_topo_v = vei_topology.read_branch_topo(self.graph)

        # === Select most plausible topology between artery and vein for each branch ===
        branch_av = highest_topo_plausibility([branch_topo_a, branch_topo_v], mask_inplace=True)
        b_is_art = branch_av == 0
        b_is_vei = branch_av == 1

        # === Compute optimal lines according to branch topologies ===
        art_lines = optimal_lines(branch_topo_a, self.line_list)
        vei_lines = optimal_lines(branch_topo_v, self.line_list)

        # === Post fix erroneous branch skips ===
        valid_art_shortcut = ~b_is_vei & (branch_topo_a.plausibility > branch_topo_v.plausibility)
        valid_vei_shortcut = ~b_is_art & (branch_topo_v.plausibility > branch_topo_a.plausibility)
        prioritize_existing_branch(self, art_lines, b_is_art, valid_art_shortcut, branch_topo_a.p_dirs)
        prioritize_existing_branch(self, vei_lines, b_is_vei, valid_vei_shortcut, branch_topo_v.p_dirs)

        # === Compute AV and dir probabilities ===
        self._branch_fp_p = np.where(b_is_art | b_is_vei, 0.0, 1.0)
        self._branch_av_p = b_is_art.astype(float)
        self._branch_fp_logit = self._branch_av_logit = None

        branch_dir_p = branch_topo_a.p_dirs * branch_topo_a.plausibility * (~b_is_vei)
        branch_dir_p += branch_topo_v.p_dirs * branch_topo_v.plausibility * (~b_is_art)
        self._branch_dir_logit = branch_dir_p * 6
        self._branch_dir_p = None

        # === Compute lines probabilities ===
        line_p = art_lines | vei_lines

        # Ensure missing branches only have not-null probability for root lines
        b0, _, b1, b1_tip = self.line_list.T
        missing_branches_lines = self.branch_fp()[b1]
        root_lines = (b0 == -1) & (b1_tip == (branch_dir_p[b1] <= 0))
        line_p[missing_branches_lines & ~root_lines] = False
        line_p[missing_branches_lines & root_lines] = True

        # === Smooth lines probabilities ===
        if smooth_p > 0:
            assert smooth_p < 0.5, "smooth_p must be inferior to 0.5"
            gt_parent = np.full((self.branch_count,), -1, dtype=int)
            gt_parent[b1[line_p]] = b0[line_p]
            dist, ca_dist = tree_distance(gt_parent)

            line_p = line_p.astype(float)

            DESC_OFFSET = 1
            for b in np.where(~self.branch_fp())[0]:
                # For each branch b, select its valid incoming lines...
                b_incoming_lines = (self.b1 == b) & (self.b1_tip == (branch_dir_p[b] <= 0))
                optimal_line = np.where(b_incoming_lines)[0][line_p[b_incoming_lines].argmax()]
                b_incoming_lines[optimal_line] = False
                if b_incoming_lines.sum() < 1:
                    continue

                # ... weight them accordingly to their distance to the optimal line
                optimal_b0, b0 = self.b0[optimal_line], self.b0[b_incoming_lines]
                # (add an offset if b0 is a descendant of optimal_b0)
                dist_b0 = dist[optimal_b0, b0] + np.where(ca_dist[optimal_b0, b0] < 0, DESC_OFFSET, 0)
                same_subtree = ~np.isnan(dist_b0)
                b_incoming_lines[b_incoming_lines] = same_subtree
                p = gaussian(dist_b0[same_subtree], sigma=smooth_std)
                p = p / p.sum() * smooth_p

                line_p[optimal_line] -= p.sum()
                line_p[b_incoming_lines] += p

        # === Final assignment and check ===
        self.line_p = line_p.astype(float)
        if check:
            self.check_line_p(on_invalid="warn")

        return line_p

    # === DIGRAPH CHECKING ===
    @overload
    def check_lines(
        self, on_invalid: Literal["raise", "warn", "ignore"] = "ignore", branch_mask: Optional[Bool1DArray] = None
    ) -> bool: ...
    @overload
    def check_lines(self, on_invalid: Literal["report"], branch_mask: Optional[Bool1DArray] = None) -> str: ...
    def check_lines(
        self,
        on_invalid: Literal["raise", "warn", "ignore", "report"] = "ignore",
        branch_mask: Optional[Bool1DArray] = None,
    ) -> str | bool:
        msg = "Invalid lines: \n"
        B = self.branch_count
        if branch_mask is None:
            b0, b0_tip, b1, b1_tip = self.line_list.T
            lines_lookup = np.arange(len(self.line_list))
            branch_mask = np.ones(B, dtype=bool)
            valid_branch = np.arange(B)
        else:
            valid_branch = np.argwhere(branch_mask).flatten()
            line_mask = branch_mask[self.line_list[:, 0]] & branch_mask[self.line_list[:, 2]]
            b0, b0_tip, b1, b1_tip = self.line_list[line_mask].T
            lines_lookup = np.arange(len(line_mask))[line_mask]

        # === Check that no line connects a branch tip to itself ===
        if len(invalid_l := np.argwhere((b0 == b1) & (b0_tip == b1_tip)).flatten()):
            invalid_l = ", ".join(str(int(lines_lookup[_])) for _ in invalid_l)
            msg += f" - lines {invalid_l} are self-loop lines;\n"

        # === Check that all branches have at least one ingoing line ===
        incoming_b = np.unique(b1)
        if len(no_parent := np.setdiff1d(valid_branch, incoming_b)):
            msg += f" - branches {', '.join(str(int(_)) for _ in no_parent)} have no ingoing line;\n"

        # === Check that all branches are accessible from the root ===
        accessible_b = accessible_from_root(np.stack([b0, b1], axis=1), B, root=-1)
        if len(inaccessible_b := np.argwhere(~accessible_b & branch_mask).flatten()) > 1:
            msg += f" - branches {', '.join(str(int(_)) for _ in inaccessible_b)} are not accessible from the root;\n"

        # === Report results ===
        if len(msg.splitlines()) > 1:
            if on_invalid == "raise":
                raise ValueError(msg)
            elif on_invalid == "warn":
                warnings.warn(msg, stacklevel=3)
            return msg if on_invalid == "report" else False
        return "" if on_invalid == "report" else True

    @overload
    def check_line_p(self, on_invalid: Literal["raise", "warn", "ignore"] = "ignore", strict: bool = True) -> bool: ...
    @overload
    def check_line_p(self, on_invalid: Literal["report"], strict: bool = True) -> str: ...
    def check_line_p(
        self, on_invalid: Literal["raise", "warn", "ignore", "report"] = "ignore", strict: bool = True
    ) -> bool | str:
        msg = "Invalid line probabilities: \n"
        if strict:
            assert self.line_p is not None, "line_p must be provided to check line probabilities"
            optimal_lines = self.line_list[self.line_p > 0.5]
        else:
            optimal_lines = self.line_list[self.max_lines()]

        if strict:
            # === Check that missing branches have no outgoing lines ===
            missing_b = self.branch_fp()
            outgoing_b = np.unique(optimal_lines[:, 0])
            if len(outgoing_b) > 0 and outgoing_b[0] == -1:
                outgoing_b = outgoing_b[1:]
            if len(invalid_outgoing_b := outgoing_b[missing_b[outgoing_b]]):
                invalid_outgoing_b = ", ".join(str(int(_)) for _ in invalid_outgoing_b)
                msg += f" - missing branches {invalid_outgoing_b} have outgoing lines;\n"

            # === Check that missing branches have only a root ingoing line ===
            not_root_b = np.unique(optimal_lines[optimal_lines[:, 0] != -1, 2])
            if len(invalid_ingoing_b := not_root_b[missing_b[not_root_b]]):
                msg += f" - missing branches {', '.join(str(int(_)) for _ in invalid_ingoing_b)} have ingoing lines;\n"

        # === Check all branches have exactly one parent ===
        incoming_b, incoming_b_count = np.unique(optimal_lines[:, 2], return_counts=True)
        if len(too_much_parent := incoming_b[incoming_b_count > 1]):
            msg += f" - branches {', '.join(str(int(_)) for _ in too_much_parent)} have more than one parent;\n"
        if len(no_parent := np.setdiff1d(np.arange(self.branch_count), incoming_b)):
            msg += f" - branches {', '.join(str(int(_)) for _ in no_parent)} have no parent;\n"

        # === Check no cycles in optimal lines ===
        optimal_parents = np.full((self.branch_count,), -1, dtype=np.int_)
        optimal_parents[optimal_lines[:, 2]] = optimal_lines[:, 0]
        if has_cycle(optimal_parents):
            cycles = ", ".join("{" + ", ".join(str(_) for _ in c) + "}" for c in find_cycles(optimal_parents))
            msg += f" - optimal parents form cycles: {cycles};\n"

        # === Check line_p and branch_dir_p consistency ===
        if (b_dir_p := self.branch_dir_p) is not None:
            b0, b0_tip, b1, b1_tip = optimal_lines.T
            valid_lines = (b0 == -1) | (b_dir_p[b0] == 0.5) | (b0_tip.astype(bool) == (b_dir_p[b0] > 0.5))
            if len(invalid_b := np.unique(optimal_lines[~valid_lines, 0])) > 0:
                invalid_b = ", ".join(str(int(_)) for _ in invalid_b)
                msg += f" - outgoing branches {invalid_b} have inconsistent direction probabilities;\n"
            valid_lines = (b_dir_p[b1] == 0.5) | (b1_tip.astype(bool) == (b_dir_p[b1] < 0.5))
            if len(invalid_b := np.unique(optimal_lines[~valid_lines, 2])) > 0:
                invalid_b = ", ".join(str(int(_)) for _ in invalid_b)
                msg += f" - ingoing branches {invalid_b} have inconsistent direction probabilities;\n"

        # === Report results ===
        if len(msg.splitlines()) > 1:
            if on_invalid == "raise":
                raise ValueError(msg)
            elif on_invalid == "warn":
                warnings.warn(msg, stacklevel=3)
            return msg if on_invalid == "report" else False
        return "" if on_invalid == "report" else True

    # === ARBORESCENCE SOLVING ===
    def max_parent(self, check_dir_consistency=False) -> npt.NDArray[np.int_]:
        """Compute for each branch the index of the parent with the highest incoming line probability.

        Returns
        -------
        npt.NDArray[np.int_]
            An array of shape (B,) storing for each branch, the index of its parent or -1 if it has no parent.
        """
        assert self.line_p is not None, "line_p must be provided to compute maximum parent"

        if check_dir_consistency:
            raise NotImplementedError("Direction consistency check is not implemented yet in max_parent computation")

        line_list = self.line_list[self.line_p.argsort()[::-1]]
        b0, _, b1, _ = line_list.T
        b1, first_idx = np.unique(b1, return_index=True)
        parent = np.full((self.branch_count,), -1, dtype=np.int_)
        parent[b1] = b0[first_idx]
        return parent

    def max_lines(self, both_direction=False) -> npt.NDArray[np.bool_]:
        """Compute which lines are the most probable amongst all lines incoming to each branch.

        Returns
        -------
        npt.NDArray[np.bool_]
            An array of shape (L,) indicating which lines are the most probable for each branch.
        """
        assert self.line_p is not None, "line_p must be provided to compute maximum parent"

        argsort = self.line_p.argsort()[::-1]
        b1 = self.b1[argsort]
        if both_direction:
            b1 += self.b1_tip[argsort] * self.branch_count
        b1, first_idx = np.unique(b1, return_index=True)
        max_lines = np.zeros((len(argsort),), dtype=bool)
        max_lines[argsort[first_idx]] = True
        return max_lines

    def solve_optimal_arboresence(
        self, *, remove_missing_branch=False, detect_major_av_error=False
    ) -> tuple[Indices, Bool1DArray]:
        """Compute the optimal arborescence of the directed graph. Missing branches are ignored in the optimization and can optionally be removed from the output.

        Parameters
        ----------
        remove_missing_branch : bool, optional
            If true, remove missing branches from the output. In this case, the returned branch indices are re-indexed to match the indices of the non-missing branches (i.e. if branch 3 is missing, then branch 4 will be re-indexed as 3 in the output).
            If false (by default), missing branches are kept in the output with a parent of -1 and with their most probable direction according to ``self.branch_dir_p``.

        Returns
        -------
        tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]
            A tuple containing:
            - An array of shape (B,) representing the parent branch of each branch in the optimal arborescence. The root branch has a parent of -1.
            - An array of shape (B,) representing the direction of each branch in the optimal arborescence: True if the branch is oriented from its first node to its second node, False otherwise.
        """  # noqa: E501
        assert self.line_p is not None, (
            "Impossible to optimize the arborescence: the probabilities of link between branches (line_p) is missing."
        )
        fp_branch = self.branch_fp()

        # Filter out lines connected to false positive branches
        if fp_branch.any():
            removal_lookup, branch_lookup = create_removal_lookup(
                fp_branch, add_empty="no increment", replace_value=-1, return_inverse=True
            )
            b0, _, b1, _ = self.line_list.T
            valid_lines = np.invert((fp_branch[b0] & (b0 != -1)) | fp_branch[b1])
            line_list = self.line_list[valid_lines].copy()
            line_list[:, 0] = removal_lookup[line_list[:, 0] + 1]
            line_list[:, 2] = removal_lookup[line_list[:, 2] + 1]
            line_p = self.line_p[valid_lines]
            dir_p = self.branch_dir_p[~fp_branch] if self.branch_dir_p is not None else None

        else:
            line_list = self.line_list
            line_p = self.line_p
            dir_p = self.branch_dir_p
            branch_lookup = None

        if line_list.shape[0] == 0:
            if remove_missing_branch:
                return np.empty((0,), dtype=np.int_), np.empty((0,), dtype=np.bool_)

            # No valid line, return trivial solution with all branches as root
            branch_parents = np.full((self.branch_count,), -1, dtype=np.int_)
            if self.branch_dir_p is None:
                branch_dir = np.ones((self.branch_count,), dtype=np.bool_)
            else:
                branch_dir = self.branch_dir_p > 0.5
            return branch_parents, branch_dir

        branch_parents, branch_dir = solve_line_digraph_approx(
            line_list=line_list,
            line_p=line_p,
            branch_dir_p=dir_p,
            ignore_branch_dir_in_MSA=dir_p is None,
        )

        if detect_major_av_error and VBranchDigraph.has_fp_av_p(self):
            av_local = 2 - self.branch_av_class()[~fp_branch]
            branch_rank = tree_node_rank(branch_parents)
            cumulative_av = np.zeros_like(branch_rank)
            for r in reversed(range(branch_rank.max() + 1)):
                rank_mask = branch_rank == r
                cumulative_av[rank_mask] += 2 * av_local[branch_rank == r] - 1
                np.add.at(cumulative_av, branch_parents[rank_mask], cumulative_av[rank_mask])

            subtree = tree_connected_components(branch_parents)
            av_subtree = np_groupby_mean(self.branch_av_logit[~fp_branch], subtree)[subtree]
            cumulative_av *= np.sign(av_subtree).astype(int)

            branch_parents[(cumulative_av < -5) & (cumulative_av[branch_parents] >= 0)] = -1

        if remove_missing_branch or branch_lookup is None:
            return branch_parents, branch_dir

        branch_parents_full = np.full((self.branch_count,), -1, dtype=np.int_)
        branch_parents_full[~fp_branch] = branch_lookup[branch_parents + 1]

        branch_dir_full = np.ones((self.branch_count,), dtype=np.bool_)
        branch_dir_full[~fp_branch] = branch_dir
        if self.branch_dir_p is not None:
            branch_dir_full[fp_branch] = self.branch_dir_p[fp_branch] > 0.5

        return branch_parents_full, branch_dir_full

    # === COMPUTE TREE FROM DIGRAPH ===
    def compute_tree_from_arborescence(
        self,
        branch_parents: Int1DArray,
        branch_dir: Bool1DArray,
        fp_branch: Optional[Bool1DArray] = None,
        *,
        keep_missing_branch: bool = False,
    ) -> VTree:
        """Resolve the directed graph into an arborescence (a directed tree).

        Returns
        -------
        VTree
            The tree representation of the directed graph.
        """
        assert self.graph is not None, "The graph attribute must be set to compute the optimized tree"

        # === Update graph according to optimal arborescence ===
        vgraph = self.graph.copy()

        if fp_branch is None:
            fp_branch = self.branch_fp()

        if not keep_missing_branch:
            # - Remove missing branches from the graph if needed
            vgraph.delete_branch(fp_branch, inplace=True)

        # - Insert branches on connections of not-adjacent branches
        added_branch_parents = np.array([], dtype=np.int_)
        for b1, b0 in enumerate(branch_parents):
            if b0 == -1:
                continue

            # If branches are not adjacent (namely if the nodes b0_head != b1_tail) ...
            b0_head = vgraph.branch_list[b0, 1 if branch_dir[b0] else 0]
            b1_tail = vgraph.branch_list[b1, 0 if branch_dir[b1] else 1]
            if b0_head != b1_tail:
                # ... check if a branch was already added
                new_b = None
                added_branches_b0 = np.argwhere(added_branch_parents == b0).flatten()
                if len(added_branches_b0):
                    n0, n1 = vgraph.branch_list[added_branches_b0].T
                    new_b = np_first_true((n0 == b0_head) & (n1 == b1_tail))

                if new_b is None:
                    # ... or insert a branch in the graph
                    new_b = vgraph.add_branch([b0_head, b1_tail], return_branch_id=True, inplace=True)[1][0]
                    added_branch_parents = np.append(added_branch_parents, b0)

                # ... update parent of b1 new_b --> b1
                branch_parents[b1] = new_b

        # === Build the final VTree ===
        branch_parents = np.hstack([branch_parents, np.array(added_branch_parents, dtype=np.int_)])
        branch_dir = np.hstack([branch_dir, np.ones(len(added_branch_parents), dtype=np.bool_)])
        tree = VTree.from_graph(vgraph, branch_parents, branch_dir, copy=False)

        return tree

    def optimize_tree(self, keep_missing_branch: bool = False) -> VTree:
        """Resolve the directed graph into an arborescence (a directed tree).

        Returns
        -------
        VTree
            The tree representation of the directed graph.
        """
        # === Solve Optimal Arborescence ===
        branch_parents, branch_dir = self.solve_optimal_arboresence(remove_missing_branch=not keep_missing_branch)
        return self.compute_tree_from_arborescence(branch_parents, branch_dir, keep_missing_branch=keep_missing_branch)

    # === UTILS ===
    def lines_info(
        self,
        b: Optional[int | Int1DArrayLike] = None,
        sort_by_p: Optional[bool] = None,
        *,
        b0: Optional[int | Int1DArrayLike] = None,
        b1: Optional[int | Int1DArrayLike] = None,
    ) -> pd.DataFrame:
        """Get the lines in the directed graph that start from a given branch.

        Parameters
        ----------
        branch : int
            The branch id.
        sort_by_p : Optional[bool], optional
            Whether to sort the lines by their probabilities, by default None.

        Returns
        -------
        npt.NDArray[np.float64]
            An array of shape (M, 4) representing the directed edges starting from the given branch.
        """
        if b is None:
            assert b0 is not None or b1 is not None, "Either b or (b0 and b1) must be provided"
            if b0 is not None:
                concerned_lines = np.isin(self.line_list[:, 0], np.asarray(b0))
            else:
                concerned_lines = np.ones(len(self.line_list), dtype=bool)
            if b1 is not None:
                concerned_lines &= np.isin(self.line_list[:, 2], np.asarray(b1))
        else:
            b = np.asarray(b)
            concerned_lines = np.isin(self.line_list[:, 0], b) | np.isin(self.line_list[:, 2], b)

        digraph = self[concerned_lines]
        data_p = {}
        if sort_by_p is None and self.line_p is not None:
            sort_by_p = True
        if VBranchDigraph.has_all_p(digraph):
            total_p = digraph.line_p + digraph.line_dir_p()
            if sort_by_p:
                sorted_ids = np.argsort(total_p)[::-1]
                total_p = total_p[sorted_ids]
                digraph = digraph[sorted_ids]

            av_p = digraph.line_av_p()
            data_p["line_p"] = digraph.line_p - av_p
            data_p["av_p"] = av_p
            data_p["total_p"] = total_p
            b0_dir_p, b1_dir_p = digraph.branch_dir_p[digraph.b0], digraph.branch_dir_p[digraph.b1]
            data_p["b0_dir_p"] = np.where(digraph.b0_dir, b0_dir_p, 1 - b0_dir_p)
            data_p["b1_dir_p"] = np.where(digraph.b1_dir, b1_dir_p, 1 - b1_dir_p)

        data = {
            "b0": digraph.b0,
            "b1": digraph.b1,
        }
        if self.graph is not None:
            data["n0"] = self.graph.branch_list[digraph.b0, digraph.b0_tip]
            data["n1"] = self.graph.branch_list[digraph.b1, digraph.b1_tip]
        else:
            data["tip0"] = digraph.b0_tip
            data["tip1"] = digraph.b1_tip

        return pd.DataFrame(data=data | data_p)


class _VBranchDigraphWithAVProba(VBranchDigraph):
    """Utility class for type checker specifying VBranchDigraph with not null branch_av_p and branch_fp_p attributes.

    This class is not meant to be instantiated!!!
    """

    @property
    def branch_fp_p(self) -> Float1DArray: ...

    @property
    def branch_fp_logit(self) -> Float1DArray: ...

    def branch_fp(self) -> Bool1DArray: ...

    @property
    def branch_av_p(self) -> Float1DArray: ...

    @property
    def branch_av_logit(self) -> Float1DArray: ...

    def branch_av_class(self) -> Int1DArray: ...

    def line_av_p(self) -> Float1DArray: ...

    @classmethod
    def check(cls, digraph: VBranchDigraph) -> TypeGuard[Self]:
        """Check if the given digraph has not null branch_fp_p and branch_av_p attributes."""
        return (digraph._branch_fp_p is not None or digraph._branch_fp_logit is not None) and (
            digraph._branch_av_p is not None or digraph._branch_av_logit is not None
        )


class _VBranchDigraphWithAllProba(_VBranchDigraphWithAVProba):
    """Utility class for type checker specifying VBranchDigraph with not null branch_av_p, branch_fp_p, branch_dir_p and line_p attributes.

    This class is not meant to be instantiated!!!
    """  # noqa: E501

    line_p: Float1DArray

    @property
    def branch_dir_p(self) -> Float1DArray: ...

    @property
    def branch_dir_logit(self) -> Float1DArray: ...

    def branch_dir(self) -> Bool1DArray: ...

    def line_dir_p(self) -> Float1DArray: ...

    @classmethod
    def check(cls, digraph: VBranchDigraph) -> TypeGuard[Self]:
        """Check if the given digraph has not null branch_dir_p, branch_dir_logit and line_p attributes."""
        return (
            super().check(digraph)
            and (digraph._branch_dir_p is not None or digraph._branch_dir_logit is not None)
            and digraph.line_p is not None
        )


########################################################################################################################
#       === DIGRAPH BUILDING UTILS ===
########################################################################################################################
def prepare_graph_for_reconnections(
    graph: VGraph,
    *,
    max_distance: float = 100,
    max_angle: float = 30,
    snap_tip_max_distance: float = 15,
    snap_tip_max_angle: float = 180,
    snap_new_node_max_distance: float = 25,
    av_attr: Optional[str] = None,
    inplace: bool = False,
) -> tuple[VGraph, npt.NDArray[np.int_]]:
    """Find reconnection candidates in the graph using a directed line graph approach.

    Parameters
    ----------
    graph : VGraph
        The input vascular graph.

    max_distance : float, optional
        The maximum distance between two endpoints or between an endpoint and a branch to consider a reconnection
        candidate, by default 100.

    max_angle : float, optional
        The maximum angle between two branches or between an endpoint and a branch to consider a reconnection
        candidate, by default 30.

    snap_tip_max_distance : float, optional
        The maximum distance between two endpoints to snap them together as a reconnection candidate, by default 30.

    snap_tip_max_angle : float, optional
        The maximum angle between two endpoints to snap them together as a reconnection candidate, by default 30.

    snap_new_node_max_distance : float, optional
        The maximum distance between two new nodes created on branches to snap them together as a single new node,
        by default 25.

    av_attr : Optional[str], optional
        The name of the branch attribute containing the artery/vein class probabilities. If provided, branches tips who are not connected to any branch of the same class will be considered as endpoints, by default None.

    Returns
    -------
    VTree
        A tree storing the best reconnection candidates.

    npt.NDArray[np.int_]
        An array of shape (N , 3) representing the reconnections between two existing endpoints or between one
        existing endpoint and a new node created on a branch. Each row contains:
        - the branch id of the existing endpoint,
        - the tip id of the existing endpoint,
        - the id of the previous or new branch to connect to
        - the tip id of the branch to connect to
    """  # noqa: E501
    import torch

    from ..utils.cpp_extensions.fvt_cpp import terminal_tips
    from .graph_simplification import find_reconnection_candidates

    if not inplace:
        graph = graph.copy()

    if av_attr is not None and av_attr in graph.branch_attr.columns:
        av = graph.branch_attr[av_attr].to_numpy()
        a_branch_mask = np.isin(av, [AVLabel.ART, AVLabel.BOTH])
        v_branch_mask = np.isin(av, [AVLabel.VEI, AVLabel.BOTH])
        subgraph_mask = torch.from_numpy(np.stack([a_branch_mask, v_branch_mask], axis=1))
    else:
        subgraph_mask = torch.empty((0, 0), dtype=torch.bool)
    endpoints = terminal_tips(
        torch.from_numpy(graph.branch_list.astype(np.int32)),
        graph.node_count,
        subgraph_mask,
    )

    candidates = find_reconnection_candidates(
        graph,
        max_distance=max_distance,
        max_angle=max_angle,
        snap_max_distance=snap_tip_max_distance,
        snap_max_angle=snap_tip_max_angle,
        endpoint_ids=endpoints.numpy()[:, 1:].astype(np.int_),
    )
    # Candidates format:  0     1   2          3         4  5  6
    #                   (b0, tip0, n1, branch_id, curve_id, y, x)
    new_node_mask = candidates[:, 2] == -1  # Node2 is a new node
    reconnections = candidates[~new_node_mask][:, [0, 1, 3, 2]].copy()  # [b0, tip0, b1, n1]
    reconnections[:, 3] = np.where(graph.branch_list[reconnections[:, 2], 0] == reconnections[:, 3], 0, 1)  # n1 -> tip1
    new_nodes_candidates = candidates[new_node_mask]

    if not len(new_nodes_candidates):
        return graph, reconnections[0]

    reconnections = [reconnections]

    if snap_new_node_max_distance <= 0:
        # Deduplicate new nodes
        new_nodes_specs, new_nodes_lookup = np.unique(new_nodes_candidates[:, 3:], axis=0, return_inverse=True)
        new_nodes_specs = np.hstack([np.arange(len(new_nodes_specs))[:, None], new_nodes_specs])

    else:
        # Snap new nodes of the same branch if they are close enough
        nodes_yx = graph.geometric_data().node_coord()
        nodes_specs = new_nodes_candidates[:, 2:]
        CANDIDATE, N1_BRANCH, N1_CURVE_ID, N1_YX = 0, 1, 2, slice(3, 5)
        nodes_specs[:, CANDIDATE] = np.arange(len(nodes_specs))
        new_nodes_lookup = np.full(len(new_nodes_candidates), -1, dtype=np.int_)

        merged_nodes_specs = []
        for b_id, branch_specs in np_group_by(nodes_specs, keys=nodes_specs[:, N1_BRANCH]):
            if len(branch_specs) == 1:
                new_id = len(merged_nodes_specs)
                merged_nodes_specs.append([new_id, *branch_specs[0, 1:]])
                new_nodes_lookup[branch_specs[:, CANDIDATE]] = new_id
                continue

            clusters = cluster_by_distance(branch_specs[:, N1_YX], snap_new_node_max_distance)
            for c in clusters:
                cluster_specs = branch_specs[c]
                new_id = len(merged_nodes_specs)
                if len(c) == 1:
                    merged_nodes_specs.append([new_id, *cluster_specs[0, 1:]])
                else:
                    b0, b0tip = new_nodes_candidates[cluster_specs[:, CANDIDATE], :2].T
                    n0 = graph.branch_list[b0, b0tip]
                    sqr_dist = np.square(nodes_yx[n0] - cluster_specs[:, N1_YX]).sum(axis=1)
                    n0_weight = softmax(-sqr_dist * 1e-3)
                    centroid_yx = (cluster_specs[:, N1_YX] * n0_weight[:, None]).sum(axis=0)
                    centroid_i = np.round((cluster_specs[:, N1_CURVE_ID] * n0_weight).sum(axis=0)).astype(np.int_)

                    merged_nodes_specs += [(new_id, b_id, centroid_i, *centroid_yx)]
                new_nodes_lookup[cluster_specs[:, CANDIDATE]] = new_id
        new_nodes_specs = np.array(merged_nodes_specs)

    LOOKUP_KEY, NEW_NODE_BRANCH, NEW_NODE_CURVE_ID, NEW_NODE_YX = 0, 1, 2, slice(3, 5)

    branch_last_tip_lookup = np.arange(graph.branch_count)
    # Split the branches at the new nodes
    branch_ids, node_specs = zip(*np_group_by(new_nodes_specs, new_nodes_specs[:, NEW_NODE_BRANCH]), strict=True)
    for b, branch_specs in zip(graph.branches(branch_ids, dynamic_iterator=True), node_specs, strict=True):
        if len(branch_specs) == 0:
            continue
        branch_specs = branch_specs[np.argsort(branch_specs[:, NEW_NODE_CURVE_ID])]  # Sort by curve index
        _, new_branch_id, new_nodes_id = graph.split_branch(
            branch_id=b.id,
            split_curve_id=branch_specs[:, NEW_NODE_CURVE_ID],
            split_coord=branch_specs[:, NEW_NODE_YX],
            return_node_ids=True,
            return_branch_ids=True,
            inplace=True,
        )
        branch_last_tip_lookup[b.id] = new_branch_id[-1]
        for lookup_key, new_node_id, adjacent_branch_id in zip(
            branch_specs[:, LOOKUP_KEY], new_nodes_id, pairwise(new_branch_id), strict=True
        ):
            b0_b0tip = new_nodes_candidates[new_nodes_lookup == lookup_key][:, :2]
            assert len(b0_b0tip) > 0, "Lookup error for new node reconnection"
            for b1 in adjacent_branch_id:
                tip1 = 0 if graph.branch_list[b1, 0] == new_node_id else 1
                reconnections += [np.hstack([b0_b0tip, np.repeat([[b1, tip1]], len(b0_b0tip), axis=0)])]

    reconnections = np.vstack(reconnections)
    last_tip_recon = reconnections[:, 1] == 1
    reconnections[last_tip_recon, 0] = branch_last_tip_lookup[reconnections[last_tip_recon, 0]]

    return graph, reconnections


def prioritize_existing_branch(
    digraph: VBranchDigraph,
    line_opti: Bool1DArray,
    active_branch: Bool1DArray,
    valid_branch: Bool1DArray,
    branch_dir: npt.NDArray[np.float32],
):
    assert digraph.graph is not None, "Provided digraph is missing its graph attribute."

    # === Redirect distant connections through existing branches if any ===
    lines_lookup = np.arange(digraph.line_list.shape[0])
    optimal_lines = digraph.line_list[line_opti]
    lines_lookup = lines_lookup[line_opti]

    not_root_lines = optimal_lines[:, 0] != -1  # Ignore root lines
    optimal_lines = optimal_lines[not_root_lines]
    lines_lookup = lines_lookup[not_root_lines]

    branch_list = digraph.graph.branch_list
    distant_lines = branch_list[*optimal_lines[:, :2].T] != branch_list[*optimal_lines[:, 2:].T]

    for distant_line_id in np.argwhere(distant_lines).flatten():
        b0, b0_tip, b1, b1_tip = optimal_lines[distant_line_id]
        n0, n1 = branch_list[b0, b0_tip], branch_list[b1, b1_tip]
        shortcut_ids = np.argwhere(
            np.all(branch_list == [n0, n1], axis=1) | np.all(branch_list == [n1, n0], axis=1)
        ).flatten()
        if len(shortcut_ids) == 1 and valid_branch[shortcut_ids[0]]:
            shortcut_id = shortcut_ids[0]
            shortcut_dir = branch_list[shortcut_id, 0] == n0
            if branch_dir[shortcut_id] != 0 and (branch_dir[shortcut_id] > 0) != shortcut_dir:
                continue

            branch_dir[shortcut_id] = 1 if shortcut_dir else -1
            active_branch[shortcut_id] = True

            s = shortcut_id
            s_tip0 = 0 if shortcut_dir else 1
            redirected_lines = digraph.search_lines([[b0, b0_tip, s, s_tip0], [s, 1 - s_tip0, b1, b1_tip]])
            line_opti[lines_lookup[distant_line_id]] = False
            line_opti[redirected_lines] = True


########################################################################################################################
#       === Edge Attribute Extractor ===
########################################################################################################################
class BaseEdgeAttrExtractor(Protocol):
    def __call__(self, digraph: VBranchDigraph) -> npt.NDArray: ...  # type: ignore


class EdgeAttrExtractor(BaseEdgeAttrExtractor):
    def __init__(
        self,
        distance: Literal["scalar", "bins", None] = "bins",
        angle: bool = True,
        calibre: Literal["scalar", "bins", None] = None,
        *,
        distance_bins: Sequence[float] = (4, 16, 64, 254),
        calibre_bins: Sequence[float] = (2, 4, 16, 32),
    ):
        assert distance in ["scalar", "bins", None], "distance must be either 'scalar', 'bins' or None"
        assert calibre in ["scalar", "bins", None], "calibre must be either 'scalar', 'bins' or None'"

        self.distance_bins = np.array(distance_bins)
        self.calibre_bins = np.array(calibre_bins)
        self.distance = distance
        self.angle = angle
        self.calibre = calibre

    def __call__(self, digraph: VBranchDigraph) -> npt.NDArray:
        lines = digraph.not_root_lines()
        assert digraph.graph is not None, "The graph attribute of the digraph must be set to compute edge attributes"
        geodata = digraph.graph.geometric_data()
        attr = []

        if self.distance is not None or self.angle:
            b0_p = geodata.tip_coord(lines.b0, lines.b0_tip == 0)
            b1_p = geodata.tip_coord(lines.b1, lines.b1_tip == 0)
            b0b1 = b0_p - b1_p
            b0b1_d = np.linalg.norm(b0b1, axis=1)

        if self.distance == "scalar":
            attr.append(b0b1_d[:, None])
        elif self.distance == "bins":
            attr.append(1 - np.clip(b0b1_d[:, None] / self.distance_bins, 0, 1))

        if self.calibre is not None:
            b0_calibre = geodata.tip_data(VBranchGeoData.Fields.TIPS_CALIBRE, lines.b0, lines.b0_tip == 0)
            b1_calibre = geodata.tip_data(VBranchGeoData.Fields.TIPS_CALIBRE, lines.b1, lines.b1_tip == 0)
            if self.calibre == "scalar":
                attr.append(b0_calibre[:, None])
                attr.append(b1_calibre[:, None])
            elif self.calibre == "bins":
                attr.append(1 - np.clip(b0_calibre[:, None] / self.calibre_bins, 0, 1))
                attr.append(1 - np.clip(b1_calibre[:, None] / self.calibre_bins, 0, 1))

        if self.angle:
            b0_t = -geodata.tip_tangent(lines.b0, lines.b0_tip == 0)
            b1_t = geodata.tip_tangent(lines.b1, lines.b1_tip == 0)
            b0_b1_t = np.zeros_like(b0_t)
            b0_b1_t[b0b1_d != 0, :] = b0b1[b0b1_d != 0] / b0b1_d[b0b1_d != 0, None]  # Avoid division by zero
            attr += [
                np.einsum("ij,ij->i", t1, t2)[:, None] for t1, t2 in [(b0_t, b0_b1_t), (b1_t, b0_b1_t), (b0_t, b1_t)]
            ]

        return np.hstack(attr)


def branch_dist_tangent_extractor(digraph: VBranchDigraph) -> npt.NDArray[np.float64]:
    assert digraph.graph is not None, "The graph attribute of the digraph must be set to compute edge attributes"

    line_list = digraph.line_list
    b0, b0_tip, b1, b1_tip = line_list[line_list[:, 0] != -1].T

    b0_p = digraph.graph.geometric_data().tip_coord(b0, b0_tip == 0)
    b1_p = digraph.graph.geometric_data().tip_coord(b1, b1_tip == 0)
    b0b1 = b0_p - b1_p
    b0b1_d = np.linalg.norm(b0b1, axis=1)

    b0_t = -digraph.graph.geometric_data().tip_tangent(b0, b0_tip == 0)
    b1_t = digraph.graph.geometric_data().tip_tangent(b1, b1_tip == 0)
    b0_b1_t = np.zeros_like(b0_t)
    b0_b1_t[b0b1_d != 0, :] = b0b1[b0b1_d != 0] / b0b1_d[b0b1_d != 0, None]  # Avoid division by zero

    # Smooth one hot encoding of the distance b0->b1 in 4 bins:
    dist_f = 1 - np.clip(b0b1_d[:, None] / np.array([4, 16, 64, 256]), 0, 1)
    tan_f = [np.einsum("ij,ij->i", t1, t2)[:, None] for t1, t2 in [(b0_t, b0_b1_t), (b1_t, b0_b1_t), (b0_t, b1_t)]]
    return np.hstack([dist_f] + tan_f)


########################################################################################################################
#       === DIGRAPH SOLVING UTILS ===
########################################################################################################################
def solve_line_digraph_approx(
    line_list: npt.NDArray[np.int_] | LineDigraph,
    line_p: npt.NDArray[np.float64],
    branch_dir_p: Optional[npt.NDArray[np.float64]] = None,
    ignore_branch_dir_in_MSA: bool = False,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
    """Resolve the directed graph into an arborescence (a directed tree).

    Parameters
    ----------
    graph : VGraph
        The vascular graph.
    line_list : npt.NDArray[np.int_]
        An array of shape (N, 4) representing the directed edges connecting the branch b0 to the branch b1. Each row is in the format ``(b0, b0_tip, b1, b1_tip)``, where ``b0_tip`` and ``b1_tip`` are in {0, 1} indicates if the branches are connected through their first (0) or second (1) node.
        (Namely: ``graph.branch_list[b0,b0_tip]`` and ``graph.branch_list[b1,b1_tip]``).
        The number of branches B is inferred as ``line_list.max() + 1``.
    line_p : Optional[npt.NDArray[np.float_]], optional
        An array of shape (N,) representing the probabilities of each edge, by default None.
    branch_dir_p : Optional[npt.NDArray[np.float_]], optional
        An array of shape (B,) representing the direction probabilities of each branch, by default None
    ignore_branch_dir : bool, optional
        Whether to ignore the branch direction i.e. a branch can be both a parent and a daughter at a single node.
    Returns
    -------
    tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]
        The branch parents and branch directions as arrays of shape (B,).
    """  # noqa: E501
    import networkx as nx
    from networkx.algorithms.tree.branchings import maximum_spanning_arborescence

    line_digraph = LineDigraph(line_list=line_list) if not isinstance(line_list, LineDigraph) else line_list
    line_list = line_digraph.line_list
    assert line_p.ndim == 1 and line_list.shape[0] == line_p.shape[0], (
        "line_p must be a 1D array of the same length as line_list"
    )

    B = line_list.max() + 1 if branch_dir_p is None else branch_dir_p.shape[0]

    if not ignore_branch_dir_in_MSA:
        assert branch_dir_p is not None, "branch_dir_p must be provided if ignore_branch_dir is False"
        assert branch_dir_p.ndim == 1 and branch_dir_p.shape[0] >= B, (
            "branch_dir_p must be a 1D array of length at least the number of branches in line_list"
        )

        # === Build the directed graph with both directions for each branch ===
        digraph = nx.DiGraph()
        line_total_p = np.zeros(len(line_list), dtype=float)
        for id, (line, p) in enumerate(zip(line_list, line_p, strict=True)):
            # → Unpack line: (source branch, target branch, source tip, target tip)
            b0, b0_tip, b1, b1_tip = line

            b0_reversed = b0_tip == 0  # Source branch is reversed if tip is 0
            b1_reversed = b1_tip == 1  # Target branch is reversed if tip is 1

            # → Add direction probability to the edge probability
            if b0 != -1:
                p_dir = branch_dir_p[b0] if not b0_reversed else 1 - branch_dir_p[b0]
                p_dir += branch_dir_p[b1] if not b1_reversed else 1 - branch_dir_p[b1]
                p_dir *= 0.5
            else:
                p_dir = branch_dir_p[b1] if not b1_reversed else 1 - branch_dir_p[b1]
            p += p_dir

            # → Shift branch ids and encode direction in the sign
            b0 = b0 + 1 if not b0_reversed else -(b0 + 1)
            b1 = b1 + 1 if not b1_reversed else -(b1 + 1)

            if (already_added := digraph.edges.get((b0, b1), None)) is not None and already_added["p"] >= p:
                continue
            digraph.add_edge(b0, b1, p=p, id=id)
            line_total_p[id] = p

        # === Solve the double optimal tree (using both direction for each branch) ===
        try:
            optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)
        except nx.NetworkXException as e:
            warnings.warn(f"Error while presolving the optimal tree: {e}", stacklevel=2)
            optimal_tree = maximum_branching(digraph, attr="p", preserve_attrs=True)

        # === Build the simplified directed graph (merging both directions of each branch) ===
        digraph = nx.DiGraph()
        for B0, B1, data in optimal_tree.edges(data=True):
            b0_tip = 1 if B0 > 0 else 0  # Head tip if source branch is not reversed else tail tip
            b1_tip = 0 if B1 > 0 else 1  # Tail tip if target branch is not reversed else head tip
            b0, b1 = abs(B0) - 1, abs(B1) - 1  # Remove direction encoding

            if (already_added := digraph.edges.get((b0, b1), None)) is not None and already_added["p"] >= data["p"]:
                continue
            digraph.add_edge(b0, b1, p=data["p"], id=data["id"], tips=[b0_tip, b1_tip])
        invalid_roots = [_ for _, in_degree in digraph.in_degree() if in_degree == 0 and _ != -1]
        if len(invalid_roots):
            warnings.warn(f"After presolving the optimal tree: {invalid_roots} were not rooted properly.", stacklevel=2)
            for root_B in invalid_roots:
                root_b = abs(root_B) - 1
                root_tip = 0 if root_B > 0 else 1
                p = branch_dir_p[root_b] if root_tip == 1 else 1 - branch_dir_p[root_b]
                lines = line_digraph.search_lines([-1, 0, root_b, root_tip])
                if len(lines):
                    p += lines.argmax()
                digraph.add_edge(
                    -1, root_b, p=p, id=-1, tips=[0, root_tip]
                )  # Add a dummy root edge for branches without parent

    else:
        digraph = nx.DiGraph()
        for id, (line, p) in enumerate(zip(line_list, line_p, strict=True)):
            digraph.add_edge(*line[:2], p=p, id=id, tips=line[2:])

    # === Solve the simplified directed tree (ignoring branch direction constraints) ===
    try:
        optimal_tree = maximum_spanning_arborescence(digraph, attr="p", preserve_attrs=True)
    except nx.NetworkXException as e:
        if ignore_branch_dir_in_MSA:
            raise e
        else:
            warnings.warn(
                f"Impossible to solve the simplified optimal tree: {e}. \n Fallback to maximum branching.",
                stacklevel=2,
            )
            try:
                optimal_tree = maximum_branching(digraph, attr="p", preserve_attrs=True)
            except nx.NetworkXException as e:
                warnings.warn(
                    f"Impossible to solve the maximum branching over the simplified optimal tree: {e}. \n"
                    "Fall back to single step solving.",
                    stacklevel=2,
                )
                return solve_line_digraph_approx(line_list, line_p, branch_dir_p, True)

    # === Clean the MSA to prevent rebound ===
    branch_tree = -np.ones(B, dtype=np.int_)
    branch_dir = np.empty(B, dtype=np.bool_)
    incoming_tip = np.empty(B, dtype=np.int_)

    for b0, b1 in nx.edge_bfs(optimal_tree, -1):
        data = optimal_tree[b0][b1]
        b0_tip, b1_tip = data["tips"]

        # Check for rebound of b0: if b0 is already a child of n1 redirect it to its parent
        if b0 != -1:
            if incoming_tip[b0] == b0_tip:  # → b0_tip is both the incoming and the outgoing tip ...
                b0 = branch_tree[b0]  # ... change b0 to its parent branch
                b0_tip = 1 - incoming_tip[b0]  # Update b0_tip accordingly

        branch_tree[b1] = b0
        incoming_tip[b1] = b1_tip
        branch_dir[b1] = b1_tip == 0  # Direction is True if incoming tip is 0 (tail)

    return branch_tree, branch_dir
