import warnings
from typing import Optional, Self

import numpy as np
import numpy.typing as npt

from ..utils.cluster import cluster_by_distance
from ..utils.math import sigmoid, softmax
from ..utils.numpy import np_group_by
from ..utils.typing import Int2DArrayLike
from ..vascular_data_objects import VGraph
from ..vascular_data_objects.fundus_data import AVLabel
from ..vascular_data_objects.vbranch_geodata import VBranchGeoData
from ..vascular_data_objects.vtree import VTree
from .geometry_parsing import derive_tips_geometry_from_curve_geometry
from .graph_simplification import find_facing_tips
from .tree_topology import TreeTopology


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

    @classmethod
    def search_lines(cls, lines: npt.NDArray[np.int_], searched_lines: Int2DArrayLike) -> npt.NDArray[np.int_]:
        """Search for lines in the directed graph.

        Parameters
        ----------
        lines : npt.NDArray[np.int_]
            A list of lines in the format (l0, l0_tip, l1, l1_tip).
        searched_lines : Int2DArrayLike
            A list of M lines to search for in the format (l0, l0_tip, l1, l1_tip).

        Returns
        -------
        npt.NDArray[np.int_]
            An array of shape (M,) representing the indices of the searched lines in the directed graph.
        """
        searched_lines = np.asarray(searched_lines, dtype=np.int_)
        if searched_lines.ndim == 1:
            searched_lines = searched_lines[None, :]

        lines_ids = np.full(len(searched_lines), -1, dtype=np.int_)
        for i, line in enumerate(searched_lines):
            match = np.argwhere(np.all(lines == line, axis=1)).flatten()
            if len(match) > 0:
                lines_ids[i] = match[0]

        return lines_ids


class VBranchDigraph(LineDigraph):
    def __init__(
        self,
        graph: VGraph,
        line_list: npt.NDArray[np.int_],
        line_p: Optional[npt.NDArray[np.float64]] = None,
        branch_dir_p: Optional[npt.NDArray[np.float64]] = None,
        branch_av_p: Optional[npt.NDArray[np.float64]] = None,
    ):
        """A directed graph representing possible reconnections between branches.

        Parameters
        ----------
        graph : VGraph
            The vascular graph.
        line_list : npt.NDArray[np.int_]
            An array of shape (N, 4) representing the directed edges connecting the branch b0 to the branch b1. Each row is in the format ``(b0, b1, b0_tip, b1_tip)``, where ``b0_tip`` and ``b1_tip`` are in {0, 1} indicates if the branches are connected through their first (0) or second (1) node.
            (Namely: ``graph.branch_list[b0,b0_tip]`` and ``graph.branch_list[b1,b1_tip]``).
        line_p : Optional[npt.NDArray[np.float_]], optional
            An array of shape (N,) representing the probabilities of each edge, by default None.
        branch_dir_p : Optional[npt.NDArray[np.float_]], optional
            An array of shape (B,) representing the direction probabilities of each branch, by default None
        branch_av_p : Optional[npt.NDArray[np.float_]], optional
            An array of shape (B,2) representing the class probabilities (artery/vein) of each branch, by default None.
            If, for a given branch, the 1- (pArt+pVein) is inferior to both pArt and pVein, then the branch is considered invalid.
        """  # noqa: E501
        super().__init__(line_list=line_list)
        self.graph = graph
        self.line_p: Optional[npt.NDArray[np.float64]] = line_p
        self.branch_dir_p: Optional[npt.NDArray[np.float64]] = branch_dir_p
        self.branch_av_p: Optional[npt.NDArray[np.float64]] = branch_av_p

    def branch_av(self) -> npt.NDArray[np.int_]:
        """Get the artery/vein class of each branch.

        Returns
        -------
        npt.NDArray[np.int_]
            An array of shape (B,) representing the class of each branch: 0 for invalid, 1 for artery, 2 for vein.
        """
        assert self.branch_av_p is not None, "branch_av_p must be provided to compute branch classes"
        av_p = np.hstack([1 - self.branch_av_p.sum(axis=1, keepdims=True), self.branch_av_p])
        return np.argmax(av_p, axis=1).astype(np.int_)

    def invalid_branch(self) -> npt.NDArray[np.bool_]:
        """Get a mask of invalid branches.

        Returns
        -------
        npt.NDArray[np.bool_]
            An array of shape (B,) representing whether each branch is invalid.
        """
        if self.branch_av_p is None:
            return np.ones(self.graph.branch_count, dtype=bool)
        return 1 - self.branch_av_p.sum(axis=1) >= self.branch_av_p.max(axis=1)

    @classmethod
    def from_graph(cls, graph: VGraph, *, max_distance=200, max_angle=30, tan_max_angle=110, pos_tolerance=25) -> Self:
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
        graph.geometric_data().clear_attribute(
            all_except={VBranchGeoData.Fields.TANGENTS, VBranchGeoData.Fields.TIPS_TANGENT}
        )
        _, candidates = prepare_graph_for_reconnections(
            graph, max_distance=max_distance, max_angle=max_angle, av_attr="av", inplace=True
        )
        derive_tips_geometry_from_curve_geometry(graph, tangent=True, inplace=True)

        facing_tips = find_facing_tips(
            graph,
            max_distance=max_distance,
            max_angle=max_angle,
            tan_max_angle=tan_max_angle,
            pos_tolerance=pos_tolerance,
            as_mask=True,
        )
        for b0, tip0, n1 in candidates:
            node = graph.node(n1)
            b1 = np.array(node.adjacent_branch_ids)
            tip1 = np.where(node.adjacent_branches_first_node, 0, 1)
            facing_tips[b0, tip0, b1, tip1] = True
            facing_tips[b1, tip1, b0, tip0] = True
        graph.branch_tips_connectivity_matrix(facing_tips)

        line_list = np.argwhere(facing_tips)

        B = graph.branch_count
        line_list = [
            line_list,
            np.stack([np.full(B, -1), np.zeros(B), np.arange(B), np.zeros(B)], axis=-1).astype(np.int_),
            np.stack([np.full(B, -1), np.zeros(B), np.arange(B), np.ones(B)], axis=-1).astype(np.int_),
        ]
        line_list = np.vstack(line_list)

        digraph = cls(graph=graph, line_list=line_list)
        return digraph

    def compute_p_from_gt(self, art_topology: TreeTopology, vei_topology: TreeTopology) -> None:
        """Compute the probabilities of each edge in the directed graph from a ground truth tree topology.

        Parameters
        ----------
        art_topology : TreeTopology
            The ground truth arterial tree topology.
        vei_topology : TreeTopology
            The ground truth venous tree topology.
        """
        from .tree_topology import optimal_lines

        art_lines, art_branch_dir, art_plausibility = optimal_lines(self.graph, art_topology, self.line_list)
        vei_lines, vei_branch_dir, vei_plausibility = optimal_lines(self.graph, vei_topology, self.line_list)

        # === Mark branches as invalid if they are in both tree ===
        art_b, vei_b = art_plausibility > 0, vei_plausibility > 0
        both_branch = art_b & vei_b
        art_invalid = both_branch & (vei_plausibility + 0.15 > art_plausibility)
        vei_invalid = both_branch & (art_plausibility + 0.15 > vei_plausibility)

        art_b[art_invalid] = False
        vei_b[vei_invalid] = False
        self.branch_av_p = np.stack([art_b, vei_b], axis=1).astype(float)
        # Transfer the line
        B = self.graph.branch_count

        def transfer_line_p_to_parent(lines, line_opti, invalid_branch, branch_dir):
            branch_parent = np.full(B, -1, dtype=np.int_)
            optimal_lines = lines[line_opti]
            branch_parent[optimal_lines[:, 2]] = optimal_lines[:, 0]
            for b in np.argwhere(invalid_branch).flatten():
                branch_parent[branch_parent == b] = branch_parent[b]

            invalid_lines = line_opti & (invalid_branch[lines[:, 0]] | invalid_branch[lines[:, 2]])

            for b1, b1_dir in lines[invalid_lines, 2:]:
                if invalid_branch[b1]:
                    continue
                new_b1_parent = int(branch_parent[b1])
                redirected_line = LineDigraph.search_lines(
                    lines, [new_b1_parent, branch_dir[new_b1_parent] > 0, b1, b1_dir]
                )
                if redirected_line == -1:
                    redirected_line = LineDigraph.search_lines(lines, [-1, 0, b1, b1_dir])
                line_opti[redirected_line] = True
            line_opti[invalid_lines] = False

        transfer_line_p_to_parent(self.line_list, art_lines, art_invalid, art_branch_dir)
        transfer_line_p_to_parent(self.line_list, vei_lines, vei_invalid, vei_branch_dir)

        # === Convert optimal mask to probabilities ===
        self.line_p = (art_lines | vei_lines).astype(float)
        branch_dir_p = art_branch_dir * art_plausibility + vei_branch_dir * vei_plausibility
        self.branch_dir_p = sigmoid(branch_dir_p * 6)

    def optimize_tree(self, keep_invalid_branch: bool = False) -> VTree:
        """Resolve the directed graph into an arborescence (a directed tree).

        Returns
        -------
        VTree
            The tree representation of the directed graph.
        """
        assert self.line_p is not None, (
            "Impossible to resolve the tree: the probabilities of link between branches (line_p) is missing."
        )

        # === Ignore invalid branches ===
        invalid_branch = self.invalid_branch()
        B_inv = int(np.sum(invalid_branch))

        # Filter out lines connected to invalid branches
        invalid_lines = invalid_branch[self.line_list[:, 0]] | invalid_branch[self.line_list[:, 2]]
        line_list = self.line_list[np.invert(invalid_lines)]
        line_p = self.line_p[np.invert(invalid_lines)]

        # Add dummy lines from root to invalid branch
        if self.branch_dir_p is not None:
            dummy_lines = [
                np.repeat([[-1, 0]], B_inv, axis=0),
                np.argwhere(invalid_branch),
                self.branch_dir_p[invalid_branch, None] < 0.5,
            ]
            line_list = np.vstack([line_list, np.hstack(dummy_lines)])
            line_p = np.concatenate([line_p, np.ones(B_inv)])
            dir_p = np.concatenate([self.branch_dir_p[~invalid_branch], self.branch_dir_p[invalid_branch]])
        else:
            dir_p = None
            dummy_lines = [
                np.repeat([[-1, 0]], B_inv, axis=0),
                np.argwhere(invalid_branch),
                np.ones(B_inv),
            ]
            line_list = np.vstack([line_list, np.hstack(dummy_lines)])
            line_p = np.concatenate([line_p, np.ones(B_inv)])

        # === Solve the directed graph into an arborescence ===
        branch_parents, branch_dir = solve_line_digraph_approx(
            line_list=line_list,
            line_p=line_p,
            branch_dir_p=dir_p,
            ignore_branch_dir_in_MSA=dir_p is None,
        )

        vgraph = self.graph.copy()

        # === Insert branches on connections of not-adjacent branches ===
        added_branch_parents = []
        for b1, b0 in enumerate(branch_parents):
            if b0 == -1:
                continue

            # If branches are not adjacent (namely if the nodes b0_head != b1_tail) ...
            b0_head = vgraph.branch_list[b0, 1 if branch_dir[b0] else 0]
            b1_tail = vgraph.branch_list[b1, 0 if branch_dir[b1] else 1]
            if b0_head != b1_tail:
                # ... insert a branch in the graph
                new_b = vgraph.add_branch([b0_head, b1_tail], return_branch_id=True, inplace=True)[1][0]
                assert new_b == len(branch_parents) + len(added_branch_parents), "Unexpected branch id"

                # ... update parent of b1 and new_b so that b0 -> new_b --> b1
                branch_parents[b1] = new_b
                added_branch_parents.append(b0)

        # === Build the final VTree ===
        branch_parents = np.hstack([branch_parents, np.array(added_branch_parents, dtype=np.int_)])
        branch_dir = np.hstack([branch_dir, np.ones(len(added_branch_parents), dtype=np.bool_)])
        tree = VTree.from_graph(vgraph, branch_parents, branch_dir, copy=False)

        # === Optionally remove invalid branches from the tree ===
        if not keep_invalid_branch and B_inv > 0:
            tree.delete_branch(np.where(invalid_branch)[0], inplace=True)

        return tree

    def lines_by_branch(self, branch: int, sort_by_p: Optional[bool] = None) -> npt.NDArray[np.float64]:
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
        concerned_lines = (self.line_list[:, 0] == branch) | (self.line_list[:, 2] == branch)
        lines = self.line_list[concerned_lines]
        if sort_by_p is None and self.line_p is not None:
            sort_by_p = True
        if sort_by_p and self.line_p is not None:
            p = self.line_p[concerned_lines]
            total_p = p.copy()
            if self.branch_dir_p is not None:
                b0, b0_tip, b1, b1_tip = lines.T
                b0_reversed = b0_tip == 0  # Source branch is reversed if tip is 0
                b1_reversed = b1_tip == 1  # Target branch is reversed if tip is 1
                b0_dir_p = np.where(~b0_reversed, self.branch_dir_p[b0], 1 - self.branch_dir_p[b0])
                b1_dir_p = np.where(~b1_reversed, self.branch_dir_p[b1], 1 - self.branch_dir_p[b1])
                line_dir_p = np.where(b0 != -1, (b0_dir_p + b1_dir_p) / 2, b1_dir_p)
                total_p += line_dir_p
            else:
                b0_dir_p = b1_dir_p = np.array([])
            sorted_ids = np.argsort(total_p)[::-1]
            lines = lines[sorted_ids]
            lines[:, 1] = self.graph.branch_list[lines[:, 0], lines[:, 1]]
            lines[:, 3] = self.graph.branch_list[lines[:, 2], lines[:, 3]]
            lines = np.concatenate([lines, p[sorted_ids][:, None]], axis=1)
            if self.branch_dir_p is not None:
                lines = np.concatenate([lines, b0_dir_p[sorted_ids][:, None], b1_dir_p[sorted_ids][:, None]], axis=1)
        return lines.astype(np.float64)


########################################################################################################################
#       === DIGRAPH BUILDING UTILS ===
########################################################################################################################
def prepare_graph_for_reconnections(
    graph: VGraph,
    *,
    max_distance: float = 100,
    max_angle: float = 30,
    snap_tip_max_distance: float = 30,
    snap_tip_max_angle: float = 30,
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
        - the id of the existing or new node to connect to
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
    reconnections = [candidates[~new_node_mask][:, :3]]
    new_nodes_candidates = candidates[new_node_mask]

    if not len(new_nodes_candidates):
        return graph, reconnections[0]

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
        for lookup_key, new_node_id in zip(branch_specs[:, LOOKUP_KEY], new_nodes_id, strict=True):
            b0_b0tip = new_nodes_candidates[new_nodes_lookup == lookup_key][:, :2]
            assert len(b0_b0tip) > 0, "Lookup error for new node reconnection"
            reconnections += [np.hstack([b0_b0tip, np.full((len(b0_b0tip), 1), new_node_id)])]

    reconnections = np.vstack(reconnections)
    last_tip_recon = reconnections[:, 1] == 1
    reconnections[last_tip_recon, 0] = branch_last_tip_lookup[reconnections[last_tip_recon, 0]]

    return graph, reconnections


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

    B = line_list.max() + 1

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
            optimal_tree = digraph

        # === Build the simplified directed graph (merging both directions of each branch) ===
        digraph = nx.DiGraph()
        for B0, B1, data in optimal_tree.edges(data=True):
            b0_tip = 1 if B0 > 0 else 0  # Head tip if source branch is not reversed else tail tip
            b1_tip = 0 if B1 > 0 else 1  # Tail tip if target branch is not reversed else head tip
            b0, b1 = abs(B0) - 1, abs(B1) - 1  # Remove direction encoding

            if (already_added := digraph.edges.get((b0, b1), None)) is not None and already_added["p"] >= data["p"]:
                continue
            digraph.add_edge(b0, b1, p=data["p"], id=data["id"], tips=[b0_tip, b1_tip])

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
                f"Impossible to solve the simplified optimal tree: {e}. \n Fallback to single step solving.",
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
