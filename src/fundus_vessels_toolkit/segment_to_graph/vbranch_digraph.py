import warnings
from typing import Optional, Self

import numpy as np
import numpy.typing as npt

from ..utils.cluster import cluster_by_distance
from ..utils.numpy import np_group_by
from ..vascular_data_objects import VGraph
from ..vascular_data_objects.vgraph import BranchIndicesLike, NodeIndicesLike
from ..vascular_data_objects.vtree import VTree
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
            An array of shape (N, 4) representing the directed edges connecting the line segment l0 to the line segment l1. Each row is in the format ``(l0, l1, l0_tip, l1_tip)``, where ``l0_tip`` and ``l1_tip`` are in {0, 1} indicates if the line segments are connected through their first (0) or second (1) node.
            (Namely: ``line_list[l0,l0_tip]`` and ``line_list[l1,l1_tip]``).
        """  # noqa: E501
        assert line_list.ndim == 2 and line_list.shape[1] == 4, "line_list must be of shape (N, 4)"
        self.line_list = line_list


class VBranchDigraph(LineDigraph):
    def __init__(
        self,
        graph: VGraph,
        line_list: npt.NDArray[np.int_],
        line_p: Optional[npt.NDArray[np.float_]] = None,
        branch_dir_p: Optional[npt.NDArray[np.float_]] = None,
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
        """  # noqa: E501
        super().__init__(line_list=line_list)
        self.graph = graph
        self.line_p = line_p
        self.branch_dir_p = branch_dir_p

    def optimize_tree(self) -> VTree:
        """Resolve the directed graph into an arborescence (a directed tree).

        Returns
        -------
        VTree
            The tree representation of the directed graph.
        """
        assert self.line_p is not None, (
            "Impossible to resolve the tree: the probabilities of link between branches (line_p) is missing."
        )

        # === Solve the directed graph into an arborescence ===
        branch_parents, branch_dir = solve_line_digraph_approx(
            line_list=self.line_list,
            line_p=self.line_p,
            branch_dir_p=self.branch_dir_p,
            ignore_branch_dir_in_MSA=self.branch_dir_p is None,
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
        return VTree.from_graph(vgraph, branch_parents, branch_dir, copy=False)

    @classmethod
    def from_graph(cls, graph: VGraph, max_distance=100, max_angle=30) -> Self:
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
        graph, _ = prepare_graph_for_reconnections(graph, max_distance=max_distance, max_angle=max_angle, inplace=False)
        line_list = find_facing_tips(graph, max_distance=max_distance, max_angle=max_angle)
        return cls(graph=graph, line_list=line_list)

    def compute_p_from_gt(self, art_topology: TreeTopology, vei_topology: TreeTopology) -> None:
        """Compute the probabilities of each edge in the directed graph from a ground truth tree topology.

        Parameters
        ----------
        topology : TreeTopology
            The ground truth tree topology.
        """
        from .branch_directionality import compute_branch_directionality_p
        from .line_connection_probability import compute_line_connection_p

        self.line_p = compute_line_connection_p(self.graph, self.line_list, topology)
        self.branch_dir_p = compute_branch_directionality_p(self.graph, topology)


########################################################################################################################
#       === DIGRAPH BUILDING UTILS ===
########################################################################################################################
def prepare_graph_for_reconnections(
    graph: VGraph,
    *,
    max_distance: float = 100,
    max_angle: float = 30,
    end_max_angle: Optional[float] = 20,
    snap_tip_max_distance: float = 30,
    snap_tip_max_angle: float = 30,
    snap_new_node_max_distance: float = 25,
    endpoint_ids: Optional[NodeIndicesLike] = None,
    branch_ids: Optional[BranchIndicesLike] = None,
    inplace: bool = False,
) -> tuple[VGraph, npt.NDArray[np.int_]]:
    """Find reconnection candidates in the graph using a directed line graph approach.

    Parameters
    ----------
    graph : VGraph
        The input vascular graph.

    Returns
    -------
    VTree
        A tree storing the best reconnection candidates.

    npt.NDArray[np.int_]
        An array of shape (N, 2) representing the reconnections made. Each row is in the format ``(node1, node2)``, where ``node1`` and ``node2`` are the ids of the nodes to connect.
    """  # noqa: E501
    from .graph_simplification import find_reconnection_candidates

    if not inplace:
        graph = graph.copy()

    candidates = find_reconnection_candidates(
        graph,
        max_distance=max_distance,
        max_angle=max_angle,
        end_max_angle=end_max_angle,
        snap_max_distance=snap_tip_max_distance,
        snap_max_angle=snap_tip_max_angle,
        endpoint_ids=endpoint_ids,
        branch_ids=branch_ids,
    )
    # Candidates format:    0     1        2         3      4  5
    #                   (node1, node2, branch_id, curve_id, y, x)
    new_node_mask = candidates[:, 1] == -1  # Node2 is a new node
    reconnections = [candidates[~new_node_mask][:, :2]]
    new_nodes = candidates[new_node_mask]

    if not len(new_nodes):
        return graph, reconnections[0]

    # Deduplicate new nodes
    new_nodes_specs, new_nodes_lookup = np.unique(new_nodes[:, 2:], axis=0, return_inverse=True)
    new_nodes_specs = np.hstack([np.arange(len(new_nodes_specs))[:, None], new_nodes_specs])
    NEW_NODE_ID, NEW_NODE_BRANCH, NEW_NODE_CURVE_ID, NEW_NODE_YX = 0, 1, 2, slice(3, 5)

    if snap_new_node_max_distance > 0:
        # Snap new nodes of the same branch if they are close enough
        merged_nodes_specs = []
        for b_id, nodes in np_group_by(new_nodes_specs, keys=new_nodes_specs[:, NEW_NODE_BRANCH]):
            if len(nodes) <= 1:
                merged_nodes_specs.append(nodes[0])
                continue

            clusters = cluster_by_distance(nodes[:, NEW_NODE_YX], snap_new_node_max_distance)
            for c in clusters:
                if len(c) == 1:
                    merged_nodes_specs.append(nodes[c[0]])
                else:
                    merged_node_id = np.min(nodes[c, NEW_NODE_ID])
                    centroid_yx = nodes[c, NEW_NODE_YX].mean(axis=0)
                    centroid_i = np.round(nodes[c, NEW_NODE_CURVE_ID].mean()).astype(np.int_)
                    merged_nodes_specs += [(merged_node_id, b_id, centroid_i, *centroid_yx)]

                    new_nodes_lookup[np.isin(new_nodes_lookup, nodes[c, NEW_NODE_ID])] = merged_node_id
        new_nodes_specs = np.array(merged_nodes_specs)

    # Split the branches at the new nodes
    branch_ids, node_specs = zip(*np_group_by(new_nodes_specs, new_nodes_specs[:, NEW_NODE_BRANCH]), strict=True)
    for b, nodes in zip(graph.branches(branch_ids, dynamic_iterator=True), node_specs, strict=True):
        if len(nodes) == 0:
            continue
        nodes = nodes[np.argsort(nodes[:, NEW_NODE_CURVE_ID])]  # Sort by curve index
        _, new_nodes_id = graph.split_branch(
            branch_id=b.id,
            split_curve_id=nodes[:, NEW_NODE_CURVE_ID],
            split_coord=nodes[:, NEW_NODE_YX],
            return_node_ids=True,
            inplace=True,
        )
        for node1_id, new_node_id in zip(nodes[:, NEW_NODE_ID], new_nodes_id, strict=True):
            node1_ids = new_nodes[new_nodes_lookup == node1_id][:, 0]
            reconnections += [np.hstack([node1_ids[:, None], np.full((len(node1_ids), 1), new_node_id)])]

    return graph, np.vstack(reconnections)


########################################################################################################################
#       === DIGRAPH SOLVING UTILS ===
########################################################################################################################
def solve_line_digraph_approx(
    line_list: npt.NDArray[np.int_] | LineDigraph,
    line_p: npt.NDArray[np.float_],
    branch_dir_p: Optional[npt.NDArray[np.float_]] = None,
    ignore_branch_dir_in_MSA: bool = False,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
    """Resolve the directed graph into an arborescence (a directed tree).

    Parameters
    ----------
    graph : VGraph
        The vascular graph.
    line_list : npt.NDArray[np.int_]
        An array of shape (N, 4) representing the directed edges connecting the branch b0 to the branch b1. Each row is in the format ``(b0, b1, b0_tip, b1_tip)``, where ``b0_tip`` and ``b1_tip`` are in {0, 1} indicates if the branches are connected through their first (0) or second (1) node.
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
            b0, b1, b0_tip, b1_tip = line

            b0_reversed = b0_tip == 0  # Source branch is reversed if tip is 0
            b1_reversed = b1_tip == 1  # Target branch is reversed if tip is 1

            # → Shift branch ids and encode direction in the sign
            b0 = b0 + 1 if not b0_reversed else -(b0 + 1)
            b1 = b1 + 1 if not b1_reversed else -(b1 + 1)

            if b0 != -1:
                p += branch_dir_p[b0] if not b0_reversed else 1 - branch_dir_p[b0]
                p += branch_dir_p[b1] if not b1_reversed else 1 - branch_dir_p[b1]
            else:
                p += 2 * (branch_dir_p[b1] if not b1_reversed else 1 - branch_dir_p[b1])

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
        if not ignore_branch_dir_in_MSA:
            raise e
        else:
            warnings.warn(
                f"Impossible to solve the simplified optimal tree: {e}. \n Fallback to single step solving.",
                stacklevel=2,
            )
            return solve_line_digraph_approx(line_list, line_p, branch_dir_p, True)

    # === Clean the MSA to prevent rebound ===
    branch_tree = np.empty(B, dtype=np.int_)
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
