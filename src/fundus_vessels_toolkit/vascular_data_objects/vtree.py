from __future__ import annotations

__all__ = ["VTree"]

from typing import Dict, Generator, Iterable, List, Literal, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.lookup_array import add_empty_to_lookup, complete_lookup, invert_complete_lookup
from ..utils.math import as_1d_array
from ..utils.tree import has_cycle
from .vgeometric_data import VBranchGeoDataKey, VGeometricData
from .vgraph import VGraph, VGraphBranch, VGraphNode


########################################################################################################################
#  === TREE NODES AND BRANCHES ACCESSORS CLASSES ===
########################################################################################################################
class VTreeNode(VGraphNode):
    def __init__(self, graph: VTree, id: int, *, source_branch_id: Optional[int] = None):
        super().__init__(graph, id)
        self._incoming_branches_id = None
        self._outgoing_branches_id = None

        #: The index of the branch that emitted this node.
        self.source_branch_id = source_branch_id

    @property
    def graph(self) -> VTree:
        return self._graph

    def _update_incident_branches_cache(self):
        self._outgoing_branches_id = self.graph.node_outgoing_branches(self.id)
        self._incoming_branches_id = self.graph.node_incoming_branches(self.id)
        self._ibranch_ids = np.concatenate([self._outgoing_branches_id, self._incoming_branches_id])
        self._ibranch_dirs = self.graph.branch_list[self._ibranch_ids][:, 0] == self.id

    def _clear_incident_branches_cache(self):
        self._outgoing_branches_id = None
        self._incoming_branches_id = None
        self._ibranch_ids = None
        self._ibranch_dirs = None

    # __ Incoming branches __
    @property
    def incoming_branches_id(self) -> List[int]:
        if self._incoming_branches_id is None:
            self._update_incident_branches_cache()
        return self._incoming_branches_id

    @property
    def indegree(self) -> int:
        if self._incoming_branches_id is None:
            self._update_incident_branches_cache()
        return len(self._incoming_branches_id)

    def incoming_branches(self) -> Iterable[VTreeBranch]:
        if self._incoming_branches_id is None:
            self._update_incident_branches_cache()
        return (VTreeBranch(self.graph, i) for i in self._incoming_branches_id)

    # __ Outgoing branches __
    @property
    def outgoing_branches_id(self) -> List[int]:
        if self._outgoing_branches_id is None:
            self._update_incident_branches_cache()
        return self._outgoing_branches_id

    @property
    def outdegree(self) -> int:
        if self._outgoing_branches_id is None:
            self._update_incident_branches_cache()
        return len(self._outgoing_branches_id)

    def outgoing_branches(self) -> Iterable[VTreeBranch]:
        if self._outgoing_branches_id is None:
            self._update_incident_branches_cache()
        return (VTreeBranch(self.graph, i) for i in self._outgoing_branches_id)

    def source_branch(self) -> VTreeBranch | None:
        return VTreeBranch(self.graph, self.source_branch_id) if self.source_branch_id is not None else None


class VTreeBranch(VGraphBranch):
    def __init__(
        self,
        graph: VTree,
        id: int,
        nodes_id: Optional[npt.NDArray[np.int_]] = None,
        dir: Optional[bool] = None,
    ):
        super().__init__(graph, id, nodes_id)
        self._dir = dir if dir is not None else bool(graph.branch_dirs(id))
        self._children_branches_id = None

    @property
    def graph(self) -> VTree:
        return super().graph

    @property
    def dir(self) -> bool:
        """The direction of the branch in graph.branch_list."""
        return self._dir

    # __ Head and tail nodes __
    @property
    def directed_nodes_id(self) -> Tuple[int, int]:
        """The indexes of the nodes: (tail_id, head_id)."""
        return self._nodes_id if self._dir else self._nodes_id[::-1]

    @property
    def tail_id(self) -> int:
        """The index of the tail node."""
        return self._nodes_id[0] if self._dir else self._nodes_id[1]

    @property
    def head_id(self) -> int:
        """The index of the head node."""
        return self._nodes_id[1] if self._dir else self._nodes_id[0]

    def tail_node(self) -> VTreeNode:
        """Return the tail node of the branch as a :class:`VTreeNode`."""
        return VTreeNode(self.graph, self._nodes_id[0] if self._dir else self._nodes_id[1])

    def head_node(self) -> VTreeNode:
        """Return the head node of the branch as a :class:`VTreeNode`."""
        return VTreeNode(self.graph, self._nodes_id[1] if self._dir else self._nodes_id[0], source_branch_id=self.id)

    # __ Ancestor branch(es) __
    @property
    def ancestor_id(self) -> int:
        """The index of the ancestor branch, or -1 if the branch is a root branch."""
        return self.graph.branch_tree[self.id]

    def ancestor(self) -> VTreeBranch | None:
        """Returns the ancestor of this branch as a :class:`VTreeBranch`, or None if the branch is a root branch."""
        p = self.graph.branch_tree[self.id]
        return VTreeBranch(self.graph, p) if p >= 0 else None

    def ancestors(self, *, max_depth: int | None = 1) -> Generator[VTreeBranch]:
        """Iterate over the ancestors of the branch as :class:`VTreeBranch`."""
        b = self.ancestor()
        while b is not None and (max_depth is None or b.depth < max_depth):
            yield b
            b = b.ancestor()

    # __ Successor branches __
    def _update_children(self):
        self._children_branches_id = self.graph.branch_successors(self.id)

    def _clear_children(self):
        self._children_branches_id = None

    @property
    def n_successors(self) -> int:
        """Number of direct successors of the branch."""
        if not self.is_valid:
            return 0
        if self._children_branches_id is None:
            self._update_children()
        return len(self._children_branches_id)

    @property
    def has_successors(self) -> bool:
        """Check if the branch has successors."""
        return self.n_successors > 0

    @property
    def successors_ids(self) -> List[int]:
        """The indexes of the direct successors of the branch."""
        if not self.is_valid:
            return []
        if self._children_branches_id is None:
            self._update_children()
        return [int(_) for _ in self._children_branches_id]

    def successors(self) -> Iterable[VTreeBranch]:
        """Iterate over the direct successors of the branch as :class:`VTreeBranch`."""
        if not self.is_valid:
            return ()
        if self._children_branches_id is None:
            self._update_children()
        return (VTreeBranch(self.graph, i) for i in self._children_branches_id)

    def successor(self, index: int) -> VTreeBranch:
        """Return the direct successor of the branch at the given index."""
        assert self.is_valid, "The branch was removed from the tree."
        if self._children_branches_id is None:
            self._update_children()
        try:
            return VTreeBranch(self.graph, self._children_branches_id[index])
        except IndexError:
            raise IndexError(
                f"Index {index} out of range for branch {self.id} with {self.n_successors} successors."
            ) from None

    def walk(self, *, depth_first=False, dynamic=False) -> Generator[VTreeBranch]:
        """Create a walker object to traverse the successors of the branch.

        Parameters
        ----------
        depth_first : bool, optional
            If True, the walker will traverse the tree in depth-first order. By default: False.

        Returns
        -------
        Generator[int]
            A generator that yields :class:`VTreeBranch` objects.
        """
        return self.graph.walk_branches(self.id, depth_first=depth_first, dynamic=False)

    # __ GeoAttr Accessor __
    def head_tip_geodata(
        self,
        attrs: Optional[VBranchGeoDataKey | List[VBranchGeoDataKey]] = None,
        geodata: Optional[VGeometricData | int] = None,
    ) -> np.ndarray | Dict[str, np.ndarray]:
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)
        return geodata.tip_data(attrs, self._id, first_tip=not self._dir)

    def successors_tip_geodata(
        self,
        attrs: Optional[VBranchGeoDataKey | List[VBranchGeoDataKey]] = None,
        geodata: Optional[VGeometricData | int] = None,
    ) -> np.ndarray | Dict[str, np.ndarray]:
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)

        succ_id = self.successors_ids
        succ_dirs = self.graph.branch_dirs(succ_id)
        return geodata.tip_data(attrs, succ_id, first_tip=succ_dirs)


########################################################################################################################
#  === VASCULAR TREE CLASS ===
########################################################################################################################
class VTree(VGraph):
    def __init__(
        self,
        branch_list: npt.NDArray[np.int_],
        branch_tree: npt.NDArray[np.int_],
        branch_dirs: npt.NDArray[np.bool_] | None,
        geometric_data: VGeometricData | Iterable[VGeometricData],
        nodes_attr: Optional[pd.DataFrame] = None,
        branches_attr: Optional[pd.DataFrame] = None,
        nodes_count: Optional[int] = None,
        check_integrity: bool = True,
    ):
        """Create a Graph object from the given data.

        Parameters
        ----------
        branches_list :
            A 2D array of shape (B, 2) where B is the number of branches. Each row contains the indexes of the nodes
            connected by each branch.

        geometric_data : VGeometricData or Iterable[VGeometricData]
            The geometric data associated with the graph.

        nodes_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each node. The index of the dataframe must be the nodes indexes.

        branches_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each branch. The index of the dataframe must be the branches indexes.

        nodes_count : int, optional
            The number of nodes in the graph. If not provided, the number of nodes is inferred from the branch list and the integrity of the data is checked with :meth:`VGraph.check_integrity`.


        Raises
        ------
        ValueError
            If the input data does not match the expected shapes.
        """  # noqa: E501
        branch_tree = np.asarray(branch_tree, dtype=np.int_)
        assert (
            branch_tree.ndim == 1 and branch_tree.size == branch_list.shape[0]
        ), "branches_tree must be a 1D array of shape (B,) where B is the number of branches. "
        if branch_dirs is not None:
            branch_dirs = np.asarray(branch_dirs, dtype=np.bool_)
            assert branch_tree.shape == branch_dirs.shape, "branches_tree and branches_dirs must have the same shape."

        #: The tree structure of the branches as a 1D vector.
        #: Each element correspond to a branch and contains the index of the parent branch.
        self.branch_tree = branch_tree

        #: The direction of the branches.
        #: Each element correspond to a branch, if True the branch is directed from its first node to its second.
        self._branch_dirs = branch_dirs

        super().__init__(
            branch_list,
            geometric_data,
            nodes_attr=nodes_attr,
            branches_attr=branches_attr,
            nodes_count=nodes_count,
            check_integrity=check_integrity,
        )

    def check_tree_integrity(self):
        """Check the integrity of the tree.

        Raises
        ------
        ValueError
            If the tree is not a tree (i.e. contains cycles).
        """
        B = self.branches_count
        assert self.branch_tree.min() >= -1, "The tree contains branches with invalid branch indexes."
        assert self.branch_tree.max() < B, "The tree contains branches with invalid branch indexes."
        assert np.all(self.branch_tree != np.arange(B)), "The tree contains branches that are their own parent."
        assert not has_cycle(self.branch_tree), "The tree contains cycles."

    def check_integrity(self):
        super().check_integrity()
        self.check_tree_integrity()

    def copy(self) -> VTree:
        """Return a copy of the tree."""
        return VTree(
            self.branches_list.copy(),
            self.branch_tree.copy(),
            self._branch_dirs.copy() if self._branch_dirs is not None else None,
            self.geometric_data.copy(),
            nodes_attr=self.nodes_attr.copy() if self.nodes_attr is not None else None,
            branches_attr=self.branches_attr.copy() if self.branches_attr is not None else None,
            nodes_count=self.nodes_count,
            check_integrity=False,
        )

    @classmethod
    def from_graph(
        cls,
        graph: VGraph,
        branch_tree: npt.NDArray[np.int_],
        branch_dirs: npt.NDArray[np.bool_] | None,
        copy=True,
        check=True,
    ) -> VTree:
        """Create a tree from a graph.

        Parameters
        ----------
        graph : VGraph
            The graph to convert to a tree.
        branch_tree : npt.NDArray[np.int_]
            The tree structure of the branches as a 1D vector. Each element correspond to a branch and contains the index of the parent branch.
        branch_dirs : npt.NDArray[np.bool_] | None
            The direction of the branches. Each element correspond to a branch, if True the branch is directed from its first node to its second.
        copy : bool, optional
            If True, the graph is copied before conversion. By default: True.
        check : bool, optional
            If True, the integrity of the tree is checked after the conversion. By default: True.

        Returns
        -------
        VTree
            The tree created from the graph.
        """  # noqa: E501
        if copy:
            graph = graph.copy()
        tree = cls(
            branch_list=graph._branch_list,
            branch_tree=branch_tree,
            branch_dirs=branch_dirs,
            geometric_data=graph._geometric_data,
            nodes_attr=graph._nodes_attr,
            branches_attr=graph._branches_attr,
            nodes_count=graph.nodes_count,
            check_integrity=False,
        )
        if check:
            tree.check_tree_integrity()
        return tree

    ####################################################################################################################
    #  === TREE BRANCHES PROPERTIES ===
    ####################################################################################################################
    @overload
    def branch_dirs(self, branch_ids: int) -> np.bool_: ...
    @overload
    def branch_dirs(self, branch_ids: npt.ArrayLike[int]) -> np.ndarray[np.bool_]: ...
    def branch_dirs(self, branch_ids: int | npt.ArrayLike[int]) -> np.bool_ | np.ndarray:
        """Return the direction of the given branch(es).

        If ``True``, the tail node is the first of  ``tree.branch_list[branch_ids]`` and the head node is the second.
        Otherwise, the head node is the first and the tail node is the second.

        Parameters
        ----------
        branch_ids : int | npt.ArrayLike[int]
            The index of the branch(es).

        Returns
        -------
        np.bool_ | np.ndarray
            The direction of the branch(es).
        """
        if self._branch_dirs is None:
            return True if np.isscalar(branch_ids) else np.ones(len(branch_ids), dtype=bool)
        return self._branch_dirs[branch_ids]

    def root_branches_ids(self) -> np.ndarray:
        """Return the indexes of the root branches.

        Returns
        -------
        np.ndarray
            The indexes of the root branches.
        """
        return np.argwhere(self.branch_tree == -1).flatten()

    def tree_branch_list(self) -> npt.NDArray[np.int_]:
        """Return the list of branches of the tree.

        Returns
        -------
        npt.NDArray[np.int_]
            The list of branches of the tree.
        """
        if self._branch_dirs is None:
            return self.branch_list

        dirs = self._branch_dirs
        branch_list = self.branch_list.copy()
        branch_list[dirs] = branch_list[dirs][:, ::-1]
        return branch_list

    def branch_ancestors(
        self, branch_id: int | npt.ArrayLike[int], *, max_depth: int | None = 1
    ) -> np.ndarray[np.int_]:
        """Return the index of the ancestor (parent) branches of a given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        max_depth : int, optional
            The maximum distance between the branch and its ancestors. By default: 1.

            If None, the function will return all the ancestors of the branch.

        Returns
        -------
        np.ndarray
            The indexes of the ancestor branches.
        """
        active_branches = np.atleast_1d(branch_id).astype(int).flatten()
        ancestors = []
        depth = 0
        while active_branches.size > 0 and (max_depth is None or depth < max_depth):
            active_branches = self.branch_tree[active_branches]
            ancestors.append(active_branches)
            depth += 1
        return np.unique(np.concatenate(ancestors))

    def branch_successors(
        self, branch_id: int | npt.ArrayLike[int], *, max_depth: int | None = 1
    ) -> np.ndarray[np.int_]:
        """Return the index of the successor (children) branches of a given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        depth : int, optional
            The maximum distance between the branch and its successors. By default: 1.

            If None the function will return all the ancestors of the branch.

        Returns
        -------
        np.ndarray
            The indexes of the successors branches.
        """
        active_branches = np.atleast_1d(branch_id).astype(int).flatten()
        successors = []
        depth = 0
        while active_branches.size > 0 and (max_depth is None or depth < max_depth):
            active_branches = np.argwhere(np.isin(self.branch_tree, active_branches)).flatten()
            successors.append(active_branches)
            depth += 1
        return np.unique(np.concatenate(successors))

    def branch_has_successors(self, branch_id: int | npt.ArrayLike[int]) -> bool | np.ndarray:
        """Check if the given branch(es) have successors.

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        Returns
        -------
        np.ndarray
            A boolean array indicating if the branches have successors.
        """
        branch_id, is_single = as_1d_array(branch_id, dtype=int)
        has_succ = np.any(branch_id[:, None] == self.branch_tree[None, :], axis=1)
        return has_succ[0] if is_single else has_succ

    def subtrees_branch_labels(self) -> npt.NDArray[int]:
        """Label each branch with a unique identifier of its subtree.

        Returns
        -------
        np.ndarray
            The labels of the subtrees.
        """
        labels = np.empty(self.branches_count, dtype=int)
        set_branches = np.zeros(self.branches_count, dtype=bool)
        for i, branch_id in enumerate(self.root_branches_ids()):
            labels[branch_id] = i
            succ = self.branch_successors(branch_id, max_depth=None)
            labels[succ] = i
            set_branches[branch_id] = True
            set_branches[succ] = True
        assert set_branches.all(), "Some branches were not assigned to a subtree."
        return labels

    def branch_head(self, branch_id: int | npt.ArrayLike[int]) -> int | npt.NDArray[np.int_]:
        """Return the head node(s) of the given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        Returns
        -------
        np.ndarray
            The indexes of the head nodes.
        """
        branch_id, is_single = as_1d_array(branch_id, dtype=int)
        heads = self.branch_list[branch_id]
        if self._branch_dirs is not None:
            heads = heads[:, np.where(self._branch_dirs[branch_id], 1, 0)]
        return heads[0] if is_single else heads

    def branch_tail(self, branch_id: int | npt.ArrayLike[int]) -> int | npt.NDArray[np.int_]:
        """Return the tail node(s) of the given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        Returns
        -------
        np.ndarray
            The indexes of the tail nodes.
        """
        branch_id, is_single = as_1d_array(branch_id, dtype=int)
        tails = self.branch_list[branch_id]
        if self._branch_dirs is not None:
            tails = tails[:, np.where(self._branch_dirs[branch_id], 0, 1)]
        return tails[0] if is_single else tails

    def walk(
        self, branch_id: Optional[int | npt.ArrayLike[int]] = None, *, depth_first=False, ignore_provided_id=True
    ) -> Generator[int]:
        """Create a walker object to traverse the tree from the given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch to start the walk from.

        depth_first : bool, optional
            If True, the walker will traverse the tree in depth-first order. By default: False.

        ignore_provided_id : bool, optional
            If True, the walker will ignore the starting branch(es) and start from their children. By default: True.

        Returns
        -------
        Generator[Tuple[int, int]]
            A generator that yields the indexes of the child branches.
        """
        if branch_id is None:
            stack = self.root_branches_ids()
        else:
            stack = np.atleast_1d(branch_id).astype(int).flatten()
            if ignore_provided_id:
                stack = self.branch_successors(stack)
        stack = stack.tolist()

        while stack:
            branch_id = stack.pop() if depth_first else stack.pop(0)
            yield branch_id
            stack.extend(self.branch_successors(branch_id))

    ####################################################################################################################
    #  === TREE NODES PROPERTIES ===
    ####################################################################################################################
    def root_nodes_ids(self) -> npt.NDArray[np.int_]:
        """Return the indexes of the root nodes.

        Returns
        -------
        np.ndarray
            The indexes of the root nodes.
        """
        root_branches = np.argwhere(self.branch_tree == -1).flatten()
        root_nodes = self.branch_list[root_branches]
        if self._branch_dirs is not None:
            root_nodes = root_nodes[:, np.where(self._branch_dirs[root_branches], 0, 1)]
        return np.unique(root_nodes)

    def node_incoming_branches(self, node_id: int | npt.ArrayLike[int]) -> npt.NDArray[np.int_]:
        """Return the ingoing branches of the given node(s).

        Parameters
        ----------
        node_id : int
            The index of the node.

        Returns
        -------
        np.ndarray
            The indexes of the ingoing branches of the node.
        """
        branch_list = self.tree_branch_list()
        if isinstance(node_id, int):
            return np.argwhere(branch_list[:, 1] == node_id).flatten()
        else:
            return np.argwhere(np.isin(branch_list[:, 1], np.asarray(node_id))).flatten()

    def node_indegree(self, node_id: Optional[int | npt.ArrayLike[int]] = None) -> int | npt.NDArray[np.int_]:
        """Return the indegree of the given node(s).

        Parameters
        ----------
        node_id : int
            The index of the node.

        Returns
        -------
        np.ndarray
            The indegree of the node.
        """
        branch_list = self.tree_branch_list()
        if isinstance(node_id, int):
            return np.any(branch_list[:, 1] == node_id) * 1
        elif node_id is None:
            return np.bincount(branch_list[:, 1], minlength=self.nodes_count)
        else:
            node_id = np.atleast_1d(node_id).astype(int).flatten()
            return np.any(node_id[:, None] == branch_list[None, :, 1], axis=1) * 1

    def node_outgoing_branches(self, node_id: int | npt.ArrayLike[int]) -> npt.NDArray[np.int_]:
        """Return the outgoing_branches branches of the given node.

        Parameters
        ----------
        node_id : int
            The index of the node.

        Returns
        -------
        np.ndarray
            The indexes of the ingoing branches of the node.
        """
        branch_list = self.tree_branch_list()
        if isinstance(node_id, int):
            return np.argwhere(branch_list[:, 0] == node_id).flatten()
        else:
            return np.argwhere(np.isin(branch_list[:, 0], np.asarray(node_id))).flatten()

    def node_outdegree(self, node_id: Optional[int | npt.ArrayLike[int]] = None) -> int | npt.NDArray[np.int_]:
        """Return the outdegree of the given node(s).

        Parameters
        ----------
        node_id : int
            The index of the node.

        Returns
        -------
        np.ndarray
            The outdegree of the node.
        """
        branch_list = self.tree_branch_list()
        if isinstance(node_id, int):
            return np.any(branch_list[:, 0] == node_id) * 1
        elif node_id is None:
            return np.bincount(branch_list[:, 0], minlength=self.nodes_count)
        else:
            node_id = np.atleast_1d(node_id).astype(int).flatten()
            return np.any(node_id[:, None] == branch_list[None, :, 0], axis=1) * 1

    def node_predecessors(self, node_id: int | npt.ArrayLike[int], *, max_depth: int | None = 1):
        """Return the index of the ancestor (parent) nodes of a given node(s).

        Parameters
        ----------
        node_id : int
            The index of the node(s).

        max_depth : int, optional
            The maximum distance between the node and its ancestors. By default: 1.

            If 0 or None, the function will return all the ancestors of the node.

        Returns
        -------
        np.ndarray
            The indexes of the ancestor nodes.
        """
        node_id = np.atleast_1d(node_id).astype(int).flatten()
        max_depth = max_depth - 1 if max_depth is not None else None
        branch_list = self.tree_branch_list()
        pred_branches = np.argwhere(np.isin(branch_list[:, 1], node_id)).flatten()  # Ingoig branches
        pred_branches = np.concatenate([self.branch_ancestors(pred_branches, max_depth=max_depth), pred_branches])
        return np.unique(branch_list[pred_branches, 0])

    def node_successors(self, node_id: int | npt.ArrayLike[int], *, max_depth: int | None = 1):
        """Return the index of the successor (children) nodes of a given node(s).

        Parameters
        ----------
        node_id : int
            The index of the node(s).

        max_depth : int, optional
            The maximum distance between the node and its successors. By default: 1.

            If 0 or None, the function will return all the successors of the node.

        Returns
        -------
        np.ndarray
            The indexes of the successor nodes.
        """
        node_id = np.atleast_1d(node_id).astype(int).flatten()
        max_depth = max_depth - 1 if max_depth is not None else None
        branch_list = self.tree_branch_list()
        succ_branches = np.argwhere(np.isin(branch_list[:, 0], node_id)).flatten()  # Outgoing branches
        succ_branches = np.concatenate([self.branch_successors(succ_branches, max_depth=max_depth), succ_branches])
        return np.unique(branch_list[succ_branches, 1])

    @overload
    def passing_nodes(self, as_mask: Literal[False] = False) -> npt.NDArray[np.int_]: ...
    @overload
    def passing_nodes(self, as_mask: Literal[True]) -> npt.NDArray[np.bool_]: ...
    def passing_nodes(self, as_mask=False) -> npt.NDArray[np.int_ | np.bool_]:
        """Return the indexes of the nodes that have exactly one incoming and one outgoing branch.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a boolean mask instead of the indexes. By default: False.

        Returns
        -------
        npt.NDArray[np.int_ | np.bool_]
            The indexes of the passing nodes (or if ``as_mask`` is True, a boolean mask of shape (N,) where N is the number of nodes).
        """  # noqa: E501
        branch_list = self.tree_branch_list()
        incdeg = np.bincount(branch_list[:, 1], minlength=self.nodes_count)
        outdeg = np.bincount(branch_list[:, 0], minlength=self.nodes_count)
        mask = (incdeg == 1) & (outdeg == 1)
        return np.where(mask)[0] if not as_mask else mask

    ####################################################################################################################
    #  === TREE MANIPULATION ===
    ####################################################################################################################
    def reindex_branches(self, indexes, inverse_lookup=False) -> VTree:
        """Reindex the branches of the tree.

        Parameters
        ----------
        indexes : npt.NDArray
            A lookup table to reindex the branches.

        inverse_lookup : bool, optional

            - If False, indexes is sorted by old indexes and contains destinations: indexes[old_index] -> new_index.
            - If True, indexes is sorted by new indexes and contains origin positions: indexes[new_index] -> old_index.

            By default: False.

        Returns
        -------
        VTree
            The modified tree.
        """
        indexes = complete_lookup(indexes, max_index=self.branches_count - 1)
        if inverse_lookup:
            indexes = invert_complete_lookup(indexes)

        super().reindex_branches(indexes, inverse_lookup=False)
        self.branch_tree = self.branch_tree[indexes]
        if self._branch_dirs is not None:
            self._branch_dirs = self._branch_dirs[indexes]

        indexes = add_empty_to_lookup(indexes, increment_index=False)
        self.branch_tree = indexes[self.branch_tree + 1]
        return self

    def reindex_nodes(self, indexes, inverse_lookup=False) -> VTree:
        """Reindex the nodes of the tree.

        Parameters
        ----------
        indexes : npt.NDArray
            A lookup table to reindex the nodes.

        inverse_lookup : bool, optional

            - If False, indexes is sorted by old indexes and contains destinations: indexes[old_index] -> new_index.
            - If True, indexes is sorted by new indexes and contains origin positions: indexes[new_index] -> old_index.

            By default: False.

        Returns
        -------
        VTree
            The modified tree.
        """
        super().reindex_nodes(indexes, inverse_lookup=inverse_lookup)
        return self

    def flip_branches_to_tree_dir(self) -> VTree:
        """Flip the direction of the branches to match the tree structure.

        Returns
        -------
        VTree
            The modified tree.
        """
        if self._branch_dirs is None:
            return self
        return self.flip_branches_direction(np.argwhere(~self._branch_dirs).flatten())

    def flip_branches_direction(self, branches_id: int | Iterable[int]) -> VTree:
        super().flip_branches_direction(branches_id)

        if self._branch_dirs is None:
            dirs = np.ones(self.branches_count, dtype=bool)
            dirs[branches_id] = False
            self._branch_dirs = dirs
        else:
            self._branch_dirs[branches_id] = ~self._branch_dirs[branches_id]
            if self._branch_dirs.all():
                self._branch_dirs = None

    def _delete_branches(self, branch_indexes: npt.NDArray[np.int_], update_refs: bool = True) -> npt.NDArray[np.int_]:
        branches_reindex = super()._delete_branches(branch_indexes, update_refs=update_refs)
        reindex = add_empty_to_lookup(branches_reindex, increment_index=False)
        self.branch_tree = reindex[np.delete(self.branch_tree, branch_indexes) + 1]
        if self._branch_dirs is not None:
            self._branch_dirs = np.delete(self._branch_dirs, branch_indexes)
        return branches_reindex

    def delete_branches(
        self,
        branch_indexes: int | npt.ArrayLike[int],
        delete_orphan_nodes=True,
        *,
        delete_successors=False,
        inplace=True,
    ) -> VTree:
        """Remove the branches with the given indexes from the tree.

        Parameters
        ----------
        branch_indexes : int | npt.ArrayLike[int]
            The indexes of the branches to remove.

        delete_orphan_nodes : bool, optional
            If True (by default), the nodes that are not connected to any branch after the deletion are removed from the tree.

        delete_successors : bool, optional
            If True, the successors of the deleted branches are also removed. By default: False.
            Otherwise if a branch has successors, an error is raised.

            By default: False.

        inplace : bool, optional
            If True (by default), the tree is modified in place. Otherwise, a new tree is returned.

        Returns
        -------
        VTree
            The modified tree.
        """  # noqa: E501
        tree = self.copy() if not inplace else self

        branch_indexes = np.atleast_1d(branch_indexes).astype(int).flatten()
        if delete_successors:
            branch_indexes = np.unique(np.concatenate([branch_indexes, tree.branch_successors(branch_indexes)]))
        else:
            assert (
                invalid := np.setdiff1d(
                    branch_indexes[np.where(tree.branch_has_successors(branch_indexes))[0]], branch_indexes
                )
            ).size == 0, f"The branches {branch_indexes[invalid]} can't be deleted: they still have successors."
        super(self.__class__, tree).delete_branches(
            branch_indexes, delete_orphan_nodes=delete_orphan_nodes, inplace=True
        )
        return self

    def fuse_nodes(
        self,
        nodes: npt.NDArray[np.int_],
        *,
        quiet_invalid_node=False,
        inplace=False,
        incident_branches: Optional[List[npt.NDArray[np.int_]]] = None,
    ) -> VTree:
        """Fuse nodes connected to exactly two branches.

        The nodes are removed from the tree and their corresponding branches are merged.

        Parameters
        ----------
        nodes : npt.NDArray
            Array of indexes of the nodes to fuse.

        quiet_invalid_node : bool, optional
            If True, do not raise an error if a node is not connected to exactly two branches.

        inplace : bool, optional
            If True, the tree is modified in place. Otherwise, a new tree is returned.

            By default: False.

        incident_branches : npt.NDArray, optional
            The incident branches of the nodes. If None, the incident branches are computed.

        Returns
        -------
        VTree
            The modified tree.
        """
        nodes = np.atleast_1d(nodes).astype(int).flatten()

        if not quiet_invalid_node:
            assert (
                len(invalid_nodes := np.where((self.node_indegree(nodes) != 1) | (self.node_outdegree(nodes) != 1))[0])
                == 0
            ), (
                f"The nodes {nodes[invalid_nodes]} can't be fuse: "
                "they don't have exactly one incoming branch and one outgoing branch."
            )
        else:
            nodes = nodes[(self.node_indegree(nodes) == 1) & (self.node_outdegree(nodes) == 1)]

        tree = self.copy() if not inplace else self

        # === Flip the branches to match the tree structure ===
        # Necessary to reindex branches correctly in branch_tree
        # (otherwise, using the reindexing code underneath, a branch can be its own parent)
        # tree.flip_branches_to_tree_dir()
        branch_tree = tree.branch_tree
        branch_rank = invert_complete_lookup(np.array(list(self.walk(depth_first=False)), dtype=int))

        # === Fuse the node in the graph and geometrical data ===
        merged_branch, flip_branch, del_lookup, del_branch = tree._fuse_nodes(
            nodes, quiet_invalid_node=quiet_invalid_node, incident_branches=incident_branches
        )

        # === Redirect the branch ancestors and direction===
        ancestors_lookup = np.arange(branch_tree.size)
        for branches, flip in zip(merged_branch, flip_branch, strict=True):
            b0 = np.argmin(branch_rank[branches])  # Select the lower ancestor of all merged branches
            ancestors_lookup[branches[:b0]] = branches[b0]
            ancestors_lookup[branches[b0:]] = branches[b0]
            if flip:
                main_branch = del_lookup[branches[b0]]
                tree._branch_dirs[main_branch] = ~tree._branch_dirs[main_branch]
        branch_tree = branch_tree[ancestors_lookup]

        # === Apply branch deletion to branch tree ===
        del_lookup = add_empty_to_lookup(del_lookup, increment_index=False)
        tree.branch_tree = np.delete(del_lookup[branch_tree + 1], del_branch)
        return tree

    def merge_nodes(self, nodes: npt.NDArray[np.int_], *, inplace=False) -> VTree:
        """Merge the given nodes into a single node.

        The nodes are removed from the tree and their corresponding branches are merged.

        Parameters
        ----------
        nodes : npt.NDArray
            Array of indexes of the nodes to merge.

        inplace : bool, optional
            If True, the tree is modified in place. Otherwise, a new tree is returned.

            By default: False.

        Returns
        -------
        VTree
            The modified tree.
        """
        raise NotImplementedError("The merge_nodes method is not implemented yet.")

    def split_branch(self, branch_id: int, node_id: int, *, inplace=False) -> VTree:
        """Split the given branch at the given node.

        Parameters
        ----------
        branch_id : int
            The index of the branch to split.

        node_id : int
            The index of the node where the branch should be split.

        inplace : bool, optional
            If True, the tree is modified in place. Otherwise, a new tree is returned.

            By default: False.

        Returns
        -------
        VTree
            The modified tree.
        """
        raise NotImplementedError("The split_branch method is not implemented yet.")

    def bridge_nodes(self, node_id1: int, node_id2: int, *, inplace=False) -> VTree:
        """Bridge the two given nodes with a new branch.

        Parameters
        ----------
        node_id1 : int
            The index of the first node.

        node_id2 : int
            The index of the second node.

        inplace : bool, optional
            If True, the tree is modified in place. Otherwise, a new tree is returned.

            By default: False.

        Returns
        -------
        VTree
            The modified tree.
        """
        raise NotImplementedError("The bridge_nodes method is not implemented yet.")

    # TODO: Implement split_branch, bridge_nodes, merge_nodes methods

    ####################################################################################################################
    #  === BRANCH AND NODE ACCESSORS ===
    ####################################################################################################################
    NODE_ACCESSOR_TYPE = VTreeNode
    BRANCH_ACCESSOR_TYPE = VTreeBranch

    def branches(
        self, ids: Optional[int | npt.ArrayLike[int]] = None, /, *, only_terminal=False, dynamic_iterator: bool = False
    ) -> Generator[VTreeBranch]:
        """Iterate over the branches of a tree, encapsulated in :class:`VTreeBranch` objects.

        Parameters
        ----------
        ids : int or npt.ArrayLike[int], optional
            The indexes of the branches to iterate over. If None, iterate over all branches.

        only_terminal : bool, optional
            If True, iterate only over the terminal branches.

        dynamic_iterator : bool, optional
            If True, iterate over all the branches present in the ytrr when this method is called.
            All branches added during the iteration will be ignored. Branches reindexed during the iteration will be visited in their original order at the time of the call. Deleted branches will not be visited.

            Enable this option if you plan to modify the tree during the iteration. If you only plan to read the tree, disable this option for better performance.

        Returns
        -------
        Generator[VTreeBranch]
            A generator that yields branches.
        """  # noqa: E501
        return super().branches(ids, only_terminal=only_terminal, dynamic_iterator=dynamic_iterator)

    def branch(self, branch_id: int, /) -> VTreeBranch:
        return super().branch(branch_id)

    def root_branches(self) -> Generator[VTreeBranch]:
        """Iterate over the root branches of the tree, encapsulated in :class:`VTreeBranch` objects.

        Returns
        -------
        Generator[VTreeBranch]
            A generator that yields the root branches.
        """
        for b in self.root_branches_ids():
            yield VTreeBranch(self, b)

    def walk_branches(
        self, branch_id: Optional[int] = None, depth_first=False, dynamic=False
    ) -> Generator[VTreeBranch]:
        """Create a walker object to traverse the tree from the given node.

        Parameters
        ----------
        branch_id : int
            The index of the branch to start the walk from.

        Returns
        -------
        Generator[int]
            A generator that yields the indexes of the child branches.
        """
        if dynamic:
            for b in list(self.walk_branches(branch_id, depth_first)):
                yield b
            return

        stack = list(self.root_branches_ids()) if branch_id is None else [branch_id]
        while stack:
            branch_id = stack.pop() if depth_first else stack.pop(0)
            branch_children = self.branch_successors(branch_id)
            branch_dir = self._branch_dirs[branch_id] if self._branch_dirs is not None else True
            branch = VTreeBranch(self, branch_id, dir=branch_dir, nodes_id=self.branch_list[branch_id])
            branch._children_branches_id = branch_children
            yield branch
            stack.extend(branch_children)

    def root_nodes(self) -> Generator[VTreeNode]:
        """Iterate over the root nodes of the tree, encapsulated in :class:`VTreeNode` objects.

        Returns
        -------
        Generator[VTreeNode]
            A generator that yields the root nodes.
        """
        for n in self.root_nodes_ids():
            yield VTreeNode(self, n)

    def walk_nodes(self, node_id: Optional[int] = None, depth_first=False) -> Generator[VTreeNode]:
        """Create a walker object to traverse the tree from the given node.

        Parameters
        ----------
        node_id : int
            The index of the node to start the walk from.

        Returns
        -------
        Generator[int]
            A generator that yields the indexes of the child nodes.
        """
        branch_list = self.tree_branch_list()
        if node_id is None:
            outgoing_branches = self.root_branches_ids()
            if depth_first:
                outgoing_branches = outgoing_branches[::-1]
        else:
            outgoing_branches = np.argwhere(branch_list[:, 0] == node_id).flatten()

        for b in self.walk(outgoing_branches, depth_first=depth_first, ignore_provided_id=False):
            yield VTreeNode(self, branch_list[b, 1])

    def node(self, node_id: int, /) -> VGraphNode:
        return super().node(node_id)

    def nodes(
        self,
        ids: Optional[int | npt.ArrayLike[int]] = None,
        /,
        *,
        only_degree: Optional[int | npt.ArrayLike[np.int_]] = None,
        only_outdegree: Optional[int | npt.ArrayLike[np.int_]] = None,
        dynamic_iterator: bool = False,
    ) -> Generator[VTreeNode]:
        """Iterate over the nodes of the tree encapsulated in :class:`VTreeNode` objects.

        Parameters
        ----------
        ids : Optional[int  |  npt.ArrayLike[int]], optional
            The indexes of the nodes to iterate over. If None, iterate over all the nodes.

        only_degree : Optional[int  |  npt.ArrayLike[np.int_]], optional
            If provided, only iterate over the nodes with the given total degree.

        only_outdegree : Optional[int  |  npt.ArrayLike[np.int_]], optional
            If provided, only iterate over the nodes with the given outdegree.

        dynamic_iterator : bool, optional
            If True, iterate over all the branches present in the ytrr when this method is called.
            All branches added during the iteration will be ignored. Branches reindexed during the iteration will be visited in their original order at the time of the call. Deleted branches will not be visited.

            Enable this option if you plan to modify the tree during the iteration. If you only plan to read the tree, disable this option for better performance.

        Yields
        ------
        Generator[VTreeNode]
            The nodes of the tree as :class:`VTreeNode` objects.
        """  # noqa: E501
        # Filter nodes by outdegree
        if only_outdegree is not None:
            ids = np.atleast_1d(ids).astype(int) if ids is not None else np.arange(self.nodes_count)
            ids = np.intersect1d(ids, np.argwhere(np.isin(self.node_outdegree(), only_outdegree)).flatten())

        return super().nodes(ids=ids, only_degree=only_degree, dynamic_iterator=dynamic_iterator)
