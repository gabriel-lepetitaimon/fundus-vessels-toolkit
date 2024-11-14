from __future__ import annotations

__all__ = ["VTree"]

from typing import Dict, Generator, Iterable, List, Literal, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.lookup_array import add_empty_to_lookup, complete_lookup, invert_complete_lookup
from ..utils.numpy import as_1d_array
from ..utils.tree import find_cycles, has_cycle
from .vgeometric_data import VBranchGeoDataKey, VGeometricData
from .vgraph import IndexesLike, VGraph, VGraphBranch, VGraphNode


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
        return super().graph

    def _update_incident_branches_cache(self):
        self._outgoing_branches_id = self.graph.node_outgoing_branches(self.id)
        self._incoming_branches_id = self.graph.node_incoming_branches(self.id)
        self._ibranch_ids = np.concatenate([self._outgoing_branches_id, self._incoming_branches_id])
        self._ibranch_dirs = self.graph._branch_list[self._ibranch_ids][:, 0] == self.id

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

    def has_ancestor(self) -> bool:
        """Check if the branch has an ancestor."""
        return self.ancestor_id >= 0

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
        self._branch_tree = branch_tree

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
        assert self.branch_tree.min() >= -1, "Invalid tree: the provided branch parents contains invalid indexes."
        assert self.branch_tree.max() < B, "Invalid tree: the provided branch parents contains invalid indexes."
        assert np.all(self.branch_tree != np.arange(B)), "Invalid tree: some branches are their own parent."
        assert not has_cycle(self.branch_tree), "Invalid tree: it contains the cycles " + "; ".join(
            "{" + ", ".join(str(_) for _ in cycle) + "}" for cycle in find_cycles(self.branch_tree)
        )

    def check_integrity(self):
        super().check_integrity()
        self.check_tree_integrity()

    def copy(self) -> VTree:
        """Return a copy of the tree."""
        return VTree(
            self._branch_list.copy(),
            self._branch_tree.copy(),
            self._branch_dirs.copy() if self._branch_dirs is not None else None,
            [gdata.copy(None) for gdata in self._geometric_data],
            nodes_attr=self._nodes_attr.copy() if self.nodes_attr is not None else None,
            branches_attr=self._branches_attr.copy() if self._branches_attr is not None else None,
            nodes_count=self._nodes_count,
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

    @classmethod
    def empty(cls) -> VTree:
        """Create an empty tree."""
        return cls(np.empty((0, 2), dtype=int), np.empty(0, dtype=int), None, [], nodes_count=0, check_integrity=False)

    @classmethod
    def empty_like(cls, other: VTree) -> VTree:
        """Create an empty tree with the same attributes as another tree."""
        assert isinstance(other, VTree), "The input must be a VTree object."
        return super().empty_like(other)

    @classmethod
    def _empty_like_kwargs(cls, other: VTree) -> Dict[str, any]:
        return super()._empty_like_kwargs(other) | {"branch_tree": np.empty(0, dtype=int), "branch_dirs": None}

    ####################################################################################################################
    #  === TREE BRANCHES PROPERTIES ===
    ####################################################################################################################
    @property
    def branch_tree(self) -> npt.NDArray[np.int_]:
        """The tree structure of the branches as a 1D vector.

        Each element correspond to a branch and contains the index of the parent branch.
        """
        return self._branch_tree

    @overload
    def branch_dirs(self, branch_ids: int) -> np.bool_: ...
    @overload
    def branch_dirs(self, branch_ids: Optional[npt.ArrayLike[int]] = None) -> np.ndarray[np.bool_]: ...
    def branch_dirs(self, branch_ids: Optional[int | npt.ArrayLike[int]] = None) -> np.bool_ | np.ndarray:
        """Return the direction of the given branch(es).

        If ``True``, the tail node is the first of  ``tree.branch_list[branch_ids]`` and the head node is the second.
        Otherwise, the head node is the first and the tail node is the second.

        Parameters
        ----------
        branch_ids : int | npt.ArrayLike[int] | None
            The index of the branch(es). If None, the function will return the direction of all the branches.

        Returns
        -------
        np.bool_ | np.ndarray
            The direction of the branch(es).
        """
        if branch_ids is None:
            return self._branch_dirs if self._branch_dirs is not None else np.ones(self.branches_count, dtype=bool)
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
        return np.argwhere(self._branch_tree == -1).flatten()

    def leaf_branches_ids(self) -> np.ndarray:
        """Return the indexes of the leaf branches.

        Returns
        -------
        np.ndarray
            The indexes of the leaf branches.
        """
        return np.setdiff1d(np.arange(self.branches_count), self._branch_tree)

    def tree_branch_list(self) -> npt.NDArray[np.int_]:
        """Return the list of branches of the tree.

        Returns
        -------
        npt.NDArray[np.int_]
            The list of branches of the tree.
        """
        if self._branch_dirs is None:
            return self._branch_list

        dirs = self._branch_dirs
        branch_list = self._branch_list.copy()
        branch_list[~dirs] = branch_list[~dirs][:, ::-1]
        return branch_list

    def branch_ancestor(self, branch_id: int | npt.ArrayLike[int], *, max_depth: int | None = 1) -> np.ndarray[np.int_]:
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
        if len(ancestors) == 0:
            return np.empty(0, dtype=int)
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
        if len(successors) == 0:
            return np.empty(0, dtype=int)
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

    def branch_distance_to_root(self, branch_id: Optional[int | npt.ArrayLike[int]] = None) -> np.ndarray:
        """Return the distance of the given branch(es) to the root.

        Roots branches have a distance of 0.

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        Returns
        -------
        np.ndarray
            The distance of the branches to the root.
        """
        if branch_id is None:
            dist = np.zeros(self.branches_count, dtype=int)
            b = self.root_branches_ids()
            i = 1
            while len(b := self.branch_successors(b, max_depth=1)) > 0:
                dist[b] = i
                i += 1
            return dist

        else:
            branch_id, is_single = as_1d_array(branch_id, dtype=int)
            dist = np.zeros(len(branch_id), dtype=int)
            b = self.branch_tree[branch_id]
            while np.any(b != -1):
                non_root_b = b != -1
                b[non_root_b] = self.branch_tree[b[non_root_b]]
                dist[non_root_b] += 1
            return dist[0] if is_single else dist

    def subtrees(self) -> List[npt.NDArray[np.int_]]:
        """Return the indexes of the branches of each subtree.

        Returns
        -------
        List[np.ndarray]
            The indexes of the branches of each subtree.
        """
        subtrees = []
        set_branches = np.zeros(self.branches_count, dtype=bool)
        for branch_id in self.root_branches_ids():
            subtree_branches = np.concatenate([[branch_id], self.branch_successors(branch_id, max_depth=None)])
            subtrees.append(subtree_branches)

            assert not np.any(set_branches[subtree_branches]), "Some branches are assigned to multiple subtrees."
            set_branches[subtree_branches] = True

        assert set_branches.all(), "Some branches were not assigned to a subtree."
        return subtrees

    def subtrees_branch_labels(self) -> npt.NDArray[np.int_]:
        """Label each branch with a unique identifier of its subtree.

        Returns
        -------
        np.ndarray
            The labels of the subtrees.
        """
        labels = np.empty(self.branches_count, dtype=int)
        for i, branches_ids in enumerate(self.subtrees()):
            labels[branches_ids] = i
        return labels

    def branch_head(self, branch_id: Optional[int | npt.ArrayLike[int]] = None) -> int | npt.NDArray[np.int_]:
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

        if branch_id is None:
            branch_list = self._branch_list
            is_single = False
        else:
            is_single = np.isscalar(branch_id)
            branch_id = self.as_branches_ids(branch_id)
            branch_list = self._branch_list[branch_id]
        if self._branch_dirs is None:
            heads = branch_list[:, 1]
        else:
            branch_dir = self._branch_dirs if branch_id is None else self._branch_dirs[branch_id]
            heads = np.take_along_axis(branch_list, np.where(branch_dir, 1, 0)[:, None], axis=1)[:, 0]
        return heads[0] if is_single else heads

    def branch_tail(self, branch_id: Optional[int | npt.ArrayLike[int]] = None) -> int | npt.NDArray[np.int_]:
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
        if branch_id is None:
            branch_list = self._branch_list
            is_single = False
        else:
            is_single = np.isscalar(branch_id)
            branch_id = self.as_branches_ids(branch_id)
            branch_list = self._branch_list[branch_id]
        if self._branch_dirs is None:
            tails = branch_list[:, 0]
        else:
            branch_dir = self._branch_dirs if branch_id is None else self._branch_dirs[branch_id]
            tails = np.take_along_axis(branch_list, np.where(branch_dir, 0, 1)[:, None], axis=1)[:, 0]
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
        root_branches = np.argwhere(self._branch_tree == -1).flatten()
        root_nodes = self._branch_list[root_branches]
        if self._branch_dirs is not None:
            node_id = np.where(self._branch_dirs[root_branches], 0, 1)
            root_nodes = np.take_along_axis(root_nodes, node_id[:][:, None], axis=1).squeeze(1)
        else:
            root_nodes = root_nodes[:, 0]
        return np.unique(root_nodes)

    def leaf_nodes_ids(self) -> npt.NDArray[np.int_]:
        """Return the indexes of the leaf nodes.

        Returns
        -------
        np.ndarray
            The indexes of the leaf nodes.
        """
        leaf_branches = np.setdiff1d(np.arange(self.branches_count), self._branch_tree)
        leaf_nodes = self._branch_list[leaf_branches]
        if self._branch_dirs is not None:
            node_id = np.where(self._branch_dirs[leaf_branches], 1, 0)
            leaf_nodes = np.take_along_axis(leaf_nodes, node_id[:][:, None], axis=1).squeeze(1)
        else:
            leaf_nodes = leaf_nodes[:, 1]
        return np.unique(leaf_nodes)

    @overload
    def crossing_nodes_ids(
        self,
        branch_ids: Optional[npt.ArrayLike[int]] = None,
        *,
        return_branch_ids: Literal[False] = False,
        only_traversing: bool = True,
    ) -> npt.NDArray[np.int_]: ...
    @overload
    def crossing_nodes_ids(
        self,
        branch_ids: Optional[npt.ArrayLike[int]] = None,
        *,
        return_branch_ids: Literal[True],
        only_traversing: bool = True,
    ) -> Tuple[npt.NDArray[np.int_], List[Dict[int, npt.NDArray[np.int_]]]]: ...
    def crossing_nodes_ids(
        self,
        branch_ids: Optional[npt.ArrayLike[int]] = None,
        *,
        return_branches: bool = False,
        only_traversing: bool = True,
    ) -> npt.NDArray[np.int_] | Tuple[npt.NDArray[np.int_], List[Dict[int, npt.NDArray[np.int_]]]]:
        """Return the indexes of the crossing nodes.

        Crossing nodes are nodes with two or more incoming branches with successors.

        Parameters
        ----------
        branch_ids : np.ndarray
            The indexes of the branches to consider. If None (by default), all the branches are considered.

        return_branch_ids : bool, optional
            If True, the function will return a dictionary containing the indexes of the crossing branches for each node.

        only_traversing : bool, optional
            If True, the function will only consider the branches that are traversing the node. By default: True.

        Returns
        -------
        np.ndarray
            The indexes of the crossing nodes.

        List[Dict[int, np.ndarray]]
            For each crossing nodes, a dictionary containing the indexes of the incoming branches (as key) and their successors (as value).
        """  # noqa: E501
        crossing_candidates = np.argwhere(self.node_indegree() >= 2).flatten()
        crossings = []
        crossing_branches = []

        if only_traversing:
            for node_id in crossing_candidates:
                outgoing_branches = self.node_outgoing_branches(node_id)
                if branch_ids is not None:
                    outgoing_branches = np.intersect1d(outgoing_branches, branch_ids)
                incoming_branches_with_successors = np.unique(self.branch_tree[outgoing_branches])
                if branch_ids is not None:
                    incoming_branches_with_successors = np.intersect1d(incoming_branches_with_successors, branch_ids)

                if (l := len(incoming_branches_with_successors)) > 1:
                    if incoming_branches_with_successors[0] != -1 or l > 2:
                        crossings.append(node_id)
                        if return_branches:
                            branches = {}
                            outgoing_branch_anc = self.branch_tree[outgoing_branches]
                            for b in incoming_branches_with_successors:
                                branches[b] = outgoing_branches[np.argwhere(outgoing_branch_anc == b).flatten()]
                            crossing_branches.append(branches)
            crossings = np.array(crossings, dtype=int)
        else:
            crossings = crossing_candidates
            crossing_incoming_branches = [self.node_incoming_branches(c) for c in crossings]
            if branch_ids is not None:
                crossing_incoming_branches = [np.intersect1d(b, branch_ids) for b in crossing_incoming_branches]
                crossings = [c for c, b in zip(crossings, crossing_incoming_branches, strict=True) if len(b) > 0]
                crossing_incoming_branches = [b for b in crossing_incoming_branches if len(b) > 0]
            crossing_branches = [
                {b: self.branch_successors(b, max_depth=1) for b in branches} for branches in crossing_incoming_branches
            ]
        return (crossings, crossing_branches) if return_branches else crossings

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
        pred_branches = np.concatenate([self.branch_ancestor(pred_branches, max_depth=max_depth), pred_branches])
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

    def node_distance_to_root(self, node_id: Optional[int | npt.ArrayLike[int]] = None) -> int | npt.NDArray[np.int_]:
        """Return the distance between the given node(s) and the root node.

        Parameters
        ----------
        node_id : int
            The index of the node(s).

        Returns
        -------
        np.ndarray
            The distance between the node and the root node.
        """
        if node_id is None:
            dist = np.zeros(self.nodes_count, dtype=int)
            branch = self.root_branches_ids()
            i = 1
            while branch:
                dist[self.branch_head(branch)] = i
                branch = self.branch_successors(branch, max_depth=1)
                i += 1
            return dist
        else:
            dist = []
            for nid in np.atleast_1d(node_id):
                branches = self.node_incoming_branches(nid)
                if len(branches) == 0:
                    dist.append(0)
                else:
                    d = 1
                    while -1 not in (branches := self.branch_tree[branches]):
                        d += 1
                    dist.append(d)
            return np.array(dist, dtype=int)

    @overload
    def passing_nodes(
        self,
        *,
        as_mask: Literal[False] = False,
        exclude_loop: bool = False,  # , return_branch_index
    ) -> npt.NDArray[np.int_]: ...
    @overload
    def passing_nodes(self, *, as_mask: Literal[True], exclude_loop: bool = False) -> npt.NDArray[np.bool_]: ...
    def passing_nodes(self, *, as_mask=False, exclude_loop: bool = False) -> npt.NDArray[np.int_ | np.bool_]:
        """Return the indexes of the nodes that have exactly one incoming and one outgoing branch.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a boolean mask instead of the indexes. By default: False.

        exclude_loop : bool, optional
            This argument have no effect for trees (self-loop are not allowed). It is kept for compatibility with :meth:`VGraph.passing_nodes`.

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

    @overload
    def passing_nodes_with_branch_index(
        self,
        *,
        exclude_loop: bool = False,
        return_branch_direction: Literal[False] = False,
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: ...
    @overload
    def passing_nodes_with_branch_index(
        self, *, exclude_loop: bool = False, return_branch_direction: Literal[True]
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.bool_]]: ...
    def passing_nodes_with_branch_index(
        self, *, exclude_loop: bool = False, return_branch_direction: bool = False
    ) -> (
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]
        | Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.bool_]]
    ):
        """Return the indexes of the nodes that are connected to exactly two branches along with the indexes of these branches.

        Parameters
        ----------
        exclude_loop : bool, optional
            This argument have no effect for trees (cycles are not allowed). It is kept for compatibility with :meth:`VGraph.passing_nodes_with_branch_index`.

        return_branch_direction : bool, optional
            If True, also return whether the branch is outgoing from the node. Default is False.

        Returns
        -------
        passing_nodes_index : npt.NDArray[np.int_]
            The indexes of the passing nodes as a array of shape (N,) where N is the number of passing nodes.

        incident_branch_index : npt.NDArray[np.int_]
            The indexes of the branches connected to the passing nodes as a array of shape (N, 2). The branches are ordered as: [incoming, outgoing].

        branch_direction : npt.NDArray[np.bool_] (optional)
            An array of shape (N,2) indicating the direction of the branches according to :attr:`VGraph.branch_list`:
            - For the first branches, True indicates that the passing node is the second node of the branch
            - For the second branches, True indicates that the passing node is the first node of the branch.

            (This is only returned if ``return_branch_direction`` is True.)

        """  # noqa: E501
        branch_tree = self._branch_tree
        ancestor, successor, ancestor_count = np.unique(branch_tree, return_index=True, return_counts=True)
        passing_mask = ancestor_count == 1
        passing_branch = np.stack([ancestor[passing_mask], successor[passing_mask]], axis=1)

        passing_branch_list = self._branch_list[passing_branch[:, 0]]  # Shape (N, 2)
        if self._branch_dirs is not None:
            passing_branch_dirs = self._branch_dirs[passing_branch]  # Shape (N,)
            inverted_inc_branch = ~passing_branch_dirs[:, 0]
            passing_branch_list[inverted_inc_branch] = passing_branch_list[inverted_inc_branch][:, ::-1]
        elif return_branch_direction:
            passing_branch_dirs = np.ones((passing_branch.shape[0], 2), dtype=bool)
        passing_nodes = passing_branch_list[:, 1]

        if not return_branch_direction:
            return passing_nodes, passing_branch
        return passing_nodes, passing_branch, passing_branch_dirs

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
        self._branch_tree[indexes] = self._branch_tree
        if self._branch_dirs is not None:
            self._branch_dirs[indexes] = self._branch_dirs

        indexes = add_empty_to_lookup(indexes, increment_index=False)
        self._branch_tree = indexes[self.branch_tree + 1]
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

    def flip_branches_direction(self, branches_id: IndexesLike) -> VTree:
        branches_id = self.as_branches_ids(branches_id)
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
        self._branch_tree = reindex[np.delete(self.branch_tree, branch_indexes) + 1]
        if self._branch_dirs is not None:
            self._branch_dirs = np.delete(self._branch_dirs, branch_indexes)
        return branches_reindex

    def delete_branches(
        self,
        branch_indexes: IndexesLike,
        delete_orphan_nodes=True,
        *,
        delete_successors=False,
        inplace=True,
    ) -> VTree:
        """Remove the branches with the given indexes from the tree.

        Parameters
        ----------
        branch_indexes : IndexesLike
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

        branch_indexes = tree.as_branches_ids(branch_indexes)
        if delete_successors:
            branch_indexes = np.unique(np.concatenate([branch_indexes, tree.branch_successors(branch_indexes)]))
        else:
            assert (
                invalid := np.setdiff1d(
                    branch_indexes[np.where(tree.branch_has_successors(branch_indexes))[0]], branch_indexes
                )
            ).size == 0, f"The branches {branch_indexes[invalid]} can't be deleted: they still have successors."
        super(tree.__class__, tree).delete_branches(
            branch_indexes, delete_orphan_nodes=delete_orphan_nodes, inplace=True
        )
        return tree

    def delete_nodes(self, node_indexes: IndexesLike, *, inplace: bool = False) -> VTree:
        """Remove the nodes with the given indexes from the tree.

        Parameters
        ----------
        node_indexes : IndexesLike
            The indexes of the nodes to remove.

        inplace : bool, optional
            If True (by default), the tree is modified in place. Otherwise, a new tree is returned.

        Returns
        -------
        VTree
            The modified tree.
        """
        tree = self.copy() if not inplace else self
        super(tree.__class__, tree).delete_nodes(node_indexes, inplace=True)
        return tree

    def fuse_nodes(
        self,
        nodes: IndexesLike,
        *,
        quiet_invalid_node=False,
        inplace=False,
        incident_branches: Optional[List[npt.NDArray[np.int_]]] = None,
    ) -> VTree:
        """Fuse nodes connected to exactly two branches.

        The nodes are removed from the tree and their corresponding branches are merged.

        Parameters
        ----------
        nodes : IndexesLike
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
        nodes = self.as_nodes_ids(nodes)

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

        branch_tree = tree.branch_tree
        branch_rank = invert_complete_lookup(np.array(list(self.walk(depth_first=False)), dtype=int))

        # === Fuse the node in the graph and geometrical data ===
        merged_branch, flip_branch, del_lookup, del_branch = tree._fuse_nodes(
            nodes, quietly_ignore_invalid_nodes=quiet_invalid_node, incident_branches=incident_branches
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
        tree._branch_tree = np.delete(del_lookup[branch_tree + 1], del_branch)
        return tree

    def merge_consecutive_branches(
        self,
        branch_pairs: npt.ArrayLike[int],
        junction_nodes: Optional[npt.ArrayLike[int]] = None,
        *,
        remove_orphan_nodes: bool = True,
        quietly_ignore_invalid_pairs: Optional[bool] = False,
        inplace=False,
    ) -> VTree:
        """Merge consecutive branches in the graph.

        Parameters
        ----------
        branch_pairs : npt.ArrayLike[int]
            The pairs of consecutive branches to merge.

        junction_nodes : npt.ArrayLike[int], optional
            The indexes of the junction nodes connecting each pair of branches. If None, the junction nodes are inferred from the branches graph.

        remove_orphan_nodes : bool, optional
            If True, the nodes that are not connected to any branch after the deletion are removed from the graph.

        quietly_ignore_invalid_pairs : bool, optional
            If True, ignore any branch pairs not consecutive.
            If False, raise an error if such branches are found.
            If None, assumes that the provided branch pairs are valid and sorted as [parent, child].

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.
        """  # noqa: E501
        tree = self.copy() if not inplace else self
        branch_pairs = np.asarray(branch_pairs, dtype=int).reshape(-1, 2)
        branch_tree = tree._branch_tree

        if quietly_ignore_invalid_pairs is not None:
            tree_branch_list = tree.tree_branch_list()
            # === Check and sort each pairs as [parent, child] ===
            flipped_pairs_id = branch_tree[branch_pairs[:, 1]] != branch_pairs[:, 0]
            if flipped_pairs_id.any():
                branch_pairs[flipped_pairs_id] = branch_pairs[flipped_pairs_id][:, ::-1]
                invalid_pairs = branch_tree[branch_pairs[:, 1]] != branch_pairs[:, 0]
                if np.any(invalid_pairs):
                    if quietly_ignore_invalid_pairs:
                        branch_pairs = np.delete(branch_pairs, invalid_pairs)
                        if junction_nodes is not None:
                            junction_nodes = np.delete(junction_nodes, invalid_pairs)
                    else:
                        raise ValueError(f"Invalid consecutive branches: {branch_pairs[invalid_pairs].tolist()}.")

            # === Check that the junction nodes are valid ===
            if junction_nodes is None:
                junction_nodes = tree_branch_list[branch_pairs[:, 0], 1]
            else:
                junction_nodes = np.asarray(junction_nodes, dtype=int)

                invalid_pairs = junction_nodes != tree_branch_list[branch_pairs[:, 0], 1]
                invalid_pairs |= junction_nodes != tree_branch_list[branch_pairs[:, 1], 0]

                if np.any(invalid_pairs):
                    if quietly_ignore_invalid_pairs:
                        branch_pairs = np.delete(branch_pairs, invalid_pairs)
                        junction_nodes = np.delete(junction_nodes, invalid_pairs)
                    else:
                        raise ValueError(f"Invalid consecutive branches: {branch_pairs[invalid_pairs].tolist()}.")
        elif junction_nodes is None:
            tree_branch_list = tree.tree_branch_list()
            junction_nodes = tree_branch_list[branch_pairs[:, 0], 1]

        # === Merge the branch in the graph and geometrical data ===
        merged_branch, flip_branch, merge_lookup, del_branch = tree._merge_consecutive_branches(
            branch_pairs, junction_nodes, remove_orphan_nodes=remove_orphan_nodes
        )

        # === Redirect the branch ancestors and direction===
        ancestors_lookup = np.arange(branch_tree.size)
        for branches, flip in zip(merged_branch, flip_branch, strict=True):
            b0 = branches[0]
            ancestors_lookup[branches] = b0
            if flip:
                main_branch = merge_lookup[b0]
                tree._branch_dirs[main_branch] = ~tree._branch_dirs[main_branch]

        # === Apply branch deletion to branch tree ===
        branch_tree = branch_tree[ancestors_lookup]
        merge_lookup = add_empty_to_lookup(merge_lookup, increment_index=False)
        tree._branch_tree = np.delete(merge_lookup[branch_tree + 1], del_branch)
        return tree

    def merge_nodes(self, nodes: IndexesLike, *, inplace=False) -> VTree:
        """Merge the given nodes into a single node.

        The nodes are removed from the tree and their corresponding branches are merged.

        Parameters
        ----------
        nodes : IndexesLike
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
        self,
        ids: Optional[int | npt.ArrayLike[int]] = None,
        /,
        *,
        filter: Optional[Literal["orphan", "endpoint", "non-endpoint"]] = None,
        dynamic_iterator: bool = False,
    ) -> Generator[VTreeBranch]:
        """Iterate over the branches of a tree, encapsulated in :class:`VTreeBranch` objects.

        Parameters
        ----------
        ids : int or npt.ArrayLike[int], optional
            The indexes of the branches to iterate over. If None, iterate over all branches.

        filter : str, optional
            Filter the branches to iterate over:

            - "orphan": iterate over the branches that are not connected to any other branch.
            - "endpoint": iterate over the branches that are connected to only one other branch.
            - "non-endpoint": iterate over the branches that are connected to more than one other branch.


        dynamic_iterator : bool, optional
            If True, iterate over all the branches present in the ytrr when this method is called.
            All branches added during the iteration will be ignored. Branches reindexed during the iteration will be visited in their original order at the time of the call. Deleted branches will not be visited.

            Enable this option if you plan to modify the tree during the iteration. If you only plan to read the tree, disable this option for better performance.

        Returns
        -------
        Generator[VTreeBranch]
            A generator that yields branches.
        """  # noqa: E501
        return super().branches(ids, filter=filter, dynamic_iterator=dynamic_iterator)

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
            branch = VTreeBranch(self, branch_id, dir=branch_dir, nodes_id=self._branch_list[branch_id])
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
