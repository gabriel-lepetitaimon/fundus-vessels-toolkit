from __future__ import annotations

__all__ = ["VTree"]

from typing import Generator, Iterable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.lookup_array import complete_lookup, invert_complete_lookup
from ..utils.tree import has_cycle
from .vgeometric_data import VGeometricData
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

    def _update_incident_branches_cache(self):
        self._outgoing_branches_id = self.graph.outgoing_branches(self.id)
        self._incoming_branches_id = self.graph.incoming_branches(self.id)
        self._ibranch_ids = np.concatenate([self._outgoing_branches_id, self._incoming_branches_id])
        self._ibranch_dirs = self.graph.branch_list[self._ibranch_ids][:, 0] == self.id

    def _clear_incident_branches_cache(self):
        self._outgoing_branches_id = None
        self._incoming_branches_id = None
        self._ibranch_ids = None
        self._ibranch_dirs = None

    @property
    def graph(self) -> VTree:
        return self._graph

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
        self._dir = dir if dir is not None else graph.branches_dirs[id]
        self._children_branches_id = None

    @property
    def graph(self) -> VTree:
        return self._graph

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


########################################################################################################################
#  === VASCULAR TREE CLASS ===
########################################################################################################################
class VTree(VGraph):
    def __init__(
        self,
        branches_list: npt.NDArray[np.int_],
        branches_tree: npt.NDArray[np.int_],
        branches_dirs: npt.NDArray[np.bool_] | None,
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
        branches_tree = np.asarray(branches_tree, dtype=np.int_)
        assert (
            branches_tree.ndim == 1 and branches_tree.size == branches_list.shape[0]
        ), "branches_tree must be a 1D array of shape (B,) where B is the number of branches. "
        if branches_dirs is not None:
            branches_dirs = np.asarray(branches_dirs, dtype=np.bool_)
            assert (
                branches_tree.shape == branches_dirs.shape
            ), "branches_tree and branches_dirs must have the same shape."

        #: The tree structure of the branches as a 1D vector.
        #: Each element correspond to a branch and contains the index of the parent branch.
        self.branch_tree = branches_tree

        #: The direction of the branches.
        #: Each element correspond to a branch, if True the branch is directed from its first node to its second.
        self.branches_dirs = branches_dirs

        super().__init__(
            branches_list,
            geometric_data,
            nodes_attr=nodes_attr,
            branches_attr=branches_attr,
            nodes_count=nodes_count,
            check_integrity=check_integrity,
        )

    def check_integrity(self):
        """Check the integrity of the tree.

        Raises
        ------
        ValueError
            If the tree is not a tree (contains cycles).
        """
        super().check_integrity()
        B = self.branches_count
        assert self.branch_tree.min() >= -1, "The tree contains branches with invalid branch indexes."
        assert self.branch_tree.max() < B, "The tree contains branches with invalid branch indexes."
        assert np.all(self.branch_tree != np.arange(B)), "The tree contains branches that are their own parent."
        assert not has_cycle(self.branch_tree), "The tree contains cycles."

    def copy(self) -> VTree:
        """Return a copy of the tree."""
        return VTree(
            self.branches_list.copy(),
            self.branch_tree.copy(),
            self.branches_dirs.copy() if self.branches_dirs is not None else None,
            self.geometric_data.copy(),
            nodes_attr=self.nodes_attr.copy() if self.nodes_attr is not None else None,
            branches_attr=self.branches_attr.copy() if self.branches_attr is not None else None,
            nodes_count=self.nodes_count,
            check_integrity=False,
        )

    ####################################################################################################################
    #  === TREE PROPERTIES ===
    ####################################################################################################################
    def root_branches(self) -> np.ndarray:
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
        if self.branches_dirs is None:
            return self.branches_list

        dirs = self.branches_dirs
        branch_tree = self.branch_tree.copy()
        branch_tree[dirs] = branch_tree[dirs][:, ::-1]
        return branch_tree

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
        active_branches = np.atleast_1d(branch_id, dtype=int).flatten()
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
        active_branches = np.atleast_1d(branch_id, dtype=int).flatten()
        successors = []
        depth = 0
        while active_branches.size > 0 and (max_depth is None or depth < max_depth):
            active_branches = np.argwhere(np.isin(self.branch_tree, active_branches)).flatten()
            successors.append(active_branches)
            depth += 1
        return np.unique(np.concatenate(successors))

    def incoming_branches(self, node_id: int | npt.ArrayLike[int]) -> npt.NDArray[np.int_]:
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

    def outgoing_branches(self, node_id: int | npt.ArrayLike[int]) -> npt.NDArray[np.int_]:
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
        node_id = np.atleast_1d(node_id, dtype=int).flatten()
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
        node_id = np.atleast_1d(node_id, dtype=int).flatten()
        max_depth = max_depth - 1 if max_depth is not None else None
        branch_list = self.tree_branch_list()
        succ_branches = np.argwhere(np.isin(branch_list[:, 0], node_id)).flatten()  # Outgoing branches
        succ_branches = np.concatenate([self.branch_successors(succ_branches, max_depth=max_depth), succ_branches])
        return np.unique(branch_list[succ_branches, 1])

    def walk(self, branch_id: Optional[int] = None, depth_first=False) -> Generator[int]:
        """Create a walker object to traverse the tree from the given branch.

        Parameters
        ----------
        branch_id : int
            The index of the branch to start the walk from.

        Returns
        -------
        Generator[Tuple[int, int]]
            A generator that yields the indexes of the child branches.
        """
        stack = self.root_branches() if branch_id is None else [branch_id]
        while stack:
            branch_id = stack.pop() if depth_first else stack.pop(0)
            yield branch_id
            stack.extend(self.branch_successors(branch_id))

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
        self.branch_tree = indexes[self.branch_tree][indexes]
        self.branches_dirs = self.branches_dirs[indexes]
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
        if self.branches_dirs is None:
            return self
        return self.flip_branches_direction(np.argwhere(self.branches_dirs).flatten())

    def flip_branches_direction(self, branches_id: int | Iterable[int]) -> VTree:
        super().flip_branches_direction(branches_id)

        if self.branches_dirs is None:
            dirs = np.ones(self.branches_count, dtype=bool)
            dirs[branches_id] = False
            self.branches_dirs = dirs
        else:
            self.branches_dirs[branches_id] = ~self.branches_dirs[branches_id]
            if self.branches_dirs.all():
                self.branches_dirs = None

    # TODO: Implement _delete_branches, delete_branches, split_branch, bridge_nodes, fuse_nodes, merge_nodes methods   # noqa: E501

    ####################################################################################################################
    #  === BRANCH AND NODE ACCESSORS ===
    ####################################################################################################################
    NODE_ACCESSOR_TYPE = VTreeNode
    BRANCH_ACCESSOR_TYPE = VTreeBranch

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

        stack = self.root_branches() if branch_id is None else [branch_id]
        while stack:
            branch_id = stack.pop() if depth_first else stack.pop(0)
            branch_children = self.branch_successors(branch_id)
            branch = VTreeBranch(
                self, branch_id, dir=self.branches_dirs[branch_id], nodes_id=self.branches_list[branch_id]
            )
            branch._children_branches_id = branch_children
            yield branch
            stack.extend(branch_children)

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
        ...
