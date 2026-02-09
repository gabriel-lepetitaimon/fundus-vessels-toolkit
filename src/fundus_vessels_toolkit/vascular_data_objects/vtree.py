from __future__ import annotations

__all__ = ["VTree"]

import itertools
import re
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Self,
    Sequence,
    Tuple,
    TypeAlias,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.data_io import NumpyDict, load_numpy_dict, save_numpy_dict
from ..utils.lookup_array import add_empty_to_lookup, complete_lookup, invert_complete_lookup, lookup_from_mapping
from ..utils.numpy import array_is_equal
from ..utils.tree import find_cycles, has_cycle
from ..utils.typing import (
    Bool1DArrayLike,
    Float1DArrayLike,
    Indices,
    Int1DArray,
    Int1DArrayLike,
    IntPairArrayLike,
    PointArrayLike,
)
from .vgeometric_data import VBranchGeoDataKey, VGeometricData
from .vgraph import (
    BranchIndex,
    BranchIndices,
    BranchIndicesLike,
    NodeIndex,
    NodeIndices,
    NodeIndicesLike,
    VGraph,
    VGraphBranch,
    VGraphNode,
)


########################################################################################################################
#  === TREE NODES AND BRANCHES ACCESSORS CLASSES ===
########################################################################################################################
class VTreeNode(VGraphNode):
    def __init__(self, graph: VTree, id: int, *, source_branch_id: Optional[int] = None):
        """Create a VTreeNode object.
        Parameters
        ----------
        graph : VTree
            The graph to which the node belongs.

        id : int
            The index of the node in the graph.

        source_branch_id : int, optional
            The index of the branch that emitted this node. If the node is a root node, it should be set to -1. If the source branch is unknown, leave it as None.
            By default: None.
        """  # noqa: E501
        super().__init__(graph, id)
        self._incoming_branch_ids = None
        self._outgoing_branch_ids = None
        self._source_branch_id = source_branch_id

    @property
    def graph(self) -> VTree:
        return super().graph  # type: ignore

    def _update_incident_branch_cache(self):
        self._outgoing_branch_ids = self.graph.node_outgoing_branches(self.id)
        self._incoming_branch_ids = self.graph.node_incoming_branches(self.id)
        self._ibranch_ids = np.concatenate([self._outgoing_branch_ids, self._incoming_branch_ids])
        self._ibranch_dirs = self.graph._branch_list[self._ibranch_ids][:, 0] == self.id
        return self._incoming_branch_ids, self._outgoing_branch_ids, self._ibranch_ids, self._ibranch_dirs

    def _clear_incident_branch_cache(self):
        self._outgoing_branch_ids = None
        self._incoming_branch_ids = None
        self._ibranch_ids = None
        self._ibranch_dirs = None

    # __ Source branch __
    @property
    def source_branch_id(self) -> int | None:
        """The index of the branch that emitted this node or None if the node is a root node."""
        if self._source_branch_id is None:
            incoming_branches = self.incoming_branch_ids
            if len(incoming_branches) == 1:
                self._source_branch_id = incoming_branches[0]
            elif len(incoming_branches) == 0:
                self._source_branch_id = -1
            else:
                raise ValueError(
                    f"Node {self.id} has multiple incoming branches: {incoming_branches}."
                    "Cannot determine the source branch."
                )
        return self._source_branch_id if self._source_branch_id >= 0 else None

    def source_branch(self) -> VTreeBranch | None:
        """Return the branch that emitted this node as a :class:`VTreeBranch`.
        If the node is a root node, it returns None.
        """
        return VTreeBranch(self.graph, b_id) if (b_id := self.source_branch_id) is not None else None

    def is_root(self) -> bool:
        """Check if the node is a root node (i.e. it has no source branch)."""
        return self.source_branch_id is None

    # __ Incoming branches __
    @property
    def incoming_branch_ids(self) -> npt.NDArray[np.int_]:
        """The indices of the branches incoming to the node."""
        if (branch_ids := self._incoming_branch_ids) is None:
            branch_ids = self._update_incident_branch_cache()[0]
        return branch_ids

    @property
    def in_degree(self) -> int:
        """Number of branches incoming to the node."""
        if (branch_ids := self._incoming_branch_ids) is None:
            branch_ids = self._update_incident_branch_cache()[0]
        return len(branch_ids)

    def incoming_branches(self) -> Iterable[VTreeBranch]:
        """Iterate over the branches incoming to the node as :class:`VTreeBranch`."""
        if (branch_ids := self._incoming_branch_ids) is None:
            branch_ids = self._update_incident_branch_cache()[0]
        return (VTreeBranch(self.graph, i) for i in branch_ids)

    # __ Outgoing branches __
    @property
    def outgoing_branch_ids(self) -> npt.NDArray[np.int_]:
        """The indices of the branches outgoing from the node."""
        if (branch_ids := self._outgoing_branch_ids) is None:
            branch_ids = self._update_incident_branch_cache()[1]
        return branch_ids

    @property
    def out_degree(self) -> int:
        """Number of branches outgoing from the node."""
        if (branch_ids := self._outgoing_branch_ids) is None:
            branch_ids = self._update_incident_branch_cache()[1]
        return len(branch_ids)

    def outgoing_branches(self) -> Iterable[VTreeBranch]:
        """Iterate over the branches outgoing from the node as :class:`VTreeBranch`."""
        if (branch_ids := self._outgoing_branch_ids) is None:
            branch_ids = self._update_incident_branch_cache()[1]
        return (VTreeBranch(self.graph, i) for i in branch_ids)


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
        self._succ_branch_ids = None

    @property
    def graph(self) -> VTree:
        return super().graph  # type: ignore

    @property
    def dir(self) -> bool:
        """The direction of the branch in graph.branch_list."""
        return self._dir

    # __ Head and tail nodes __
    @property
    def directed_node_ids(self) -> npt.NDArray[np.int_]:
        """The indices of the nodes: (tail_id, head_id)."""
        return self._node_ids if self._dir else self._node_ids[::-1]

    @property
    def tail_id(self) -> int:
        """The index of the tail node."""
        return self._node_ids[0] if self._dir else self._node_ids[1]

    @property
    def head_id(self) -> int:
        """The index of the head node."""
        return self._node_ids[1] if self._dir else self._node_ids[0]

    def tail_node(self) -> VTreeNode:
        """Return the tail node of the branch as a :class:`VTreeNode`."""
        return VTreeNode(self.graph, self._node_ids[0] if self._dir else self._node_ids[1])

    def head_node(self) -> VTreeNode:
        """Return the head node of the branch as a :class:`VTreeNode`."""
        return VTreeNode(self.graph, self._node_ids[1] if self._dir else self._node_ids[0], source_branch_id=self.id)

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
        depth = 0
        while b is not None and (max_depth is None or depth < max_depth):
            yield b
            b = b.ancestor()
            depth += 1

    # __ Successor branches __
    def _update_children(self) -> npt.NDArray[np.int_]:
        branch_ids = self.graph.branch_successors(self.id)
        self._succ_branch_ids = branch_ids
        return branch_ids

    def _clear_children(self) -> None:
        self._succ_branch_ids = None

    @property
    def n_successors(self) -> int:
        """Number of direct successors of the branch."""
        if not self.is_valid:
            return 0

        if (branch_ids := self._succ_branch_ids) is None:
            branch_ids = self._update_children()
        return len(branch_ids)

    @property
    def has_successors(self) -> bool:
        """Check if the branch has successors."""
        return self.n_successors > 0

    @property
    def successors_ids(self) -> npt.NDArray[np.int_]:
        """The indices of the direct successors of the branch."""
        if not self.is_valid:
            return np.empty(0, dtype=int)
        if (branch_ids := self._succ_branch_ids) is None:
            branch_ids = self._update_children()
        return branch_ids

    def successors(self) -> Iterable[VTreeBranch]:
        """Iterate over the direct successors of the branch as :class:`VTreeBranch`."""
        if not self.is_valid:
            return ()
        if (branch_ids := self._succ_branch_ids) is None:
            branch_ids = self._update_children()
        return (VTreeBranch(self.graph, i) for i in branch_ids)

    def successor(self, index: int = 0) -> VTreeBranch:
        """Return the direct successor of the branch at the given index."""
        assert self.is_valid, "The branch was removed from the tree."
        if (succ_ids := self._succ_branch_ids) is None:
            succ_ids = self._update_children()
        try:
            return VTreeBranch(self.graph, succ_ids[index])
        except IndexError:
            raise IndexError(
                f"Index {index} out of range for branch {self.id} with {self.n_successors} successors."
            ) from None

    def walk(self, *, traversal: Literal["dfs", "bfs"] = "bfs", dynamic=False) -> Generator[VTreeBranch]:
        """Create a walker object to traverse the successors of the branch.

        Parameters
        ----------
        traversal : Literal["dfs", "bfs"], optional
            The traversal order of the walker.
            - "bfs": (by default) Breadth-first search: each branch siblings are visited before its children.
            - "dfs": Depth-first search.

        Returns
        -------
        Generator[int]
            A generator that yields :class:`VTreeBranch` objects.
        """
        return self.graph.walk_branches(self.id, traversal=traversal, dynamic=dynamic)

    # __ GeoAttr Accessor __
    @overload
    def head_tip_geodata(
        self,
        attrs: VBranchGeoDataKey,
        geodata: Optional[VGeometricData | int] = None,
    ) -> npt.NDArray: ...
    @overload
    def head_tip_geodata(
        self,
        attrs: Optional[Sequence[VBranchGeoDataKey]] = None,
        geodata: Optional[VGeometricData | int] = None,
    ) -> Dict[str, npt.NDArray]: ...
    def head_tip_geodata(
        self,
        attrs: Optional[VBranchGeoDataKey | List[VBranchGeoDataKey]] = None,
        geodata: Optional[VGeometricData | int] = None,
    ) -> npt.NDArray | Dict[str, npt.NDArray]:
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)
        return geodata.tip_data(attrs, self._id, first_tip=not self._dir)

    @overload
    def successors_tip_geodata(
        self,
        attrs: VBranchGeoDataKey,
        geodata: Optional[VGeometricData | int] = None,
    ) -> npt.NDArray: ...
    @overload
    def successors_tip_geodata(
        self,
        attrs: Optional[Sequence[VBranchGeoDataKey]] = None,
        geodata: Optional[VGeometricData | int] = None,
    ) -> Dict[str, npt.NDArray]: ...
    def successors_tip_geodata(
        self,
        attrs: Optional[VBranchGeoDataKey | Sequence[VBranchGeoDataKey]] = None,
        geodata: Optional[VGeometricData | int] = None,
    ) -> npt.NDArray | Dict[str, npt.NDArray]:
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)

        succ_id = self.successors_ids
        succ_dirs = self.graph.branch_dirs(succ_id)
        return geodata.tip_data(attrs, succ_id, first_tip=succ_dirs)


########################################################################################################################
#  === VASCULAR TREE CLASS ===
########################################################################################################################
class VTree(VGraph):
    Branch: TypeAlias = VTreeBranch
    Node: TypeAlias = VTreeNode

    def __init__(
        self,
        branch_list: IntPairArrayLike,
        branch_tree: Int1DArrayLike,
        branch_dirs: Bool1DArrayLike | None = None,
        geometric_data: VGeometricData | Iterable[VGeometricData] = (),
        node_attr: Optional[pd.DataFrame] = None,
        branch_attr: Optional[pd.DataFrame] = None,
        node_count: Optional[int] = None,
        check_integrity: bool = True,
    ):
        """Create a Graph object from the given data.

        Parameters
        ----------
        branches_list :
            A 2D array of shape (B, 2) where B is the number of branches. Each row contains the indices of the nodes
            connected by each branch.

        geometric_data : VGeometricData or Iterable[VGeometricData]
            The geometric data associated with the graph.

        nodes_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each node. The index of the dataframe must be the nodes indices.

        branches_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each branch. The index of the dataframe must be the branches indices.

        node_count : int, optional
            The number of nodes in the graph. If not provided, the number of nodes is inferred from the branch list and the integrity of the data is checked with :meth:`VGraph.check_integrity`.


        Raises
        ------
        ValueError
            If the input data does not match the expected shapes.
        """  # noqa: E501
        branch_tree = np.atleast_1d(branch_tree)
        branch_list = np.atleast_2d(branch_list)
        assert branch_tree.ndim == 1 and branch_tree.size == branch_list.shape[0], (
            "branches_tree must be a 1D array of shape (B,) where B is the number of branches. "
        )
        if branch_dirs is not None:
            branch_dirs = np.atleast_1d(branch_dirs)
            assert branch_tree.shape == branch_dirs.shape, "branches_tree and branches_dirs must have the same shape."

        #: The tree structure of the branches as a 1D vector.
        #: Each element correspond to a branch and contains the index of the parent branch.
        self._branch_tree: npt.NDArray[np.int_] = branch_tree

        #: The direction of the branches.
        #: Each element correspond to a branch, if True the branch is directed from its first node to its second.
        self._branch_dir: npt.NDArray[np.bool_] | None = branch_dirs

        super().__init__(
            branch_list,
            geometric_data,
            node_attr=node_attr,
            branch_attr=branch_attr,
            node_count=node_count,
            check_integrity=check_integrity,
        )

    @overload
    def check_tree_integrity(self, on_error: Literal["warn", "skip"] = "skip") -> bool: ...
    @overload
    def check_tree_integrity(self, on_error: Literal["raise"]) -> Literal[True]: ...
    @overload
    def check_tree_integrity(self, on_error: Literal["report"]) -> str: ...
    def check_tree_integrity(self, on_error: Literal["raise", "warn", "skip", "report"] = "skip") -> bool | str:
        """Check the integrity of the tree.

        Raises
        ------
        ValueError
            If the tree is not a tree (i.e. contains cycles).
        """
        B = self.branch_count
        if B == 0:
            return "" if on_error == "report" else True
        errors = []
        if self.branch_tree.min() >= -1:
            errors.append("the provided branch parents contains invalid indices")
        if self.branch_tree.max() < B:
            errors.append("the provided branch parents contains invalid indices")
        if np.all(self.branch_tree != np.arange(B)):
            errors.append("some branches are their own parent")
        if not has_cycle(self.branch_tree):
            errors.append(
                "it contains the cycles "
                + "; ".join("{" + ", ".join(str(_) for _ in cycle) + "}" for cycle in find_cycles(self.branch_tree))
            )
        if len(errors) > 0:
            msg = "Invalid tree:" + "\n - ".join(errors)
            if on_error == "raise":
                raise ValueError(msg)
            elif on_error == "warn":
                warnings.warn(msg, stacklevel=2)
            elif on_error == "report":
                return msg
            return False
        return "" if on_error == "report" else True

    @overload
    def check_integrity(self, on_error: Literal["warn", "skip"] = "skip") -> bool: ...
    @overload
    def check_integrity(self, on_error: Literal["raise"]) -> Literal[True]: ...
    @overload
    def check_integrity(self, on_error: Literal["report"]) -> str: ...
    def check_integrity(self, on_error: Literal["raise", "warn", "skip", "report"] = "skip") -> bool | str:
        graph_out = super().check_integrity(on_error=on_error)
        tree_out = self.check_tree_integrity(on_error=on_error)
        if on_error == "report":
            return graph_out + "\n" + tree_out if graph_out and tree_out else graph_out + tree_out
        return graph_out and tree_out

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, VTree)
            and np.array_equal(self._branch_tree, other._branch_tree)
            and array_is_equal(self._branch_dir, other._branch_dir)
            and super().__eq__(other)
        )

    def copy(self) -> Self:
        """Return a copy of the tree."""
        return type(self)(
            self._branch_list.copy(),
            self._branch_tree.copy(),
            self._branch_dir.copy() if self._branch_dir is not None else None,
            [gdata.copy(None) for gdata in self._geometric_data],
            node_attr=self._node_attr.copy() if self._node_attr is not None else None,
            branch_attr=self._branch_attr.copy() if self._branch_attr is not None else None,
            node_count=self._node_count,
            check_integrity=False,
        )

    @classmethod
    def from_graph(
        cls,
        graph: VGraph,
        branch_tree: npt.NDArray[np.int_],
        branch_dirs: npt.NDArray[np.bool_] | None,
        copy=True,
        check_integrity=True,
    ) -> Self:
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
        check_integrity : bool, optional
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
            node_attr=graph._node_attr,
            branch_attr=graph._branch_attr,
            node_count=graph.node_count,
            check_integrity=False,
        )
        if check_integrity:
            tree.check_tree_integrity()
        return tree

    def save(
        self,
        filename: Optional[str | Path] = None,
        *,
        on_exists: Literal["raise", "warn", "skip", "overwrite"] = "warn",
    ) -> NumpyDict:
        """Save the tree data to a file.

        The tree is saved as a dictionary with the following keys:
            - ``branch_list``: The list of branches in the graph as a 2D array of shape (B, 2) where B is the number of branches. Each row contains the indices of the nodes connected by each branch.
            - ``geometric_data``: The geometric data associated with the graph as a list of dictionaries.
            - ``node_attr``: The attributes of the nodes in the graph as a dictionary.
            - ``branch_attr``: The attributes of the branches in the graph as a dictionary.
            - ``branch_tree``: The tree structure of the branches as a 1D array.
            - ``branch_dirs``: The direction of the branches as a 1D array.

        Parameters
        ----------
        filename : str or Path, optional
            The name of the file to save the data to. If None, the data is not saved to a file.

        Returns
        -------
        NumpyDict
            The graph as a dictionary of numpy arrays.
        """  # noqa: E501

        data = super().save()
        data["branch_tree"] = self._branch_tree  # type: ignore
        data["branch_dirs"] = self.branch_dirs()  # type: ignore

        if filename is not None:
            save_numpy_dict(data, filename, on_exists=on_exists)
        return data

    @classmethod
    def load(cls, filename: str | Path | NumpyDict, *, check_integrity: bool = True) -> Self:
        """Load a Graph object from a file.

        Parameters
        ----------
        filename : str or Path
            The name of the file to load the data from.

        Returns
        -------
        VTree
            The Tree object loaded from the file.
        """
        if isinstance(filename, (str, Path)):
            data = load_numpy_dict(filename)
        else:
            data = filename

        return cls.from_graph(
            VGraph.load(data, check_integrity=False),  # type: ignore
            data["branch_tree"],  # type: ignore
            data["branch_dirs"] if not np.all(data["branch_dirs"]) else None,  # type: ignore
            copy=False,
            check_integrity=check_integrity,
        )

    @classmethod
    def empty(cls) -> VTree:
        """Create an empty tree."""
        return cls(np.empty((0, 2), dtype=int), np.empty(0, dtype=int), None, [], node_count=0, check_integrity=False)

    @classmethod
    def empty_like(cls, other: Self) -> Self:
        """Create an empty tree with the same attributes as another tree."""
        assert isinstance(other, VTree), "The input must be a VTree object."
        return super().empty_like(other)

    @classmethod
    def _empty_like_kwargs(cls, other: Self) -> Dict[str, Any]:
        return super()._empty_like_kwargs(other) | {"branch_tree": np.empty(0, dtype=int), "branch_dirs": None}

    def subtree(self, branch_ids: BranchIndicesLike, check: bool = True) -> VTree:
        """Return the subtree of the given branch(es).

        Parameters
        ----------
        branch_ids : int
            The index of the branch(es).

        Returns
        -------
        Self
            The subtree.
        """
        nodes = np.unique(self.branch_list[self.as_branch_ids(branch_ids)].flatten())
        subgraph, b_lookup = super().subgraph(nodes, return_branch_lookup=True)

        branch_mask = b_lookup != -1
        b_lookup_with_empty = add_empty_to_lookup(b_lookup)
        branch_tree = b_lookup_with_empty[self._branch_tree[branch_mask] + 1] - 1
        branch_dirs = None if self._branch_dir is None else self._branch_dir[branch_mask]

        return VTree.from_graph(subgraph, branch_tree, branch_dirs, copy=False, check_integrity=check)

    @classmethod
    def parse(cls, branch_list: str) -> Self:
        """Parse an branch tree from a string.

        Parameters
        ----------
        branch_list : str
            The string containing the branch list using the following format:
            - Each branch is defined as ``n1->n2`` or ``n1➔n2`` where ``n1`` and ``n2`` are the indices of the nodes connected by the branch.
            - Each branch is separated by ``;``.
            - Consecutive branches can be defined without separation: e.g. ``n1➔n2➔n3``.
            - Whitespace characters (including tabs and new lines) are ignored.

        Returns
        -------
        VTree
            The tree object created from the branch list.

        Examples
        --------
        >>> tree = VTree.parse("0->1;3➔2➔1")
        >>> tree.branch_list.tolist()
        [[0, 1], [3, 2], [2, 1]]

        >>> tree.branch_tree.tolist()
        [-1, -1, 1]

        Raises
        ------
        ValueError
            If the branch list is not correctly formatted.
        """  # noqa: E501
        branches: List[List[int]] = []
        branch_parents: List[int] = []
        branch_list = re.sub(r"\s+", "", branch_list)

        # TODO: Add support for node labelling (e.g. "A➔B➔C")
        node_type: Optional[Literal["label", "index"]] = None
        node_count = 0
        node_labelling: Dict[str, int] = {}

        def node_idx(node_str: str, branch_str: str) -> int:
            nonlocal node_type, node_count
            assert node_str != "", f"Invalid branch definition '{branch_str}': empty node index."
            if node_str[0].isalpha():
                if node_type == "index":
                    raise ValueError("Cannot mix labelled nodes and indexed nodes in the same branch list.")
                node_type = "label"
                return node_labelling.setdefault(node_str, len(node_labelling))
            else:
                if node_type == "label":
                    raise ValueError("Cannot mix labelled nodes and indexed nodes in the same branch list.")
                node_type = "index"
                try:
                    node = int(node_str)
                except ValueError:
                    raise ValueError(
                        f"Invalid branch definition '{branch_str}': {node_str} is not a valid node index."
                    ) from None
                node_count = max(node_count, node + 1)
                return node

        for branch in re.split(r";", branch_list):
            consecutive_nodes = re.split(r"->|➔", branch)
            if len(consecutive_nodes) == 0:
                continue
            if len(consecutive_nodes) == 1:
                node_idx(consecutive_nodes[0], branch)
                continue

            for n1, n2 in itertools.pairwise(consecutive_nodes):
                n1_idx = node_idx(n1, branch)
                branches.append([n1_idx, node_idx(n2, branch)])
                parents = [i for i, (_, n) in enumerate(branches) if n == n1_idx]
                if len(parents) == 0:
                    branch_parents.append(-1)
                else:
                    branch_parents.append(parents[-1])

        if node_type == "label":
            node_count = len(node_labelling)
            node_attr = pd.DataFrame(
                index=list(node_labelling.values()), data={"_parsed_node_labels": list(node_labelling.keys())}
            )
        else:
            node_attr = None
        return cls(branches, branch_parents, node_attr=node_attr)

    def print_tree(self) -> str:
        """Print the tree structure in a human-readable format. Unlike print_graph() this displays the connection between branches instead of nodes.
        Each branch is represented as "b{branch_id}" and the parent-child relationship between branches is represented as "b{parent_id}➔b{child_id}". Root branches are represented as "|b{branch_id}".

        Example
        -------
        >>> # branch ids:        0   1 2
        >>> tree = VTree.parse("0➔1;3➔2➔1")
        >>> tree.print_tree()
        '|b0 |b1➔b2'
        """  # noqa: E501
        tree_str = ""
        previous_b = None
        for b in self.walk_branch_ids(traversal="dfs"):
            parent = self.branch_tree[b]
            if parent == -1:
                if tree_str != "":
                    tree_str += " "
                tree_str += f"|b{b}"
            elif previous_b != parent:
                tree_str += f" b{parent}➔b{b}"
            else:
                tree_str += f"➔b{b}"
            previous_b = b
        return tree_str

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
    def branch_dirs(self, branch_ids: BranchIndex) -> bool: ...
    @overload
    def branch_dirs(self, branch_ids: Optional[BranchIndices] = None) -> npt.NDArray[np.bool_]: ...
    def branch_dirs(self, branch_ids: Optional[BranchIndicesLike] = None) -> bool | npt.NDArray[np.bool_]:
        """Return the direction of the given branch(es).

        If ``True``, the tail node is the first of  ``tree.branch_list[branch_ids]`` and the head node is the second.
        Otherwise, the head node is the first and the tail node is the second.

        Parameters
        ----------
        branch_ids :
            The index of the branch(es). If None, the function will return the direction of all the branches.

        Returns
        -------
        np.bool_ | np.ndarray
            The direction of the branch(es).
        """
        if branch_ids is None:
            return self._branch_dir if self._branch_dir is not None else np.ones(self.branch_count, dtype=bool)

        single = np.isscalar(branch_ids)
        branch_ids = self.as_branch_ids(branch_ids)
        if self._branch_dir is None:
            return True if single else np.ones(len(branch_ids), dtype=bool)
        dirs = self._branch_dir[branch_ids]
        return dirs[0] if single else dirs

    @overload
    def root_branch_ids(self, as_mask: Literal[True]) -> npt.NDArray[np.bool_]: ...
    @overload
    def root_branch_ids(self, as_mask: Literal[False] = False) -> npt.NDArray[np.int_]: ...
    def root_branch_ids(self, as_mask: bool = False) -> npt.NDArray[np.int_] | npt.NDArray[np.bool_]:
        """Return the indices of the root branches.

        Returns
        -------
        np.ndarray
            The indices of the root branches.

        Example
        -------
        >>> # branch ids:        0   1 2
        >>> tree = VTree.parse("0➔1;3➔2➔1")
        >>> tree.root_branch_ids()
        array([0, 1])
        """
        if as_mask:
            return self._branch_tree == -1
        return np.argwhere(self._branch_tree == -1).flatten()

    @overload
    def leaf_branch_ids(self, *, as_mask: Literal[True]) -> npt.NDArray[np.bool_]: ...
    @overload
    def leaf_branch_ids(self, *, as_mask: Literal[False] = False) -> npt.NDArray[np.int_]: ...
    def leaf_branch_ids(self, *, as_mask: bool = False) -> npt.NDArray[np.int_] | npt.NDArray[np.bool_]:
        """Return the indices of the leaf branches.

        Returns
        -------
        np.ndarray
            The indices of the leaf branches.
        """
        if as_mask:
            return np.isin(np.arange(self.branch_count), np.unique(self._branch_tree), assume_unique=True, invert=True)
        return np.setdiff1d(np.arange(self.branch_count), self._branch_tree)

    def tree_branch_list(self) -> npt.NDArray[np.int_]:
        """Return the list of branches of the tree.

        Returns
        -------
        npt.NDArray[np.int_]
            The list of branches of the tree.
        """
        if self._branch_dir is None:
            return self._branch_list

        dirs = self._branch_dir
        branch_list = self._branch_list.copy()
        branch_list[~dirs] = branch_list[~dirs][:, ::-1]
        return branch_list

    def branch_ancestor(self, branch_id: BranchIndicesLike, *, max_depth: int | None = 1) -> npt.NDArray[np.int_]:
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
            The indices of the ancestor branches.
        """
        active_branches = self.as_branch_ids(branch_id)
        ancestors = []
        depth = 0
        while active_branches.size > 0 and (max_depth is None or depth < max_depth):
            active_branches = self.branch_tree[active_branches]
            ancestors.append(active_branches)
            depth += 1
        if len(ancestors) == 0:
            return np.empty(0, dtype=int)
        return np.unique(np.concatenate(ancestors))

    def branch_successors(self, branch_id: BranchIndicesLike, *, max_depth: int | None = 1) -> npt.NDArray[np.int_]:
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
            The indices of the successors branches.
        """
        active_branches = self.as_branch_ids(branch_id)
        successors = []
        depth = 0
        while active_branches.size > 0 and (max_depth is None or depth < max_depth):
            active_branches = np.argwhere(np.isin(self.branch_tree, active_branches)).flatten()
            successors.append(active_branches)
            depth += 1
        if len(successors) == 0:
            return np.empty(0, dtype=int)
        return np.unique(np.concatenate(successors))

    @overload
    def branch_has_successors(self, branch_id: BranchIndex) -> bool: ...
    @overload
    def branch_has_successors(self, branch_id: Optional[BranchIndices]) -> npt.NDArray[np.bool_]: ...
    def branch_has_successors(self, branch_id: Optional[BranchIndicesLike]) -> bool | npt.NDArray[np.bool_]:
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
        branch_id, single = self.as_branch_ids(branch_id, return_is_single=True)
        has_succ = np.any(branch_id[:, None] == self.branch_tree[None, :], axis=1)
        return has_succ[0] if single else has_succ

    @overload
    def branch_distance_to_root(self, branch_id: BranchIndex) -> int: ...
    @overload
    def branch_distance_to_root(self, branch_id: Optional[BranchIndices] = None) -> npt.NDArray[np.int_]: ...
    def branch_distance_to_root(self, branch_id: Optional[BranchIndicesLike] = None) -> int | npt.NDArray[np.int_]:
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
            dist = np.zeros(self.branch_count, dtype=int)
            b = self.root_branch_ids()
            i = 1
            while len(b := self.branch_successors(b, max_depth=1)) > 0:
                dist[b] = i
                i += 1
            return dist

        else:
            branch_id, is_single = self.as_branch_ids(branch_id, return_is_single=True)
            dist = np.zeros(len(branch_id), dtype=int)
            b = self.branch_tree[branch_id]
            while np.any(b != -1):
                non_root_b = b != -1
                b[non_root_b] = self.branch_tree[b[non_root_b]]
                dist[non_root_b] += 1
            return dist[0] if is_single else dist

    def branch_ids_by_subtree(self) -> List[npt.NDArray[np.int_]]:
        """Return the indices of the branches of each subtree.

        Returns
        -------
        List[np.ndarray]
            The indices of the branches of each subtree.
        """
        subtrees = []
        set_branches = np.zeros(self.branch_count, dtype=bool)
        for branch_id in self.root_branch_ids():
            subtree_branches = np.concatenate([[branch_id], self.branch_successors(branch_id, max_depth=None)])
            subtrees.append(subtree_branches)

            assert not np.any(set_branches[subtree_branches]), "Some branches are assigned to multiple subtrees."
            set_branches[subtree_branches] = True

        assert set_branches.all(), "Some branches were not assigned to a subtree."
        return sorted(subtrees, key=lambda x: len(x), reverse=True)

    def subtrees_branch_labels(self) -> npt.NDArray[np.int_]:
        """Label each branch with a unique identifier of its subtree.

        Returns
        -------
        np.ndarray
            The labels of the subtrees.
        """
        labels = np.empty(self.branch_count, dtype=int)
        for i, branches_ids in enumerate(self.branch_ids_by_subtree()):
            labels[branches_ids] = i
        return labels

    def split_subtrees(self) -> List[VTree]:
        """Split the tree into subtrees.

        Returns
        -------
        List[VTree]
            The list of subtrees.
        """
        return [self.subtree(branch_ids) for branch_ids in self.branch_ids_by_subtree()]

    @overload
    def branch_head(self, branch_id: BranchIndex) -> int: ...
    @overload
    def branch_head(self, branch_id: Optional[BranchIndices] = None) -> npt.NDArray[np.int_]: ...
    def branch_head(self, branch_id: Optional[BranchIndicesLike] = None) -> int | npt.NDArray[np.int_]:
        """Return the head node(s) of the given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        Returns
        -------
        np.ndarray
            The indices of the head node of each provided branch.
        """

        if branch_id is None:
            branch_list = self._branch_list
            is_single = False
        else:
            branch_id, is_single = self.as_branch_ids(branch_id, return_is_single=True)
            branch_list = self._branch_list[branch_id]
        if self._branch_dir is None:
            heads = branch_list[:, 1]
        else:
            branch_dir = self._branch_dir if branch_id is None else self._branch_dir[branch_id]
            heads = np.take_along_axis(branch_list, np.where(branch_dir, 1, 0)[:, None], axis=1)[:, 0]
        return heads[0] if is_single else heads

    @overload
    def branch_tail(self, branch_id: BranchIndex) -> int: ...
    @overload
    def branch_tail(self, branch_id: Optional[BranchIndices] = None) -> npt.NDArray[np.int_]: ...
    def branch_tail(self, branch_id: Optional[BranchIndicesLike] = None) -> int | npt.NDArray[np.int_]:
        """Return the tail node(s) of the given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch(es).

        Returns
        -------
        np.ndarray
            The indices of the tail nodes.
        """
        if branch_id is None:
            branch_list = self._branch_list
            is_single = False
        else:
            branch_id, is_single = self.as_branch_ids(branch_id, return_is_single=True)
            branch_list = self._branch_list[branch_id]
        if self._branch_dir is None:
            tails = branch_list[:, 0]
        else:
            branch_dir = self._branch_dir if branch_id is None else self._branch_dir[branch_id]
            tails = np.take_along_axis(branch_list, np.where(branch_dir, 0, 1)[:, None], axis=1)[:, 0]
        return tails[0] if is_single else tails

    def walk_branch_ids(
        self,
        root_branch_id: Optional[BranchIndicesLike] = None,
        *,
        traversal: Literal["dfs", "bfs"] = "bfs",
        ignore_provided_id=True,
    ) -> Generator[int]:
        """Create a walker object to traverse the tree from the given branch(es).

        Parameters
        ----------
        branch_id : int
            The index of the branch to start the walk from.

        traversal : Literal["dfs", "bfs"], optional
            The traversal order of the walker.
            - "bfs": (by default) Breadth-first search: each branch siblings are visited before its children.
            - "dfs": Depth-first search.


        ignore_provided_id : bool, optional
            If True, the walker will ignore the starting branch(es) and start from their children. By default: True.

        Returns
        -------
        Generator[Tuple[int, int]]
            A generator that yields the indices of the child branches.
        """
        if root_branch_id is None:
            stack = self.root_branch_ids()
        else:
            stack = self.as_branch_ids(root_branch_id)
            if ignore_provided_id:
                stack = self.branch_successors(stack)
        stack = stack.tolist()

        depth_first = traversal == "dfs"
        if depth_first:
            stack.reverse()

        while stack:
            branch_id = stack.pop() if depth_first else stack.pop(0)
            yield branch_id
            stack.extend(self.branch_successors(branch_id))

    ####################################################################################################################
    #  === TREE NODES PROPERTIES ===
    ####################################################################################################################
    def root_nodes_ids(self) -> npt.NDArray[np.int_]:
        """Return the indices of the root nodes.

        Returns
        -------
        np.ndarray
            The indices of the root nodes.
        """
        root_branches = np.argwhere(self._branch_tree == -1).flatten()
        root_nodes = self._branch_list[root_branches]
        if self._branch_dir is not None:
            node_id = np.where(self._branch_dir[root_branches], 0, 1)
            root_nodes = np.take_along_axis(root_nodes, node_id[:][:, None], axis=1).squeeze(1)
        else:
            root_nodes = root_nodes[:, 0]
        return np.unique(root_nodes)

    def leaf_nodes_ids(self) -> npt.NDArray[np.int_]:
        """Return the indices of the leaf nodes.

        Returns
        -------
        np.ndarray
            The indices of the leaf nodes.
        """
        leaf_branches = np.setdiff1d(np.arange(self.branch_count), self._branch_tree)
        leaf_nodes = self._branch_list[leaf_branches]
        if self._branch_dir is not None:
            node_id = np.where(self._branch_dir[leaf_branches], 1, 0)
            leaf_nodes = np.take_along_axis(leaf_nodes, node_id[:][:, None], axis=1).squeeze(1)
        else:
            leaf_nodes = leaf_nodes[:, 1]
        return np.unique(leaf_nodes)

    @overload
    def crossing_nodes_ids(
        self,
        branch_ids: Optional[BranchIndicesLike] = None,
        *,
        return_branch_ids: Literal[False] = False,
        only_traversing: bool = True,
    ) -> NodeIndices: ...
    @overload
    def crossing_nodes_ids(
        self,
        branch_ids: Optional[BranchIndicesLike] = None,
        *,
        return_branch_ids: Literal[True],
        only_traversing: bool = True,
    ) -> Tuple[NodeIndices, List[Dict[NodeIndex, BranchIndices]]]: ...
    def crossing_nodes_ids(
        self,
        branch_ids: Optional[BranchIndicesLike] = None,
        *,
        return_branch_ids: bool = False,
        only_traversing: bool = True,
    ) -> NodeIndices | Tuple[NodeIndices, List[Dict[NodeIndex, BranchIndices]]]:
        """Return the indices of the crossing nodes.

        Crossing nodes are nodes with two or more incoming branches with successors.

        Parameters
        ----------
        branch_ids :
            The indices of the branches to consider. If None (by default), all the branches are considered.

            This argument can be used to retrist the search to a specific subtree.

        return_branch_ids : bool, optional
            If True, the function will return a dictionary containing the indices of the crossing branches for each node.

        only_traversing : bool, optional
            If True, the function will only consider the branches that are traversing the node. By default: True.

        Returns
        -------
        np.ndarray
            The indices of the crossing nodes.

        List[Dict[int, np.ndarray]]
            For each crossing nodes, a dictionary containing the indices of the incoming branches (as key) and their successors (as value).
        """  # noqa: E501
        crossing_candidates = np.argwhere(self.node_indegree() >= 2).flatten()
        if branch_ids is not None:
            branch_ids = self.as_branch_ids(branch_ids)
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

                if (n := len(incoming_branches_with_successors)) > 1:
                    if incoming_branches_with_successors[0] != -1 or n > 2:
                        crossings.append(node_id)
                        if return_branch_ids:
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
        crossings = np.array(crossings, dtype=int)
        return (crossings, crossing_branches) if return_branch_ids else crossings

    def node_incoming_branches(self, node_id: NodeIndicesLike) -> npt.NDArray[np.int_]:
        """Return the ingoing branches of the given node(s).

        Parameters
        ----------
        node_id : int
            The index of the node.

        Returns
        -------
        np.ndarray
            The indices of the ingoing branches of the node.
        """
        node_id = self.as_node_ids(node_id)
        branch_list = self.tree_branch_list()
        return np.argwhere(np.isin(branch_list[:, 1], node_id)).flatten()

    def node_indegree(self, node_id: Optional[NodeIndicesLike] = None) -> int | npt.NDArray[np.int_]:
        """Return the indegree of the given node(s).

        Parameters
        ----------
        node_id :
            The index of the node.

        Returns
        -------
        np.ndarray
            The indegree of the node.
        """
        branch_list = self.tree_branch_list()
        if node_id is None:
            return np.bincount(branch_list[:, 1], minlength=self.node_count)
        else:
            node_id = self.as_node_ids(node_id)
            return np.any(node_id[:, None] == branch_list[None, :, 1], axis=1) * 1

    def node_outgoing_branches(self, node_id: NodeIndicesLike) -> npt.NDArray[np.int_]:
        """Return the outgoing_branches branches of the given node.

        Parameters
        ----------
        node_id : int
            The index of the node.

        Returns
        -------
        np.ndarray
            The indices of the ingoing branches of the node.
        """
        branch_list = self.tree_branch_list()
        node_id = self.as_node_ids(node_id)
        return np.argwhere(np.isin(branch_list[:, 0], np.asarray(node_id))).flatten()

    def node_outdegree(self, node_id: Optional[NodeIndicesLike] = None) -> int | npt.NDArray[np.int_]:
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
        if node_id is None:
            return np.bincount(branch_list[:, 0], minlength=self.node_count)
        else:
            node_id = self.as_node_ids(node_id)
            return np.any(node_id[:, None] == branch_list[None, :, 0], axis=1) * 1

    def node_predecessors(self, node_id: NodeIndicesLike, *, max_depth: int | None = 1):
        """Return the index of the ancestor (parent) nodes of a given node(s).

        Parameters
        ----------
        node_id :
            The index of the node(s).

        max_depth : int, optional
            The maximum distance between the node and its ancestors. By default: 1.

            If 0 or None, the function will return all the ancestors of the node.

        Returns
        -------
        np.ndarray
            The indices of the ancestor nodes.
        """
        node_id = self.as_node_ids(node_id)
        max_depth = max_depth - 1 if max_depth is not None else None
        branch_list = self.tree_branch_list()
        pred_branches = np.argwhere(np.isin(branch_list[:, 1], node_id)).flatten()  # Ingoig branches
        pred_branches = np.concatenate([self.branch_ancestor(pred_branches, max_depth=max_depth), pred_branches])
        return np.unique(branch_list[pred_branches, 0])

    def node_successors(self, node_id: NodeIndicesLike, *, max_depth: int | None = 1):
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
            The indices of the successor nodes.
        """
        node_id = self.as_node_ids(node_id)
        max_depth = max_depth - 1 if max_depth is not None else None
        branch_list = self.tree_branch_list()
        succ_branches = np.argwhere(np.isin(branch_list[:, 0], node_id)).flatten()  # Outgoing branches
        succ_branches = np.concatenate([self.branch_successors(succ_branches, max_depth=max_depth), succ_branches])
        return np.unique(branch_list[succ_branches, 1])

    @overload
    def node_distance_to_root(self, node_id: NodeIndex) -> int: ...
    @overload
    def node_distance_to_root(self, node_id: Optional[NodeIndices] = None) -> npt.NDArray[np.int_]: ...
    def node_distance_to_root(self, node_id: Optional[NodeIndicesLike] = None) -> int | npt.NDArray[np.int_]:
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
            dist = np.zeros(self.node_count, dtype=int)
            branch = self.root_branch_ids()
            i = 1
            while branch:
                dist[self.branch_head(branch)] = i
                branch = self.branch_successors(branch, max_depth=1)
                i += 1
            return dist

        node_id = self.as_node_ids(node_id)

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
    def passing_nodes(self, *, as_mask: Literal[False] = False, exclude_loop: bool = True) -> npt.NDArray[np.int32]: ...
    @overload
    def passing_nodes(self, *, as_mask: Literal[True], exclude_loop: bool = True) -> npt.NDArray[np.bool_]: ...
    def passing_nodes(
        self, *, as_mask=False, exclude_loop: bool = True
    ) -> npt.NDArray[np.int32 | np.bool_] | Tuple[npt.NDArray[np.int32 | np.bool_], List[npt.NDArray[np.int32]]]:
        """Return the indices of the nodes that have exactly one incoming and one outgoing branch.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a boolean mask instead of the indices. By default: False.

        exclude_loop : bool, optional
            This argument have no effect for trees (self-loop are not allowed). It is kept for compatibility with :meth:`VGraph.passing_nodes`.

        Returns
        -------
        npt.NDArray[np.int_ | np.bool_]
            The indices of the passing nodes (or if ``as_mask`` is True, a boolean mask of shape (N,) where N is the number of nodes).
        """  # noqa: E501
        branch_list = self.tree_branch_list()
        incdeg = np.bincount(branch_list[:, 1], minlength=self.node_count)
        outdeg = np.bincount(branch_list[:, 0], minlength=self.node_count)
        mask = (incdeg == 1) & (outdeg == 1)
        return np.where(mask)[0] if not as_mask else mask

    @overload
    def passing_nodes_with_branch_index(
        self, *, return_branch_direction: Literal[False] = False, exclude_loop: bool = True
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...
    @overload
    def passing_nodes_with_branch_index(
        self, *, return_branch_direction: Literal[True], exclude_loop: bool = True
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.bool_]]: ...
    def passing_nodes_with_branch_index(
        self, *, return_branch_direction: bool = False, exclude_loop: bool = True
    ) -> (
        Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]
        | Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.bool_]]
    ):
        """Return the indices of the nodes that are connected to exactly two branches along with the indices of these branches.

        Parameters
        ----------
        exclude_loop : bool, optional
            This argument have no effect for trees (cycles are not allowed). It is kept for compatibility with :meth:`VGraph.passing_nodes_with_branch_index`.

        return_branch_direction : bool, optional
            If True, also return whether the branch is outgoing from the node. Default is False.

        Returns
        -------
        passing_nodes_index : npt.NDArray[np.int_]
            The indices of the passing nodes as a array of shape (N,) where N is the number of passing nodes.

        incident_branch_index : npt.NDArray[np.int_]
            The indices of the branches connected to the passing nodes as a array of shape (N, 2). The branches are ordered as: [incoming, outgoing].

        branch_direction : npt.NDArray[np.bool_] (optional)
            An array of shape (N,2) indicating the direction of the branches according to :attr:`VGraph.branch_list`:
            True indicates that the branch is outgoing from the node, False indicates that the branch is incoming to the node.

            (This is only returned if ``return_branch_direction`` is True.)

        """  # noqa: E501
        branch_tree = self._branch_tree
        ancestor, successor, ancestor_count = np.unique(branch_tree, return_index=True, return_counts=True)
        passing_mask = (ancestor_count == 1) & (ancestor != -1)
        passing_branch = np.stack([ancestor[passing_mask], successor[passing_mask]], axis=1)

        passing_branch_list = self._branch_list[passing_branch[:, 0]]  # Shape (N, 2)
        if self._branch_dir is not None:
            passing_branch_dirs = self._branch_dir[passing_branch]  # Shape (N,)
            inverted_inc_branch = ~passing_branch_dirs[:, 0]
            passing_branch_list[inverted_inc_branch] = passing_branch_list[inverted_inc_branch][:, ::-1]
            passing_branch_dirs[:, 0] = ~passing_branch_dirs[:, 0]  # Invert the direction of the incoming branch
        else:
            # If the branch are well directed, the first is incoming (False) and the second is outgoing (True).
            passing_branch_dirs = np.zeros((passing_branch.shape[0], 2), dtype=bool)
            passing_branch_dirs[:, 1] = True
        passing_nodes = passing_branch_list[:, 1]

        if not return_branch_direction:
            return passing_nodes, passing_branch
        return passing_nodes, passing_branch, passing_branch_dirs

    ####################################################################################################################
    #  === TREE MANIPULATION ===
    ####################################################################################################################
    def reindex_branches(self, indices: Int1DArray | Mapping[int, int], inverse_lookup=False) -> VTree:
        """Reindex the branches of the tree.

        Parameters
        ----------
        indices : npt.NDArray
            A lookup table to reindex the branches.

        inverse_lookup : bool, optional

            - If False, indices is sorted by old indices and contains destinations: indices[old_index] -> new_index.
            - If True, indices is sorted by new indices and contains origin positions: indices[new_index] -> old_index.

            By default: False.

        Returns
        -------
        VTree
            The modified tree.
        """
        if isinstance(indices, Mapping):
            indices = lookup_from_mapping(indices, self.branch_count)
        indices = complete_lookup(indices, max_index=self.branch_count - 1)
        if inverse_lookup:
            indices = invert_complete_lookup(indices)

        super().reindex_branches(indices, inverse_lookup=False)
        self._branch_tree[indices] = self._branch_tree
        if self._branch_dir is not None:
            self._branch_dir[indices] = self._branch_dir

        indices = add_empty_to_lookup(indices, increment_index=False)
        self._branch_tree = indices[self.branch_tree + 1]
        return self

    def reindex_nodes(self, indices: Int1DArray | Mapping[int, int], *, inverse_lookup=False, inplace=False) -> VTree:
        """Reindex the nodes of the tree.

        Parameters
        ----------
        indices : npt.NDArray
            A lookup table to reindex the nodes.

        inverse_lookup : bool, optional

            - If False, indices is sorted by old indices and contains destinations: indices[old_index] -> new_index.
            - If True, indices is sorted by new indices and contains origin positions: indices[new_index] -> old_index.

            By default: False.

        inplace : bool, optional
            If True (by default), the tree is modified in place. Otherwise, a new tree is returned

        Returns
        -------
        VTree
            The modified tree.
        """
        tree = self.copy() if not inplace else self
        super(tree.__class__, tree).reindex_nodes(indices, inverse_lookup=inverse_lookup, inplace=True)  # type: ignore
        return tree

    def append(self, other: VGraph, *, inplace=False) -> Self:
        """Append another tree to this tree.

        Parameters
        ----------
        other : VTree
            The tree to append.

        inplace : bool, optional
            If True (by default), the tree is modified in place. Otherwise, a new tree is returned.

        Returns
        -------
        VTree
            The modified tree.
        """
        if not inplace:
            return self.copy().append(other, inplace=True)

        N_branch = self.branch_count
        super(self.__class__, self).append(other, inplace=True)  # type: ignore

        if not isinstance(other, VTree):
            from fundus_vessels_toolkit.segment_to_graph.av_tree_parsing import naive_infer_roots

            if (
                (fundus := self.geometric_data().fundus_data) is not None
                and fundus.has_od
                and fundus.od_center is not None
            ):
                root_pos = fundus.od_center
            else:
                root_pos = self.geometric_data().domain.center
            other = naive_infer_roots(other, root_pos=root_pos)
        other_branch_tree = other.branch_tree
        other_branch_tree[other_branch_tree != -1] += N_branch
        self._branch_tree = np.concatenate([self._branch_tree, other_branch_tree])
        if self._branch_dir is None and other._branch_dir is None:
            pass
        else:
            self._branch_dir = np.concatenate([self.branch_dirs(), other.branch_dirs()])

        return self

    def flip_branch_to_tree_dir(self, inplace=False) -> VTree:
        """Flip the direction of the branches to match the tree structure.

        Returns
        -------
        VTree
            The modified tree.
        """
        if self._branch_dir is None:
            return self
        return self.flip_branch_direction(np.argwhere(~self._branch_dir).flatten(), inplace=inplace)

    def flip_branch_direction(self, branch_id: BranchIndicesLike, inplace=False) -> VTree:
        tree = self.copy() if not inplace else self
        branch_ids = self.as_branch_ids(branch_id)
        super(tree.__class__, tree).flip_branch_direction(branch_ids, inplace=True)  # type: ignore

        if tree._branch_dir is None:
            dirs = np.ones(tree.branch_count, dtype=bool)
            dirs[branch_ids] = False
            tree._branch_dir = dirs
        else:
            tree._branch_dir[branch_ids] = ~tree._branch_dir[branch_ids]
            if tree._branch_dir.all():
                tree._branch_dir = None

        return tree

    def _delete_branch(self, branch_id: npt.NDArray[np.int_], update_refs: bool = True) -> npt.NDArray[np.int_]:
        branches_reindex = super()._delete_branch(branch_id, update_refs=update_refs)
        reindex = add_empty_to_lookup(branches_reindex, increment_index=False)
        self._branch_tree = reindex[np.delete(self.branch_tree, branch_id) + 1]
        assert self._branch_tree.shape[0] == self.branch_count, "Branch tree size mismatch after branch deletion."
        if self._branch_dir is not None:
            self._branch_dir = np.delete(self._branch_dir, branch_id)
        return branches_reindex

    def delete_branch(
        self, branch_id: BranchIndicesLike, delete_orphan_nodes=True, *, delete_successors=False, inplace=False
    ) -> VTree:
        """Remove the branches with the given indices from the tree.

        Parameters
        ----------
        branch_indexes : IndexLike
            The indices of the branches to remove.

        delete_orphan_nodes : bool, optional
            If True (by default), the nodes that are not connected to any branch after the deletion are removed from the tree.

        delete_successors : bool, optional
            If True, the successors of the deleted branches are also removed. By default: False.
            Otherwise if a branch has successors, they become root branches.

            By default: False.

        inplace : bool, optional
            If True, the tree is modified in place. Otherwise, a new tree is returned.

        Returns
        -------
        VTree
            The modified tree.
        """  # noqa: E501
        tree = self.copy() if not inplace else self

        branch_id = tree.as_branch_ids(branch_id)
        if delete_successors:
            branch_id = np.unique(np.concatenate([branch_id, tree.branch_successors(branch_id)]))
        super(tree.__class__, tree).delete_branch(branch_id, delete_orphan_nodes=delete_orphan_nodes, inplace=True)  # type: ignore
        return tree

    @overload
    def add_branch(
        self, branch_nodes: IntPairArrayLike, *, return_branch_id: Literal[False] = False, inplace=False
    ) -> Self: ...
    @overload
    def add_branch(
        self, branch_nodes: IntPairArrayLike, *, return_branch_id: Literal[True], inplace=False
    ) -> Tuple[Self, npt.NDArray[np.int32]]: ...
    def add_branch(
        self, branch_nodes: IntPairArrayLike, *, return_branch_id=False, inplace=False
    ) -> Self | Tuple[Self, npt.NDArray[np.int32]]:
        """Add branch(es) to the tree. The branch(es) are connected to the tree according to the following rules:
        - If the tail node of the new branch has exactly one incoming branch, the new branch becomes a successor of that branch.
        - If the head node of the new branch has exactly one outgoing branch, that branch becomes a successor of the new branch.

        Parameters
        ----------
        branch_nodes : IntPairArrayLike
            A 2D array of shape (N, 2) containing the indices of the nodes connected by the new branches.

        return_branch_id : bool, optional
            If True, return the indices of the added branches.

        inplace : bool, optional
            If True (by default), the graph is modified in place. Otherwise, a new graph is returned.
        Returns
        -------
        VTree
            The modified graph.

        new_branch_ids : np.ndarray
            The indices of the added branches. Only returned if ``return_branch_id`` is True.

        Examples
        --------
        >>> tree = VTree.parse("0➔1➔2➔3")
        >>> tree.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3]]

        >>> tree.branch_tree.tolist()
        [-1, 0, 1]

        Add a new branch connecting nodes 1 and 3 (the new branch is a successor of branch 0):

        >>> t1 = tree.add_branch([1, 3])
        >>> t1.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3], [1, 3]]

        >>> t1.branch_tree.tolist()
        [-1, 0, 1, 0]

        """  # noqa: E501
        tree = self.copy() if not inplace else self
        branch_nodes = np.atleast_2d(branch_nodes).astype(int)
        _, new_branch_ids = super(VTree, tree).add_branch(branch_nodes, return_branch_id=True, inplace=True)

        tree._branch_tree = np.concatenate([tree._branch_tree, np.full(len(new_branch_ids), -1, dtype=int)])
        if tree._branch_dir is not None:
            tree._branch_dir = np.concatenate([tree._branch_dir, np.ones(len(new_branch_ids), dtype=bool)])

        # Update the branch tree
        for b_id, (tail, head) in zip(new_branch_ids, branch_nodes, strict=True):
            incoming_tail_branch = tree.node_incoming_branches(tail)
            if len(incoming_tail_branch) == 1:
                tree._branch_tree[b_id] = incoming_tail_branch[0]

            outgoing_head_branch = tree.node_outgoing_branches(head)
            if len(outgoing_head_branch) == 1:
                tree._branch_tree[outgoing_head_branch[0]] = b_id

        return (tree, new_branch_ids) if return_branch_id else tree

    @overload
    def split_node(
        self,
        node: NodeIndex,
        branch_connectivity: List[List[int]],
        *,
        inplace=False,
        return_node_ids: Literal[False] = False,
    ) -> Self: ...
    @overload
    def split_node(
        self, node: NodeIndex, branch_connectivity: List[List[int]], *, inplace=False, return_node_ids: Literal[True]
    ) -> Tuple[Self, npt.NDArray[np.int32]]: ...
    def split_node(
        self, node: NodeIndex, branch_connectivity: List[List[int]], *, inplace=False, return_node_ids: bool = False
    ) -> Self | Tuple[Self, npt.NDArray[np.int32]]:
        """Split a node into multiple nodes according to the given branch connectivity.

        Parameters
        ----------
        node_id : int
            The index of the node to split.

        branch_connectivity : List[List[int]]
            A list of lists, where each sublist contains the indices of the branches that will stay connected together through a new node.

        inplace : bool, optional
            If True, the tree is modified in place. Otherwise, a new tree is returned.

        Returns
        -------
        VTree
            The modified tree.
        """  # noqa: E501
        tree = self.copy() if not inplace else self

        branch_tree = tree._branch_tree
        branch_clusters = [tree.as_branch_ids(cluster) for cluster in branch_connectivity]
        all_branches = np.concatenate(branch_clusters)
        _, new_node_ids = super(VTree, tree).split_node(node, branch_connectivity, inplace=True, return_node_ids=True)

        # Remove parent dependency if branches are in different clusters
        for cluster in branch_clusters:
            for b in cluster:
                if branch_tree[b] in all_branches and branch_tree[b] not in cluster:
                    branch_tree[b] = -1
        tree._branch_tree = branch_tree

        return (tree, new_node_ids) if return_node_ids else tree

    def delete_node(self, node_id: NodeIndicesLike, *, inplace: bool = False) -> VTree:
        """Remove the nodes with the given indices from the tree.

        Parameters
        ----------
        node_indexes : IndexLike
            The indices of the nodes to remove.

        inplace : bool, optional
            If True (by default), the tree is modified in place. Otherwise, a new tree is returned.

        Returns
        -------
        VTree
            The modified tree.
        """
        tree = self.copy() if not inplace else self
        super(tree.__class__, tree).delete_node(node_id, inplace=True)
        return tree

    def fuse_node(
        self,
        node_id: NodeIndicesLike,
        *,
        quietly_ignore_invalid_nodes=False,
        inplace=False,
        incident_branches: Optional[IntPairArrayLike] = None,
    ) -> Self:
        """Fuse nodes connected to exactly two branches.

        The nodes are removed from the tree and their corresponding branches are merged.

        Parameters
        ----------
        nodes : IndexLike
            Array of indices of the nodes to fuse.

        quietly_ignore_invalid_nodes : bool, optional
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
        ids = self.as_node_ids(node_id)

        if not quietly_ignore_invalid_nodes:
            assert (
                len(invalid_nodes := np.where((self.node_indegree(ids) != 1) | (self.node_outdegree(ids) != 1))[0]) == 0
            ), (
                f"The nodes {ids[invalid_nodes]} can't be fuse: "
                "they don't have exactly one incoming branch and one outgoing branch."
            )
        else:
            ids = ids[(self.node_indegree(ids) == 1) & (self.node_outdegree(ids) == 1)]

        tree = self.copy() if not inplace else self

        branch_tree = tree.branch_tree
        branch_rank = invert_complete_lookup(np.array(list(self.walk_branch_ids(traversal="bfs")), dtype=int))

        # === Fuse the node in the graph and geometrical data ===
        merged_branch, flip_branch, del_lookup, del_branch = tree._fuse_nodes(
            ids, quietly_ignore_invalid_nodes=quietly_ignore_invalid_nodes, incident_branches=incident_branches
        )

        # === Redirect the branch ancestors and direction===
        branch_dir: npt.NDArray[np.bool_] = np.ones(tree.branch_count, dtype=bool)
        if any(flip_branch):
            if tree._branch_dir is None:
                tree._branch_dir = branch_dir
            else:
                branch_dir = tree._branch_dir

        ancestors_lookup = np.arange(branch_tree.size)
        for branches, flip in zip(merged_branch, flip_branch, strict=True):
            b0 = np.argmin(branch_rank[branches])  # Select the lower ancestor of all merged branches
            ancestors_lookup[branches[:b0]] = branches[b0]
            ancestors_lookup[branches[b0:]] = branches[b0]
            if flip:
                main_branch = del_lookup[branches[b0]]
                branch_dir[main_branch] = ~branch_dir[main_branch]
        branch_tree = branch_tree[ancestors_lookup]

        # === Apply branch deletion to branch tree ===
        del_lookup = add_empty_to_lookup(del_lookup, increment_index=False)
        tree._branch_tree = np.delete(del_lookup[branch_tree + 1], del_branch)
        return tree

    def merge_consecutive_branches(
        self,
        branch_pairs: IntPairArrayLike,
        junction_nodes: Optional[Int1DArrayLike] = None,
        *,
        remove_orphan_nodes: bool = True,
        quietly_ignore_invalid_pairs: Optional[bool] = False,
        inplace=False,
    ) -> VTree:
        """Merge consecutive branches in the tree.

        Parameters
        ----------
        branch_pairs :
            The pairs of consecutive branches to merge.

        junction_nodes :
            The indices of the junction nodes connecting each pair of branches. If None, the junction nodes are inferred from the branches tree.

        remove_orphan_nodes : bool, optional
            If True, the nodes that are not connected to any branch after the deletion are removed from the tree.

        quietly_ignore_invalid_pairs : bool, optional
            If True, ignore any branch pairs not consecutive.
            If False, raise an error if such branches are found.
            If None, assumes that the provided branch pairs are valid and sorted as [parent, child].

        inplace : bool, optional
            If True, the tree is modified in place. Otherwise, a new tree is returned.

        Returns
        -------
        VTree
            The modified tree.
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
            branch_pairs,
            junction_nodes,  # type: ignore
            remove_orphan_nodes=remove_orphan_nodes,
        )

        # === Redirect the branch ancestors and direction===
        branch_dir: npt.NDArray[np.bool_] = np.ones(tree.branch_count, dtype=bool)
        if any(flip_branch):
            if tree._branch_dir is None:
                tree._branch_dir = branch_dir
            else:
                branch_dir = tree._branch_dir

        ancestors_lookup = np.arange(branch_tree.size)
        for branches, flip in zip(merged_branch, flip_branch, strict=True):
            b0 = branches[0]
            ancestors_lookup[branches] = b0
            if flip:
                main_branch = merge_lookup[b0]
                branch_dir[main_branch] = ~branch_dir[main_branch]

        # === Apply branch deletion to branch tree ===
        branch_tree = branch_tree[ancestors_lookup]
        merge_lookup = add_empty_to_lookup(merge_lookup, increment_index=False)
        tree._branch_tree = np.delete(merge_lookup[branch_tree + 1], del_branch)
        return tree

    @overload
    def merge_nodes(
        self,
        clusters: Iterable[Iterable[int]],
        *,
        nodes_weight: Optional[npt.NDArray[np.float32]] = None,
        inplace=False,
        assume_reduced=False,
        return_branch_reindex_lookup: Literal[False] = False,
    ) -> Self: ...
    @overload
    def merge_nodes(
        self,
        clusters: Iterable[Iterable[int]],
        *,
        nodes_weight: Optional[npt.NDArray[np.float32]] = None,
        inplace=False,
        assume_reduced=False,
        return_branch_reindex_lookup: Literal[True],
    ) -> tuple[Self, Indices]: ...
    def merge_nodes(
        self,
        clusters: Iterable[Iterable[int]],
        *,
        nodes_weight: Optional[npt.NDArray[np.float32]] = None,
        inplace=False,
        assume_reduced=False,
        return_branch_reindex_lookup: bool = False,
    ) -> Self | tuple[Self, Indices]:
        """Merge a cluster of nodes into a single node.

        The node with the smallest index is kept and the others are removed from the graph. The branches inside the clusters are removed, the branches incident to the cluster are connected to the kept node.

        The position of the kept node is the average of the positions of the merged nodes.

        If the resulting node is not connected to any branch, it is removed from the graph.

        Parameters
        ----------
        clusters : Iterable[int]
            The indices of the nodes to merge.

        nodes_weight : np.ndarray, optional
            The weight of each node. If provided, the nodes position are weighted by this array to compute the new position of the kept node.

        inplace : bool, optional
            If True (by default), the graph is modified in place. Otherwise, a new graph is returned.

        assume_reduced : bool, optional
            If True, the clusters are assumed to be reduced (i.e. each node appears in only one cluster).

        return_branch_reindex_lookup : bool, optional
            If True, also return the lookup table to reindex the branches after the node merging. Default is False.

        Returns
        -------
        VTree
            The modified tree.

        branch_reindex_lookup : np.ndarray
            The lookup table to reindex the branches after the node merging. Only returned if ``return_branch_reindex_lookup`` is True.

        Examples
        --------
        >>> # branch ids:        0 1    2 3    4 5
        >>> tree = VTree.parse("0➔1➔6;2➔1➔3;4➔6➔5")
        >>> tree.branch_tree.tolist()
        [-1, 0, -1, 2, -1, 4]
        >>> tree.merge_nodes([{1, 6}], inplace=True).branch_list.tolist()
        [[0, 1], [2, 1], [1, 3], [4, 1], [1, 5]]
        >>> tree.branch_tree.tolist()
        [-1, -1, 1, -1, 3]
        >>>
        """  # noqa: E501
        old_branch_tree = self._branch_tree.copy()
        tree, branch_reindex_lookup = super(VTree, self).merge_nodes(
            clusters,
            nodes_weight=nodes_weight,
            assume_reduced=assume_reduced,
            return_branch_reindex_lookup=True,
            inplace=inplace,
        )

        # === Redirect branch outgoing from the cluster to the ones incoming to it ===
        removed_branch_mask = branch_reindex_lookup == -1
        removed_branch_ids = np.where(removed_branch_mask)[0]
        redirection = removed_branch_ids.copy()
        while np.any(np.isin(redirection, removed_branch_ids)):
            redirection = old_branch_tree[redirection]
        branch_reindex_lookup[removed_branch_ids] = redirection
        branch_reindex_lookup = add_empty_to_lookup(branch_reindex_lookup, increment_index=False)

        tree._branch_tree = branch_reindex_lookup[old_branch_tree[~removed_branch_mask] + 1]

        return (tree, branch_reindex_lookup) if return_branch_reindex_lookup else tree

    @overload
    def split_branch(
        self,
        branch_id: int,
        split_curve_id: Int1DArrayLike | Float1DArrayLike,
        split_coord: Optional[PointArrayLike] = None,
        *,
        return_branch_ids: Literal[False] = False,
        return_node_ids: Literal[False] = False,
        inplace=False,
    ) -> Self: ...
    @overload
    def split_branch(
        self,
        branch_id: int,
        split_curve_id: Int1DArrayLike | Float1DArrayLike,
        split_coord: Optional[PointArrayLike] = None,
        *,
        return_branch_ids: Literal[True],
        return_node_ids: Literal[False] = False,
        inplace=False,
    ) -> Tuple[Self, npt.NDArray[np.int32]]: ...
    @overload
    def split_branch(
        self,
        branch_id: int,
        split_curve_id: Int1DArrayLike | Float1DArrayLike,
        split_coord: Optional[PointArrayLike] = None,
        *,
        return_branch_ids: Literal[False] = False,
        return_node_ids: Literal[True],
        inplace=False,
    ) -> Tuple[Self, npt.NDArray[np.int32]]: ...
    @overload
    def split_branch(
        self,
        branch_id: int,
        split_curve_id: Int1DArrayLike | Float1DArrayLike,
        split_coord: Optional[PointArrayLike] = None,
        *,
        return_branch_ids: Literal[True],
        return_node_ids: Literal[True],
        inplace=False,
    ) -> Tuple[Self, npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...
    def split_branch(
        self,
        branch_id: int,
        split_curve_id: Int1DArrayLike | Float1DArrayLike,
        split_coord: Optional[PointArrayLike] = None,
        *,
        return_branch_ids=False,
        return_node_ids=False,
        inplace=False,
    ) -> Self | Tuple[Self, npt.NDArray[np.int_]] | Tuple[Self, npt.NDArray[np.int_], npt.NDArray[np.int_]]:
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
        tree = self if inplace else self.copy()

        branch_flipped = False if tree._branch_dir is None else not tree._branch_dir[branch_id]

        _, new_branch_ids, new_nodes_ids = super(VTree, tree).split_branch(
            branch_id, split_curve_id, split_coord, return_branch_ids=True, return_node_ids=True, inplace=True
        )

        # === Update branch tree ===
        if tree._branch_dir is not None:
            tree._branch_dir = np.append(tree._branch_dir, np.full(len(new_branch_ids) - 1, branch_flipped))
        branch_tree = tree.branch_tree.copy()

        if not branch_flipped:
            branch_tree[branch_tree == branch_id] = new_branch_ids[-1]  # Redirect child to the last segment
            branch_tree = np.concatenate([branch_tree, new_branch_ids[:-1]])  # Add parent of the new segments
            assert np.all(branch_tree != np.arange(branch_tree.size)), "Branch tree is corrupted after split_branch."
        else:
            parent = branch_tree[branch_id]
            branch_tree[branch_id] = new_branch_ids[1]  # Redirect branch parent to the second segment
            branch_tree = np.concatenate([branch_tree, new_branch_ids[2:], [parent]])  # Add parent of the new segments
        tree._branch_tree = branch_tree

        if return_branch_ids:
            return (tree, new_branch_ids, new_nodes_ids) if return_node_ids else (tree, new_branch_ids)
        else:
            return (tree, new_nodes_ids) if return_node_ids else tree

    def bridge_nodes(self, node_pairs: IntPairArrayLike, *, fuse_nodes=False, check=True, inplace=False) -> Self:
        """Bridge the two given nodes with a new branch.

        Parameters
        ----------
        node_pairs :
            Array of shape (P, 2) containing P pairs of indices of the nodes to link.

        fuse_nodes : bool, optional
            If True, the nodes are fused together instead of being linked by a new branch.

        check : bool, optional
            If True, check the pairs to ensure that the nodes are not already connected by a branch.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

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
        ids: Optional[BranchIndicesLike] = None,
        /,
        *,
        filter: Optional[Literal["orphan", "endpoint", "non-endpoint"]] = None,
        dynamic_iterator: bool = False,
    ) -> Generator[VTreeBranch]:
        """Iterate over the branches of a tree, encapsulated in :class:`VTreeBranch` objects.

        Parameters
        ----------
        ids : int or npt.ArrayLike[int], optional
            The indices of the branches to iterate over. If None, iterate over all branches.

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
        return super().branches(ids, filter=filter, dynamic_iterator=dynamic_iterator)  # type: ignore

    def branch(self, branch_id: int, /) -> VTreeBranch:
        return super().branch(branch_id)  # type: ignore

    def root_branches(self) -> Generator[VTreeBranch]:
        """Iterate over the root branches of the tree, encapsulated in :class:`VTreeBranch` objects.

        Returns
        -------
        Generator[VTreeBranch]
            A generator that yields the root branches.
        """
        for b in self.root_branch_ids():
            yield VTreeBranch(self, b)

    def walk_branches(
        self, root_branch_id: Optional[int] = None, traversal: Literal["dfs", "bfs"] = "bfs", dynamic=False
    ) -> Generator[VTreeBranch]:
        """Create a walker object to traverse the tree from the given node.

        Parameters
        ----------
        branch_id : int
            The index of the branch to start the walk from.

        traversal : str, optional
            The traversal algorithm to use. Either "dfs" for depth-first search or "bfs" for breadth-first search.

        dynamic : bool, optional
            If True, iterate over all the branches present in the tree when this method is called.
            All branches added during the iteration will be ignored. Branches reindexed during the iteration will be visited in their original order at the time of the call. Deleted branches will not be visited.

            Enable this option if you plan to modify the tree during the iteration. If you only plan to read the tree, disable this option for better performance.

        Returns
        -------
        Generator[int]
            A generator that yields the indices of the child branches.
        """  # noqa: E501

        if dynamic:
            for b in list(self.walk_branches(root_branch_id, traversal=traversal, dynamic=False)):
                yield b
            return

        depth_first = traversal == "dfs"

        stack = list(self.root_branch_ids()) if root_branch_id is None else [root_branch_id]
        while stack:
            branch_id = stack.pop() if depth_first else stack.pop(0)
            branch_children = self.branch_successors(branch_id)
            branch_dir = self._branch_dir[branch_id] if self._branch_dir is not None else True
            branch = VTreeBranch(self, branch_id, dir=branch_dir, nodes_id=self._branch_list[branch_id])
            branch._succ_branch_ids = branch_children
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

    def walk_nodes(
        self, node_id: Optional[NodeIndicesLike] = None, traversal: Literal["dfs", "bfs"] = "bfs"
    ) -> Generator[VTreeNode]:
        """Create a walker object to traverse the tree from the given node.

        Parameters
        ----------
        node_id : int
            The index of the node to start the walk from.

        traversal : str, optional
            The traversal algorithm to use. Either "dfs" for depth-first search or "bfs" for breadth-first search.

        Returns
        -------
        Generator[int]
            A generator that yields the indices of the child nodes.
        """
        branch_list = self.tree_branch_list()

        if node_id is None:
            outgoing_branches = self.root_branch_ids()
            if traversal == "dfs":
                outgoing_branches = outgoing_branches[::-1]
        else:
            node_id = self.as_node_ids(node_id)
            outgoing_branches = np.argwhere(np.isin(branch_list[:, 0], node_id)).flatten()

        for b in self.walk_branch_ids(outgoing_branches, traversal=traversal, ignore_provided_id=False):
            yield VTreeNode(self, branch_list[b, 1])

    def node(self, node_id: int, /) -> VTreeNode:
        return super().node(node_id)  # type: ignore

    def nodes(
        self,
        node_ids: Optional[NodeIndicesLike] = None,
        /,
        *,
        only_degree: Optional[Int1DArrayLike] = None,
        only_outdegree: Optional[Int1DArrayLike] = None,
        dynamic_iterator: bool = False,
    ) -> Generator[VTreeNode]:
        """Iterate over the nodes of the tree encapsulated in :class:`VTreeNode` objects.

        Parameters
        ----------
        ids : Optional[int  |  npt.ArrayLike[int]], optional
            The indices of the nodes to iterate over. If None, iterate over all the nodes.

        only_degree : Optional[int  |  npt.ArrayLike[np.int_]], optional
            If provided, only iterate over the nodes with the given total degree.

        only_outdegree : Optional[int  |  npt.ArrayLike[np.int_]], optional
            If provided, only iterate over the nodes with the given outdegree.

        dynamic_iterator : bool, optional
            If True, iterate over all the branches present in the tree when this method is called.
            All branches added during the iteration will be ignored. Branches reindexed during the iteration will be visited in their original order at the time of the call. Deleted branches will not be visited.

            Enable this option if you plan to modify the tree during the iteration. If you only plan to read the tree, disable this option for better performance.

        Yields
        ------
        Generator[VTreeNode]
            The nodes of the tree as :class:`VTreeNode` objects.
        """  # noqa: E501
        # Filter nodes by outdegree
        if only_outdegree is not None:
            node_ids = self.as_node_ids(node_ids) if node_ids is not None else np.arange(self.node_count)
            node_ids = np.intersect1d(node_ids, np.argwhere(np.isin(self.node_outdegree(), only_outdegree)).flatten())
        return super().nodes(node_ids, only_degree=only_degree, dynamic_iterator=dynamic_iterator)  # type: ignore
