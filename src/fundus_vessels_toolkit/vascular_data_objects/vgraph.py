from __future__ import annotations

__all__ = ["VGraph"]

import itertools
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Literal, Optional, Tuple, Type, overload
from weakref import WeakSet

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils import if_none
from ..utils.bezier import BSpline
from ..utils.cluster import reduce_chains, reduce_clusters
from ..utils.data_io import NumpyDict, load_numpy_dict, pandas_to_numpy_dict, save_numpy_dict
from ..utils.fundus_projections import FundusProjection
from ..utils.geometric import Point, Rect
from ..utils.graph.branch_by_nodes import (
    branch_list_to_branches_by_nodes,
    branches_by_nodes_to_branch_list,
)
from ..utils.lookup_array import add_empty_to_lookup, complete_lookup, create_removal_lookup, invert_complete_lookup
from ..utils.pandas import DFSetterAccessor
from .vgeometric_data import VBranchGeoData, VBranchGeoDataKey, VGeometricData


########################################################################################################################
#  === GRAPH NODES AND BRANCHES ACCESSORS CLASSES ===
########################################################################################################################
class VGraphNode:
    """A class representing a node in a vascular network graph.
    :class:`VGraphNode` objects are only references to graph's nodes and do not store any data.
    All their method and properties makes calls to :class:`VGraph` methods to generate their results.
    """

    def __init__(self, graph: VGraph, id: int):
        self.__graph = graph
        self._id = id
        graph._nodes_refs.add(self)

        self._ibranch_ids = None
        self._ibranch_dirs = None

    def __repr__(self):
        return f"VGraphNode({self._id})"

    def __str__(self):
        return f"Node[{self._id}]"

    def __hash__(self):
        return hash(("N", self._id))

    def __eq__(self, other):
        return (
            self._id >= 0 and isinstance(other, VGraphNode) and self.__graph is other.__graph and self._id == other._id
        )

    def is_valid(self) -> bool:
        return self._id >= 0

    @property
    def graph(self) -> VGraph:
        return self.__graph

    @property
    def id(self) -> int:
        if self._id < 0:
            raise RuntimeError("The node has been removed from the graph.")
        return int(self._id)

    @property
    def attr(self) -> DFSetterAccessor:
        if not self.is_valid():
            raise RuntimeError("The node has been removed from the graph.")
        return DFSetterAccessor(self.__graph._nodes_attr, self._id)

    def coord(self, geodata: Optional[VGeometricData | int] = None) -> Point:
        if not self.is_valid():
            raise RuntimeError("The node has been removed from the graph.")
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)
        return Point(*geodata.nodes_coord(self._id))

    #  __ INCIDENT BRANCHES __
    def _update_incident_branches_cache(self):
        self._ibranch_ids, self._ibranch_dirs = self.graph.incident_branches(self._id, return_branch_direction=True)

    def clear_incident_branch_cache(self):
        self._ibranch_ids = None
        self._ibranch_dirs = None

    @property
    def branches_ids(self) -> List[int]:
        if not self.is_valid():
            return []
        if self._ibranch_ids is None:
            self._update_incident_branches_cache()
        return [int(_) for _ in self._ibranch_ids]

    def branches(self) -> Iterable[VGraphBranch]:
        """Return the branches incident to this node.

        ..warning::
            The returned :class:`VGraphBranch` objects are linked to their branch and not to this node. After merging or splitting this node or its branches, the original returned branches may not be incident to this node anymore.
        """  # noqa: E501
        if not self.is_valid():
            return ()
        if self._ibranch_ids is None:
            self._update_incident_branches_cache()
        return (VGraphBranch(self.__graph, i) for i in self._ibranch_ids)

    @property
    def degree(self) -> int:
        if not self.is_valid():
            return 0
        if self._ibranch_ids is None:
            self._update_incident_branches_cache()
        return len(self._ibranch_ids)

    def adjacent_nodes(self) -> Iterable[VGraphNode]:
        """Return the nodes connected to this node by an incident branch.

        ..warning::
            The returned :class:`VGraphNode` objects are linked to their node and not to this node adjacency. After merging or splitting this node or its incident branches, the original returned nodes may not be connected to this node anymore.
        """  # noqa: E501
        if not self.is_valid():
            return ()
        if self._ibranch_ids is None:
            self._update_incident_branches_cache()
        return (
            VGraphNode(self.__graph, self.__graph.branch_list[branch][1 if dir else 0])
            for branch, dir in zip(self._ibranch_ids, self._ibranch_dirs, strict=True)
        )


class VGraphBranch:
    """A class representing a branch in a vascular network graph.
    :class:`VGraphBranch` objects are only references to graph's branches and do not store any data.
    All their method and properties makes calls to :class:`VGraph` methods to generate their results.
    """

    def __init__(self, graph: VGraph, id: int, nodes_id: Optional[npt.NDArray[np.int_]] = None):
        self.__graph = graph
        self._id = id
        graph._branches_refs.add(self)

        self._nodes_id = graph.branch_list[id] if nodes_id is None else nodes_id

    def __repr__(self):
        return f"VGraphBranch({self._id})"

    def __str__(self):
        return f"Branch[{self._id}]"

    def __hash__(self):
        return hash(("B", self._id))

    def __eq__(self, other):
        return (
            self._id >= 0
            and isinstance(other, VGraphBranch)
            and self.__graph is other.__graph
            and self._id == other._id
        )

    def is_valid(self) -> bool:
        return self._id >= 0

    @property
    def id(self) -> int:
        if self._id < 0:
            raise RuntimeError("The branch has been removed from the graph.")
        return int(self._id)

    @property
    def graph(self) -> VGraph:
        return self.__graph

    @property
    def nodes_id(self) -> Tuple[int, int]:
        """The indexes of the nodes connected by the branch as a tuple."""
        return tuple(int(_) for _ in self._nodes_id)

    def nodes(self) -> Tuple[VGraphNode, VGraphNode]:
        """The two nodes connected by this branch.  # noqa: E501

        ..warning::
            The returned :class:`VGraphNode` objects are linked to their node and not to this branch tip. After merging, splitting or flipping this branch, the original returned nodes may not be connected by this branch anymore.
        """  # noqa: E501
        n1, n2 = self._nodes_id
        return VGraphNode(self.__graph, n1), VGraphNode(self.__graph, n2)

    @property
    def attr(self) -> DFSetterAccessor:
        return DFSetterAccessor(self.__graph._branches_attr, self._id)

    def curve(self, geodata: Optional[VGeometricData | int] = None) -> npt.NDArray[np.int_]:
        """Return the indexes of the pixels forming the curve of the branch as a 2D array of shape (n, 2).

        This method is a shortcut to :meth:`VGeometricData.branch_curve`.

        Parameters
        ----------
        geo_data : int, optional
            The index of the geometrical_data, by default 0

        Returns
        -------
        np.ndarray
            The indexes of the pixels forming the curve of the branch.
        """
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)
        return geodata.branch_curve(self._id)

    def bspline(self, geodata: Optional[VGeometricData | int] = None) -> BSpline:
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)
        return geodata.branch_bspline(self._id)

    def geodata(self, attr_name: VBranchGeoDataKey, geo_data=0) -> VBranchGeoData.Base | Dict[str, VBranchGeoData.Base]:
        """Access the geometric data of the branch.

        This method is a shortcut to :meth:`VGeometricData.branch_data`.

        Parameters
        ----------
        attr_name : VBranchGeoDataKey
            The name of the attribute to access.
        geo_data : int, optional
            The index of the geometrical_data, by default 0

        Returns
        -------
        Any
            The value of the attribute.
        """
        return self.__graph.geometric_data(geo_data).branch_data(attr_name, self._id)

    def node_to_node_length(self, geodata: Optional[VGeometricData | int] = None) -> float:
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(0 if geodata is None else geodata)
        yx1, yx2 = geodata.nodes_coord(self._nodes_id)
        return np.linalg.norm(yx1 - yx2)

    def arc_length(self, geodata: Optional[VGeometricData | int] = None) -> float:
        return len(self.curve(geodata))


########################################################################################################################
#  === VASCULAR GRAPH CLASS ===
########################################################################################################################
class VGraph:
    """
    A class to represent a graph of a vascular network.

    This class provides tools to store, retrieve and visualise the data associated with a vascular network.
    """

    def __init__(
        self,
        branch_list: np.ndarray,
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
        # === Check and store branches list ===
        assert (
            branch_list.ndim == 2 and branch_list.shape[1] == 2
        ), "branch_list must be a 2D array of shape (B, 2) where B is the number of branches"
        self._branch_list = branch_list
        B = branch_list.shape[0]

        # === Infer nodes_count ===
        if nodes_count is None:
            nodes_indexes = np.unique(branch_list)
            assert len(nodes_indexes) == nodes_indexes[-1] + 1, (
                f"The branches list must contain every node id at least once (from 0 to {len(nodes_indexes)-1}).\n"
                f"\t Nodes {sorted([int(_) for _ in np.setdiff1d(np.arange(len(nodes_indexes)), nodes_indexes)]) }"
                " are missing."
            )
            N = len(nodes_indexes)
        else:
            N = nodes_count
        self._nodes_count = N

        if nodes_attr is None:
            nodes_attr = pd.DataFrame(index=pd.RangeIndex(N))
        self._nodes_attr = nodes_attr
        if branches_attr is None:
            branches_attr = pd.DataFrame(index=pd.RangeIndex(B))
        self._branches_attr = branches_attr

        if isinstance(geometric_data, VGeometricData):
            geometric_data = [geometric_data]

        for i, gdata in enumerate(geometric_data):
            assert isinstance(
                gdata, VGeometricData
            ), "geometric_data must be a VGeometricData object or a list of VGeometricData objects."
            if gdata._parent_graph is not None:
                geometric_data[i] = gdata.copy(self)
            else:
                gdata.parent_graph = self
        self._geometric_data: List[VGeometricData] = list(geometric_data)

        if check_integrity or nodes_count is None:
            self.check_integrity()

        self._nodes_refs: WeakSet[VGraphNode] = WeakSet()
        self._branches_refs: WeakSet[VGraphBranch] = WeakSet()

    def __getstate__(self) -> object:
        d = self.__dict__.copy()
        d.drop("_vgraph_nodes", None)
        d.drop("_vgraph_branches", None)
        return d

    def check_integrity(self):
        """Check the integrity of the graph data.

        This method checks that all branches and nodes index are consistent.

        Raises
        ------
        ValueError
            If the graph data is not consistent.
        """
        N, B = self.nodes_count, self.branches_count

        # --- Check geometric data ---
        branches_idx = set()
        nodes_idx = set()
        for gdata in self._geometric_data:
            branches_idx.update(gdata.branches_id)
            nodes_idx.update(gdata.nodes_id)

        branches_idx.difference_update(np.arange(B))
        assert (
            len(branches_idx) == 0
        ), f"Geometric data contains branches indexes that are not in the branch list: {branches_idx}."

        nodes_idx.difference_update(np.arange(N))
        assert (
            len(nodes_idx) == 0
        ), f"Geometric data contains nodes indexes that are above the nodes count: {nodes_idx}."

        # --- Check nodes attributes ---
        assert (
            self._nodes_attr.index.inferred_type == "integer"
        ), "The index of nodes_attr dataframe must be nodes Index."
        assert self._nodes_attr.index.max() < N and self._nodes_attr.index.min() >= 0, (
            "The maximum value in nodes_attr index must be lower than the number of nodes."
            f" Got {self._nodes_attr.index.max()} instead of {N}"
        )
        if len(self._nodes_attr.index) != N:
            self._nodes_attr.reindex(np.arange(N))

        # --- Check branches attributes ---
        assert (
            self._branches_attr.index.inferred_type == "integer"
        ), "The index of branches_attr dataframe must be branches Index."
        assert self._branches_attr.index.max() < B and self._branches_attr.index.min() >= 0, (
            "The maximum value in branches_attr index must be lower than the number of branches."
            f" Got {self._branches_attr.index.max()} instead of {B}"
        )
        if len(self._branches_attr.index) != B:
            self._branches_attr.reindex(np.arange(B))

    @classmethod
    def from_branch_by_nodes(
        cls,
        branch_by_nodes: npt.NDArray[np.bool_],
        geometric_data: VGeometricData | Iterable[VGeometricData],
        nodes_attr: Optional[pd.DataFrame] = None,
        branches_attr: Optional[pd.DataFrame] = None,
        integrity_check: bool = True,
    ) -> VGraph:
        """Create a Graph object from a branch-by-nodes connectivity matrix instead of a branch list.

        Parameters
        ----------
        branch_by_nodes : np.ndarray
            A 2D array of shape (B, N) where N is the number of nodes and B is the number of branches. Each True value in the array indicates that the branch is connected to the corresponding node. Such matrix should therefore contains exactly two True values per row.

        geometric_data : VGeometricData or Iterable[VGeometricData]
            The geometric data associated with the graph.

        nodes_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each node. The index of the dataframe must be the nodes indexes.

        branches_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each branch. The index of the dataframe must be the branches indexes.

        Returns
        -------
            The Graph object created from the given data.
        """  # noqa: E501
        branch_list = branches_by_nodes_to_branch_list(branch_by_nodes)
        graph = cls(branch_list, geometric_data, nodes_attr, branches_attr, nodes_count=branch_by_nodes.shape[1])
        if integrity_check:
            graph.check_integrity()
        return graph

    def copy(self) -> VGraph:
        """Create a copy of the current Graph object."""
        return VGraph(
            self._branch_list.copy(),
            [gdata.copy(None) for gdata in self._geometric_data],
            self._nodes_attr.copy(),
            self._branches_attr.copy(),
            self._nodes_count,
        )

    def save(self, filename: Optional[str | Path] = None) -> NumpyDict:
        """Save the graph data to a file.

        The graph is saved as a dictionary with the following keys:
            - ``branch_list``: The list of branches in the graph as a 2D array of shape (B, 2) where B is the number of branches. Each row contains the indexes of the nodes connected by each branch.
            - ``geometric_data``: The geometric data associated with the graph as a list of dictionaries.
            - ``nodes_attr``: The attributes of the nodes in the graph as a dictionary.
            - ``branches_attr``: The attributes of the branches in the graph as a dictionary.

        Parameters
        ----------
        filename : str or Path, optional
            The name of the file to save the data to. If None, the data is not saved to a file.

        Returns
        -------
        NUMPY_DICT
            The graph as a dictionary of numpy arrays.
        """  # noqa: E501
        data = dict(
            branch_list=self._branch_list,
            geometric_data=[gdata.save() for gdata in self._geometric_data],
            nodes_attr=pandas_to_numpy_dict(self._nodes_attr),
            branches_attr=pandas_to_numpy_dict(self._branches_attr),
        )

        if filename is not None:
            save_numpy_dict(data, filename)
        return data

    @classmethod
    def load(cls, filename: str | Path | NumpyDict) -> VGraph:
        """Load a Graph object from a file.

        Parameters
        ----------
        filename : str or Path
            The name of the file to load the data from.

        Returns
        -------
        VGraph
            The Graph object loaded from the file.
        """
        if isinstance(filename, (str, Path)):
            data = load_numpy_dict(filename)
        else:
            data = filename

        return cls(
            data["branch_list"],
            [VGeometricData.load(d) for d in data["geometric_data"]] if "geometric_data" in data else [],
            pd.DataFrame(data["nodes_attr"]) if "nodes_attr" in data else None,
            pd.DataFrame(data["branches_attr"]) if "branches_attr" in data else None,
        )

    ####################################################################################################################
    #  === PROPERTIES ===
    ####################################################################################################################
    @property
    def nodes_count(self) -> int:
        """The number of nodes in the graph."""
        return self._nodes_count

    @property
    def branches_count(self) -> int:
        """The number of branches in the graph."""
        return self._branch_list.shape[0]

    @property
    def branch_list(self) -> npt.NDArray[np.int_]:
        """The list of branches in the graph as a 2D array of shape (B, 2) where B is the number of branches. Each row contains the indexes of the nodes connected by each branch."""  # noqa: E501
        return self._branch_list

    def geometric_data(self, id=0) -> VGeometricData:
        """The geometric data associated with the graph.

        Parameters
        ----------
        id : int, optional
            The index of the geometric data to retrieve. Default is 0.

        Returns
        -------
        VGeometricData
            The geometric data associated with the graph.
        """
        return self._geometric_data[id]

    @property
    def nodes_attr(self) -> pd.DataFrame:
        """The attributes of the nodes in the graph as a pandas.DataFrame."""
        return self._nodes_attr

    @property
    def branches_attr(self) -> pd.DataFrame:
        """The attributes of the branches in the graph as a pandas.DataFrame."""
        return self._branches_attr

    ####################################################################################################################
    #  === COMPUTABLE GRAPH PROPERTIES ===
    ####################################################################################################################
    def branches_by_nodes(self) -> npt.NDArray[np.bool_]:
        """Compute the branches-by-nodes connectivity matrix from this graph branch list.

        The branches-by-nodes connectivity matrix is a 2D boolean matrix of shape (B, N) where B is the number of branches and N is the number of nodes. Each True value in the array indicates that the branch is connected to the corresponding node. This matrix therefore contains exactly two True values per row.

        Returns
        -------
        np.ndarray
            A 2D boolean array of shape (B, N) where N is the number of nodes and B is the number of branches.
        """  # noqa: E501
        return branch_list_to_branches_by_nodes(
            self._branch_list, n_branches=self.branches_count, n_nodes=self.nodes_count
        )

    def branch_adjacency_matrix(self) -> npt.NDArray[np.bool_]:
        """Compute the branch adjacency matrix from this graph branches-by-nodes connectivity matrix.

        The branch adjacency matrix is a 2D boolean matrix of shape (B, B) where B is the number of branches. Each True value in the array indicates that the two branches are connected by a node. This matrix is symmetric.

        Returns
        -------
        np.ndarray
            A 2D boolean array of shape (B, B) where B is the number of branches.
        """  # noqa: E501
        branch_by_node = self.branches_by_nodes()
        return (branch_by_node @ branch_by_node.T) > 0

    def node_adjacency_matrix(self) -> npt.NDArray[np.bool_]:
        """Compute the node adjacency matrix from this graph branch list.

        The node adjacency matrix is a 2D boolean matrix of shape (N, N) where N is the number of nodes. Each True value in the array indicates that the two nodes are connected by a branch. This matrix is symmetric.

        Returns
        -------
        np.ndarray
            A 2D boolean array of shape (N, N) where N is the number of nodes.
        """  # noqa: E501
        node_adj = np.zeros((self.nodes_count, self.nodes_count), dtype=bool)
        node_adj[self._branch_list[:, 0], self._branch_list[:, 1]] = True
        node_adj[self._branch_list[:, 1], self._branch_list[:, 0]] = True
        return node_adj

    @overload
    def incident_branches(
        self, node_idx: int | npt.ArrayLike[int], return_branch_direction: Literal[False]
    ) -> npt.NDArray[np.int_]: ...
    @overload
    def incident_branches(
        self, node_idx: int | npt.ArrayLike[int], return_branch_direction: Literal[True]
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]: ...
    def incident_branches(
        self,
        node_idx: int | npt.ArrayLike[int],
        return_branch_direction: bool = False,
    ) -> npt.NDArray[np.int_] | Tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
        """Compute the indexes of the branches incident to the given node.

        Parameters
        ----------
        node_idx : int or Iterable[int]
            The indexes of the nodes to get the incident branches from.

        return_branch_direction : bool, optional
            If True, also return whether the branch is outgoing from the node. Default is False.

        individual_nodes : bool, optional
            If True, return the incident branches for each node separately. Default is True.

        Returns
        -------
        np.ndarray
            The indexes of the branches incident to the given nodes.

        np.ndarray
            The direction of the branches incident to the given nodes (only returned if ``return_branch_direction`` is True).

            True indicates that the branch is outgoing from the node.

        """  # noqa: E501
        return self._incident_branches(
            node_idx=node_idx, return_branch_direction=return_branch_direction, individual_nodes=False
        )

    @overload
    def incident_branches_individual(
        self, node_idx: npt.ArrayLike[int], return_branch_direction: Literal[False]
    ) -> List[npt.NDArray[np.int_]]: ...
    @overload
    def incident_branches_individual(
        self, node_idx: npt.ArrayLike[int], return_branch_direction: Literal[True]
    ) -> Tuple[List[npt.NDArray[np.int_], List[npt.NDArray[np.bool_]]]]: ...
    def incident_branches_individual(
        self,
        node_idx: npt.ArrayLike[int],
        return_branch_direction: bool = False,
    ) -> List[npt.NDArray[np.int_]] | Tuple[List[npt.NDArray[np.int_], List[npt.NDArray[np.bool_]]]]:
        """Compute the indexes of the branches incident to multiple nodes.
        In contrast to :meth:`VGraph.incident_branches`, this method returns the incident branches for each node separately.

        Parameters
        ----------
        node_idx : int or Iterable[int]
            The indexes of the nodes to get the incident branches from.

        return_branch_direction : bool, optional
            If True, also return whether the branch is outgoing from the node. Default is False.

        individual_nodes : bool, optional
            If True, return the incident branches for each node separately. Default is True.

        Returns
        -------
        List[np.ndarray]
            The indexes of the branches incident to the given nodes.

        List[np.ndarray]
            The direction of the branches incident to the given nodes. Only returned if ``return_branch_direction`` is True. True indicates that the branch is outgoing from the node.


        """  # noqa: E501
        return self._incident_branches(
            node_idx=node_idx, return_branch_direction=return_branch_direction, individual_nodes=True
        )

    def _incident_branches(
        self,
        node_idx: int | npt.ArrayLike[int],
        return_branch_direction: bool = False,
        individual_nodes: bool = True,
    ):
        if np.isscalar(node_idx):
            branch_ids = np.argwhere(np.any(self._branch_list == node_idx, axis=1)).flatten()
            if return_branch_direction:
                return branch_ids, self._branch_list[branch_ids][:, 0] == node_idx
            return branch_ids

        # Multiple nodes
        node_idx = np.asarray(node_idx, dtype=int)
        node_branches = np.argwhere(np.any(self._branch_list[None, :, :] == node_idx[:, None, None], axis=2))

        if not individual_nodes:
            if node_branches.shape[0] == len(node_idx) and np.all(node_branches[:, 0] == np.arange(len(node_idx))):
                branch_ids = node_branches[:, 1]
            else:
                branch_ids = np.unique(node_branches[:, 1])
            if return_branch_direction:
                return branch_ids, np.isin(self._branch_list[branch_ids][:, 0], node_idx[:, None])
            return branch_ids

        node_branches = node_branches[np.argsort(node_branches[:, 0])]
        node_split = np.searchsorted(node_branches[:, 0], np.arange(len(node_idx)))
        branch_ids = np.split(node_branches[:, 1], node_split[1:])

        if return_branch_direction:
            return branch_ids, [self._branch_list[b][:, 0] == n for b, n in zip(branch_ids, node_idx, strict=True)]
        else:
            return branch_ids

    def nodes_degree(
        self, nodes_id: int | npt.ArrayLike | None = None, /, *, count_loop_branches_once: bool = False
    ) -> npt.NDArray[np.int_]:
        """Compute the degree of each node in the graph.

        The degree of a node is the number of branches incident to it.

        Parameters
        ----------
        nodes_id : int or Iterable[int], optional
            The indexes of the nodes to compute the degree from. If None, the degree of all nodes is computed.

        count_loop_branches_only_once : bool, optional
            If True, a branch connecting a node to itself are counted as one incident branch (instead of two). Default is False.

        Returns
        -------
        np.ndarray
            An array of shape (N,) containing the degree of each node.
        """  # noqa: E501
        if np.isscalar(nodes_id):
            return (
                np.sum(np.any(self._branch_list == nodes_id, axis=1))
                if count_loop_branches_once
                else np.sum(self._branch_list == nodes_id)
            )
        elif nodes_id is not None:
            nodes_id = np.asarray(nodes_id, dtype=int)
            if count_loop_branches_once:
                return np.sum(np.any(self._branch_list[None, :, :] == nodes_id[:, None, None], axis=2), axis=1)
            else:
                return np.sum(self._branch_list.flatten()[None, :] == nodes_id[:, None], axis=1)
        else:
            nodes_count = np.bincount(self._branch_list.flatten(), minlength=self.nodes_count)
            if count_loop_branches_once:
                loop_nodes, loops_count = np.unique(
                    self._branch_list[self._branch_list[:, 0] == self._branch_list[:, 1], 0], return_counts=True
                )
                nodes_count[loop_nodes] -= loops_count
            return nodes_count

    def endpoints_nodes(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indexes of the endpoints nodes in the graph.

        The endpoints nodes are the nodes connected to zero or one branch.

        Returns
        -------
        np.ndarray
            The indexes of the endpoint nodes.
        """
        if as_mask:
            return np.bincount(self._branch_list.flatten(), minlength=self.nodes_count) <= 1
        else:
            nodes_id, nodes_count = np.unique(self._branch_list, return_counts=True)
            return nodes_id[nodes_count <= 1]

    def non_endpoints_nodes(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indexes of the non-endpoints nodes in the graph.

        The non-endpoints nodes are the nodes connected to at least two branches.

        Returns
        -------
        np.ndarray
            The indexes of the non-endpoints nodes.
        """
        if as_mask:
            return np.bincount(self._branch_list.flatten(), minlength=self.nodes_count) > 1
        else:
            nodes_id, nodes_count = np.unique(self._branch_list, return_counts=True)
            return nodes_id[nodes_count > 1]

    def endpoints_branches(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indexes of the terminal branches in the graph.

        The terminal branches are the branches connected to a single node.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the terminal branches instead of their indexes.

        Returns
        -------
        np.ndarray
            The indexes of the terminal branches.
        """
        endpoints_nodes = self.endpoints_nodes(as_mask=True)
        endpoints_branches_mask = np.any(endpoints_nodes[self.branch_list], axis=1)
        return endpoints_branches_mask if as_mask else np.argwhere(endpoints_branches_mask).flatten()

        ## Legacy implementation (slower?)
        # _, branch_index, nodes_count = np.unique(self._branch_list, return_counts=True, return_index=True)
        # return np.unique(branch_index[nodes_count == 1] // 2)

    def orphan_branches(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indexes of the branches connected to no other branch in the graph.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the orphan branches instead of their indexes.

        Returns
        -------
        np.ndarray
            The indexes of the branches connected to a single node.
        """
        terminal_nodes = self.endpoints_nodes(as_mask=True)
        orphan_branches_mask = np.all(terminal_nodes[self.branch_list], axis=1) | (
            self.branch_list[:, 0] == self.branch_list[:, 1]
        )
        return orphan_branches_mask if as_mask else np.argwhere(orphan_branches_mask).flatten()

    def self_loop_branches(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indexes of the branches connecting a node to itself in the graph.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the self-loop branches instead of their indexes.

        Returns
        -------
        np.ndarray
            The indexes of the self-loop branches.
        """
        self_loop_mask = self.branch_list[:, 0] == self.branch_list[:, 1]
        return self_loop_mask if as_mask else np.argwhere(self_loop_mask).flatten()

    def nodes_connected_graphs(self) -> Tuple[npt.NDArray[np.int_]]:
        """Compute the connected components of the graph nodes.

        Returns
        -------
        np.ndarray
            An array of shape (N,) containing the index of the connected component of each node.
        """
        return reduce_clusters(self.branch_list)

    ####################################################################################################################
    #  === COMBINE GEOMETRIC DATA ===
    ####################################################################################################################
    def nodes_coord(self) -> npt.NDArray[np.float32]:
        """Compute the coordinates of the nodes in the graph (averaged from all geometric data).

        Returns
        -------
        np.ndarray
            An array of shape (N, 2) containing the (y, x) positions of the nodes in the graph.
        """
        assert len(self._geometric_data) > 0, "No geometric data available to retrieve the nodes coordinates."
        if len(self._geometric_data) == 1:
            # Use the only geometric data
            return self.geometric_data().nodes_coord()

        # Average nodes coordinates from all geometric data
        coord = np.zeros((self.nodes_count, 2), dtype=float)
        count = np.zeros(self.nodes_count, dtype=int)
        for gdata in self._geometric_data:
            nodes_id = gdata.nodes_id
            coord[nodes_id] += gdata.nodes_coord()
            count[nodes_id] += 1
        return coord / count[:, None]

    def branches_node2node_length(self, ids: Optional[int | Iterable[int]] = None) -> npt.NDArray[np.float64]:
        """Compute the length of the branches in the graph (averaged from all geometric data).

        Parameters
        ----------
        ids : int or Iterable[int], optional
            The indexes of the branches to compute the length from. If None, the length of all branches is computed.

        Returns
        -------
        np.ndarray
            An array of shape (B,) containing the length of the branches.
        """
        if ids is None:
            branches = self.branch_list
        else:
            if isinstance(ids, int):
                ids = [ids]
            branches = self.branch_list[ids]

        return np.linalg.norm(self.nodes_coord()[branches[:, 0]] - self.nodes_coord()[branches[:, 1]], axis=1)

    def branches_arc_length(
        self, ids: Optional[int | Iterable[int]] = None, fast_approximation=True
    ) -> npt.NDArray[np.float64]:
        """Compute the arc length of the branches in the graph (averaged from all geometric data).

        Parameters
        ----------
        ids : int or Iterable[int], optional
            The indexes of the branches to compute the arc length from. If None, the arc length of all branches is computed.

        Returns
        -------
        np.ndarray
            An array of shape (B,) containing the arc length of the branches.
        """  # noqa: E501
        assert len(self._geometric_data) > 0, "No geometric data available to compute the arc length."
        if len(self._geometric_data) == 1:
            # Use the only geometric data
            return self.geometric_data().branches_arc_length(ids, fast_approximation=fast_approximation)

        # Average arc length from all geometric data...
        if ids is None:
            # ... for all branches
            total_arc_length = np.zeros(self.branches_count, dtype=float)
            count = np.zeros(self.branches_count, dtype=int)
            for gdata in self._geometric_data:
                id = gdata.branches_id
                total_arc_length[id] += gdata.branches_arc_length(fast_approximation=fast_approximation)
                count[id] += 1
        else:
            if isinstance(ids, int):
                ids = [ids]
            # ... for a subset of branches
            total_arc_length = np.zeros(len(ids), dtype=float)
            count = np.zeros(len(ids), dtype=int)
            for gdata in self._geometric_data:
                id = np.intersect1d(ids, gdata.branches_id)
                total_arc_length[id] += gdata.branches_arc_length(id, fast_approximation=fast_approximation)
                count[id] += 1

        return total_arc_length / count

    def branches_chord_length(self, ids: Optional[int | Iterable[int]] = None) -> npt.NDArray[np.float64]:
        """Compute the chord length of the branches in the graph (averaged from all geometric data).

        Parameters
        ----------
        ids : int or Iterable[int], optional
            The indexes of the branches to compute the chord length from. If None, the chord length of all branches is computed.

        Returns
        -------
        np.ndarray
            An array of shape (B,) containing the chord length of the branches.
        """  # noqa: E501
        assert len(self._geometric_data) > 0, "No geometric data available to compute the chord length."
        if len(self._geometric_data) == 1:
            # Use the only geometric data
            return self.geometric_data().branches_chord_length(ids)

        # Average chord length from all geometric data...
        if ids is None:
            # ... for all branches
            total_chord_length = np.zeros(self.branches_count, dtype=float)
            count = np.zeros(self.branches_count, dtype=int)
            for gdata in self._geometric_data:
                id = gdata.branches_id
                total_chord_length[id] += gdata.branches_chord_length()
                count[id] += 1
        else:
            # ... for a subset of branches
            total_chord_length = np.zeros(len(ids), dtype=float)
            count = np.zeros(len(ids), dtype=int)
            for gdata in self._geometric_data:
                id = np.intersect1d(ids, gdata.branches_id)
                total_chord_length[id] += gdata.branches_chord_length(id)
                count[id] += 1

        return total_chord_length / count

    def branches_label_map(
        self, geometrical_data_priority: Optional[int | Iterable[int]] = None
    ) -> npt.NDArray[np.int_]:
        """Compute the label map of the branches in the graph.

        In case of overlapping branches, the label of the branches is determined by the priority of the geometrical data.

        Parameters
        ----------
        geometrical_data_priority : int or Iterable[int], optional
            The indexes of the geometrical data to use to determine the label of the branches. The first geometrical data in the list has the highest priority.

            If None (by default), the label is determined by the order of the geometrical data in the graph.

        Returns
        -------
        np.ndarray
            An array of shape (H, W) containing the label map of the branches.
        """  # noqa: E501
        geometrical_data_priority = self._geometrical_data_priority(geometrical_data_priority)

        full_domain = Rect.union([gdata.domain for gdata in self._geometric_data])
        label_map = np.zeros(full_domain.size, dtype=int)
        for gdata_id in reversed(geometrical_data_priority):
            gdata = self._geometric_data[gdata_id]
            domain = gdata.domain - full_domain.top_left
            label_map[domain.slice()] = gdata.branches_label_map()
        return label_map

    def branches_geo_data(
        self,
        attr_name: str,
        ids: Optional[int | Iterable[int]] = None,
        geometrical_data_priority: Optional[int | Iterable[int]] = None,
    ) -> list:
        """Compute the geometric data of the branches in the graph.

        Parameters
        ----------
        ids : int or Iterable[int], optional
            The indexes of the branches to compute the geometric data from. If None, the geometric data of all branches is computed.

        geometrical_data_priority : int or Iterable[int], optional
            The indexes of the geometrical data to use to determine the label of the branches. The first geometrical data in the list has the highest priority.

            If None (by default), the label is determined by the order of the geometrical data in the graph.

        Returns
        -------
        list
            A list of VGeometricData objects containing the geometric data of the branches.
        """  # noqa: E501
        geometrical_data_priority = self._geometrical_data_priority(geometrical_data_priority)

        if ids is None:
            ids = range(self.branches_count)
        if isinstance(ids, int):
            ids = [ids]

        geo_data = [None for _ in ids]
        for gdata_id in reversed(geometrical_data_priority):
            gdata: VGeometricData = self._geometric_data[gdata_id]
            branch_data = gdata.branch_data(attr_name, ids)
            for i, data in enumerate(branch_data):
                if data is not None:
                    geo_data[i] = data

        return geo_data

    def _geometrical_data_priority(
        self, geometrical_data_priority: Optional[int | Iterable[int]] = None
    ) -> Iterable[int]:
        if isinstance(geometrical_data_priority, int):
            geometrical_data_priority = [geometrical_data_priority]
        if isinstance(geometrical_data_priority, Iterable):
            geometrical_data_priority = complete_lookup(
                geometrical_data_priority, max_index=len(self._geometric_data) - 1
            )
        elif geometrical_data_priority is None:
            geometrical_data_priority = range(len(self._geometric_data))
        else:
            raise ValueError("geometrical_data_priority must be an integer or an iterable of integers.")
        return geometrical_data_priority

    def transform(self, projection: FundusProjection, inplace=False) -> VGraph:
        """Transform the graph using the given projection.

        Parameters
        ----------
        projection : FundusProjection
            The projection to apply to the graph.

        Returns
        -------
        VGraph
            The transformed graph.
        """
        if not inplace:
            return self.copy().transform(projection, inplace=True)

        for gdata in self._geometric_data:
            gdata.transform(projection, inplace=True)
        return self

    ####################################################################################################################
    #  === GRAPH MANIPULATION ===
    ####################################################################################################################
    def reindex_nodes(self, indexes, inverse_lookup=False) -> VGraph:
        """Reindex the nodes of the graph.

        Parameters
        ----------
        new_node_indexes : npt.NDArray
            The new indexes of the nodes.

        inverse_lookup : bool, optional

            - If False, indexes is sorted by old indexes and contains the new one: indexes[old_index] -> new_index.
            - If True, indexes is sorted by new indexes and contains the old one: indexes[new_index] -> old_index.

            By default: False.

        Returns
        -------
        VGraph
            The modified graph.
        """
        indexes = complete_lookup(indexes, max_index=self.nodes_count - 1)
        if inverse_lookup:
            indexes = invert_complete_lookup(indexes)

        # Update nodes indexes in ...
        self._branch_list = indexes[self._branch_list]  # ... branch list
        self._nodes_attr = self._nodes_attr.set_index(indexes)  # ... nodes attributes
        for gdata in self._geometric_data:  # ... geometric data
            gdata._reindex_nodes(indexes)

        # Update nodes indexes in ...
        indexes = add_empty_to_lookup(indexes, increment_index=False)  # Insert -1 in lookup for missing nodes
        for node_ref in self._nodes_refs:  # ... nodes references
            if node_ref._id > -1:
                node_ref._id = indexes[node_ref._id + 1]
        for branch_ref in self._branches_refs:  # ... nodes indexes stored in branches references
            branch_ref._nodes_id = indexes[branch_ref._nodes_id]
        return self

    def sort_nodes_by_degree(self, descending=True) -> VGraph:
        degree = self.nodes_degree()
        new_order = np.argsort(degree)
        if descending:
            new_order = new_order[::-1]
        return self.reindex_nodes(new_order)

    def reindex_branches(self, indexes, inverse_lookup=False) -> VGraph:
        """Reindex the branches of the graph.

        Parameters
        ----------
        indexes : npt.NDArray
            A lookup table to reindex the branches.

        inverse_lookup : bool, optional

            - If False, indexes is sorted by old indexes and contains the new one: indexes[old_index] -> new_index.
            - If True, indexes is sorted by new indexes and contains the old one: indexes[new_index] -> old_index.

            By default: False.

        Returns
        -------
        VGraph
            The modified graph.
        """
        indexes = complete_lookup(indexes, max_index=self.branches_count - 1)
        if inverse_lookup:
            indexes = invert_complete_lookup(indexes)

        # Update branches indexes in ...
        self._branch_list[indexes, :] = self._branch_list.copy()  # ... branch list
        self._branches_attr = self._branches_attr.set_index(indexes)  # ... branches attributes
        for gdata in self._geometric_data:
            gdata._reindex_branches(indexes)

        # Update branches indexes in ...
        indexes = add_empty_to_lookup(indexes, increment_index=False)  # Insert -1 in lookup for missing branches
        for branch_ref in self._branches_refs:  # ... branches references
            branch_ref._id = indexes[branch_ref._id + 1]
        for node_ref in self._nodes_refs:  # ... incident branches stored in nodes references
            if node_ref._ibranch_ids is not None:
                node_ref._ibranch_ids = indexes[node_ref._ibranch_ids + 1]
        return self

    def flip_branches_direction(self, branches_id: int | Iterable[int]) -> VGraph:
        if isinstance(branches_id, int):
            branches_id = [branches_id]

        # Flip branches in ...
        self._branch_list[branches_id] = self._branch_list[branches_id, ::-1]  # ... branch list
        for gdata in self._geometric_data:  # ... geometric data
            gdata._flip_branches_direction(branches_id)
        for branch_ref in self._branches_refs:  # ... branches references
            if branch_ref._id in branches_id:
                branch_ref._nodes_id = self._branch_list[branch_ref._id]
        for node_ref in self._nodes_refs:  # ... incident branches stored in nodes references
            if node_ref._ibranch_ids is not None and np.any(np.isin(node_ref._ibranch_ids, branches_id)):
                node_ref._ibranch_dirs = node_ref._ibranch_dirs[node_ref._ibranch_ids][:, 0] == node_ref._id
        return self

    def sort_branches_by_nodesID(self, descending=True) -> VGraph:
        flip = self._branch_list[:, 0] > self._branch_list[:, 1]
        self.flip_branches_direction(np.argwhere(flip).flatten())

        nodesID = self._branch_list[:, 0]
        new_order = np.argsort(nodesID)
        if descending:
            new_order = new_order[::-1]
        return self.reindex_branches(new_order)

    # --- Private base edition ---
    def _delete_branches(self, branch_indexes: npt.NDArray[np.int_], update_refs: bool = True) -> npt.NDArray[np.int_]:
        if len(branch_indexes) == 0:
            return np.arange(self.branches_count)

        branches_reindex = create_removal_lookup(branch_indexes, replace_value=-1, length=self.branches_count)
        for gdata in self._geometric_data:
            gdata._reindex_branches(branches_reindex)

        self._branch_list = np.delete(self._branch_list, branch_indexes, axis=0)
        self._branches_attr = self._branches_attr.drop(branch_indexes).reset_index(drop=True)

        if update_refs:
            reindex_refs = add_empty_to_lookup(branches_reindex, increment_index=False)
            for branch in self._branches_refs:
                branch._id = reindex_refs[branch._id + 1]

        return branches_reindex

    def _delete_nodes(self, node_indexes: npt.NDArray[np.int_], update_refs: bool = True) -> npt.NDArray[np.int_]:
        if len(node_indexes) == 0:
            return
        self._nodes_attr = self._nodes_attr.drop(node_indexes).reset_index(drop=True)

        nodes_reindex = create_removal_lookup(node_indexes, replace_value=-1, length=self.nodes_count)
        self._branch_list = nodes_reindex[self._branch_list]
        for gdata in self._geometric_data:
            gdata._reindex_nodes(nodes_reindex)

        self._nodes_count -= len(node_indexes)

        if update_refs:
            nodes_reindex = add_empty_to_lookup(nodes_reindex, increment_index=False)
            for node in self._nodes_refs:
                node._id = nodes_reindex[node._id + 1]
            for branch in self._branches_refs:
                branch._nodes_id = nodes_reindex[branch._nodes_id + 1]

        return nodes_reindex

    # --- Branches edition ---
    def delete_branches(
        self, branch_indexes: int | npt.ArrayLike[int], delete_orphan_nodes=True, *, inplace=True
    ) -> VGraph:
        """Remove the branches with the given indexes from the graph.

        Parameters
        ----------
        branch_indexes : int | npt.ArrayLike[int]
            The indexes of the branches to remove from the graph.

        delete_orphan_nodes : bool, optional
            If True (by default), the nodes that are not connected to any branch after the deletion are removed from the graph.
        inplace : bool, optional
            If True (by default), the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.
        """  # noqa: E501
        graph = self if inplace else self.copy()

        branch_indexes = np.atleast_1d(branch_indexes).astype(int).flatten()

        # Find connected nodes
        connected_nodes = np.array([])
        if delete_orphan_nodes or graph._nodes_refs:
            connected_nodes = np.unique(graph._branch_list[branch_indexes])

            # Clear the incident branch cache of nodes connected to the deleted branches
            for node in graph._nodes_refs:
                if node._id in connected_nodes:
                    node.clear_incident_branch_cache()

        # Remove branches and orphan nodes
        graph._delete_branches(branch_indexes)
        if delete_orphan_nodes:
            orphan_nodes = connected_nodes[np.isin(connected_nodes, graph._branch_list, invert=True)]
            graph._delete_nodes(orphan_nodes)

        return graph

    def split_branch(
        self,
        branchID: int,
        split_coord: Optional[Point | List[Point]] = None,
        *,
        split_curve_id: Optional[int | Iterable[int]] = None,
        return_branch_ids=False,
        inplace=False,
    ) -> VGraph | Tuple[VGraph, npt.NDArray[np.int_]]:
        """Split a branch into two branches by adding a new node near the given coordinates.

        The new node is added to the nodes coordinates and the two new branches are added to the branch list.

        Parameters
        ----------
        branchID : int
            The index of the branch to split.
        split_coord : Point
            The coordinates of the new node to add to the graph.

        Returns
        -------
        VGraph
            The modified graph.
        """
        graph = self if inplace else self.copy()

        # === Check coordinates or index of the splits ===
        if split_coord is not None:
            split_coord = np.asarray(split_coord)
            single_split = split_coord.ndim == 1
            if single_split:
                split_coord = split_coord[None]
            assert split_coord.ndim == 2 and split_coord.shape[1] == 2, "split_coord must be a 2D array of shape (N, 2)"
        elif split_curve_id is not None:
            split_coord = np.asarray(split_curve_id)
            single_split = split_coord.ndim == 0
            if single_split:
                split_coord = split_coord[None]
            assert split_coord.ndim == 1, "split_curve_id must be a 1D array."
        else:
            raise ValueError("Either split_coord or split_curve_id must be provided.")
        assert len(split_coord) > 0, "At least one split coordinate must be provided."

        # === Compute index for the new nodes and branches ===
        _, nEnd = graph._branch_list[branchID]
        new_nodeIds = [graph.nodes_count + i for i in range(len(split_coord))]
        new_branchIds = [graph.branches_count + i for i in range(len(split_coord))]

        # === Insert new nodes and branches ... ===
        # 1. ... in the branch list
        graph._branch_list[branchID, 1] = new_nodeIds[0]
        new_branches = []
        for nPrev, nNext in itertools.pairwise(new_nodeIds + [nEnd]):
            new_branches.append((nPrev, nNext))
        graph._branch_list = np.concatenate((graph._branch_list, new_branches), axis=0)
        # 2. ... in the branches attributes
        new_df = pd.DataFrame(index=pd.RangeIndex(graph.branches_count), columns=graph._branches_attr.columns)
        new_df.loc[: graph.branches_count] = graph._branches_attr
        for i in new_branchIds:
            new_df.loc[i] = graph._branches_attr.loc[branchID]
        graph._branches_attr = new_df
        # 3. ... in the nodes attributes
        graph._nodes_count += len(new_nodeIds)
        new_df = pd.DataFrame(index=pd.RangeIndex(graph._nodes_count), columns=graph._nodes_attr.columns)
        new_df.loc[: graph.nodes_count] = graph._nodes_attr
        graph._nodes_attr = new_df

        # === Split the branches in the geometric data ===
        for gdata in graph._geometric_data:
            gdata._split_branch(branchID, split_coord, new_branchIds, new_nodeIds)

        # === Update the nodes and branches references ===
        for node in graph._nodes_refs:
            if node._id == nEnd:
                node.clear_incident_branch_cache()
        for branch in graph._branches_refs:
            if branch._id == branchID:
                branch._nodes_id = graph._branch_list[branchID]

        return graph if not return_branch_ids else graph, [branchID] + new_branchIds

    def bridge_nodes(self, node_pairs: npt.NDArray, *, fuse_nodes=False, no_check=False, inplace=False) -> VGraph:
        """Fuse pairs of endpoints nodes by linking them together.

        The nodes are removed from the graph and their corresponding branches are merged.

        Parameters
        ----------
        node_pairs : npt.NDArray
            Array of shape (P, 2) containing P pairs of indexes of the nodes to link.

        fuse_nodes : bool, optional
            If True, the nodes are fused together instead of being linked by a new branch.

        no_check : bool, optional
            If True, the node pairs are not checked for validity. By default, the pairs are checked to ensure that the nodes are not already connected by a branch.

        Returns
        -------
        VGraph
            The modified graph.
        """  # noqa: E501
        graph = self.copy() if not inplace else self

        assert node_pairs.ndim == 2 and node_pairs.shape[1] == 2, "node_pairs must be a 2D array of shape (P, 2)."

        if not no_check:
            node_pairs = np.unique(node_pairs, axis=0)
            node_pairs = node_pairs[np.isin(node_pairs, graph._branch_list, assume_unique=True, invert=True)]

        # === Insert new branches between the nodes ===
        graph._branch_list = np.concatenate((graph._branch_list, node_pairs), axis=0)

        # === Fuse the nodes together if needed ===
        if fuse_nodes:
            graph.fuse_nodes(node_pairs)
        else:
            # or update the nodes references
            updated_nodes = np.unique(node_pairs)
            for node_ref in graph._nodes_refs:
                if node_ref._id in updated_nodes:
                    node_ref.clear_incident_branch_cache()
        return graph

    # --- Nodes edition ---
    def delete_nodes(self, node_indexes: npt.NDArray[np.int_], *, inplace=True) -> VGraph:
        """Remove the nodes with the given indexes from the graph as well as their incident branches.

        Parameters
        ----------
        node_indexes : npt.NDArray

        inplace : bool, optional
            If True (by default), the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.
        """
        incident_branches = self.incident_branches(node_indexes)
        return self.delete_branches(incident_branches, delete_orphan_nodes=True, inplace=inplace)

    def fuse_nodes(
        self,
        nodes: npt.NDArray[np.int_],
        *,
        quiet_invalid_node=False,
        inplace=False,
        incident_branches: Optional[List[npt.NDArray[np.int_]]] = None,
    ) -> VGraph:
        """Fuse nodes connected to exactly two branches.

        The nodes are removed from the graph and their corresponding branches are merged.

        Parameters
        ----------
        nodes : npt.NDArray
            Array of indexes of the nodes to fuse.

        quiet_invalid_node : bool, optional
            If True, do not raise an error if a node is not connected to exactly two branches.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

            By default: False.

        incident_branches : npt.NDArray, optional
            The incident branches of the nodes. If None, the incident branches are computed.

        Returns
        -------
        VGraph
            The modified graph.
        """
        graph = self.copy() if not inplace else self
        graph._fuse_nodes(nodes, quiet_invalid_node=quiet_invalid_node, incident_branches=incident_branches)
        return graph

    def _fuse_nodes(
        self, nodes: npt.ArrayLike[int], quiet_invalid_node=False, incident_branches=None
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Fuse nodes connected to exactly two branches. (See :meth:`fuse_nodes` for the public method)
        This method operates inplace and returns a lookup table to reindex the branches and the indexes of the deleted branches.
        """  # noqa: E501
        first_last_merged_branches = []
        reconnected_nodes = []

        # 1. Find branches to delete and group the consecutive branches
        branch_by_fused_nodes = {}
        incident_branches = if_none(incident_branches, self.incident_branches_individual(nodes))
        for nodes_id, b in zip(nodes, incident_branches, strict=True):
            if len(b) == 2:
                branch_by_fused_nodes[nodes_id] = b
            elif not quiet_invalid_node:
                raise ValueError(f"Node {nodes_id} is not connected to exactly two branches.")
        consecutive_branches = reduce_chains(list(branch_by_fused_nodes.values()))

        # 2. For each group of consecutive branches flip them if needed and merge them
        branches_to_delete = []
        flip_main_branch = []
        for branches in consecutive_branches:
            flip_branches = np.zeros(len(branches), dtype=bool)

            # Read the first branch and check if the fused node is the first or the second one
            n0, n1 = self._branch_list[branches[0]]
            if n0 in branch_by_fused_nodes.keys():
                # If n0 is the fused node, flip the branch
                nNext = n0
                n0 = n1
                flip_branches[0] = True
                flip_main_branch.append(True)
            else:
                nNext = n1
                flip_main_branch.append(False)

            # For each branch, check that its consecutive to the previous one and store the next node
            for i, b in enumerate(branches[1:]):
                n1, n2 = self._branch_list[b]
                if n1 == nNext:
                    nNext = n2
                elif n2 == nNext:
                    nNext = n1
                    flip_branches[i + 1] = True
                else:
                    raise ValueError(f"Branches {b} is not consecutive to branch {branches[i]}.")

            # Update the first branch
            self._branch_list[branches[0]] = [n0, nNext]

            # Remember the first and last merged branches and the last node
            first_last_merged_branches.append((branches[0], branches[-1]))
            reconnected_nodes.append(nNext)

            # Merge the branches
            branches_to_delete.extend(branches[1:])
            for gdata in self._geometric_data:
                if flip_branches.any():
                    gdata._flip_branches_direction(np.asarray(branches)[flip_branches])
                gdata._merge_branches(branches)

        # 3a. Update the branch references of the first and last merged branches
        if self._branches_refs:
            first_last_branches = np.unique(first_last_merged_branches)
            last_to_first_branches = {last: first for first, last in first_last_merged_branches}
            for branch in self._branches_refs:
                if branch._id in first_last_branches:
                    # Redirect the last branches to the first ones
                    branch._id = last_to_first_branches.get(branch._id, branch._id)
                    branch._nodes_id = self._branch_list[branch._id]
        # 3b. Update the nodes references of the reconnected nodes
        if self._nodes_refs:
            reconnected_nodes = np.unique(reconnected_nodes)
            for node in self._nodes_refs:
                if node._id in reconnected_nodes:
                    node.clear_incident_branch_cache()

        # 4. Remove the branches and the fused nodes
        del_lookup = self._delete_branches(branches_to_delete)
        fused_nodes = np.array(list(branch_by_fused_nodes.keys()))
        self._delete_nodes(fused_nodes)

        for c in consecutive_branches:
            del_lookup[c[1:]] = del_lookup[c[0]]

        return consecutive_branches, flip_main_branch, del_lookup, np.array(branches_to_delete, dtype=int)

    def merge_nodes(
        self,
        clusters: Iterable[set[int]],
        *,
        nodes_weight: Optional[np.ndarray] = None,
        inplace=True,
        assume_reduced=False,
    ) -> VGraph:
        """Merge a cluster of nodes into a single node.

        The node with the smallest index is kept and the others are removed from the graph. The branches inside the clusters are removed, the branches incident to the cluster are connected to the kept node.

        The position of the kept node is the average of the positions of the merged nodes.

        If the resulting node is not connected to any branch, it is removed from the graph.

        Parameters
        ----------
        clusters : Iterable[int]
            The indexes of the nodes to merge.

        nodes_weight : np.ndarray, optional
            The weight of each node. If provided, the nodes position are weighted by this array to compute the new position of the kept node.

        inplace : bool, optional
            If True (by default), the graph is modified in place. Otherwise, a new graph is returned.

        assume_reduced : bool, optional
            If True, the clusters are assumed to be reduced (i.e. each node appears in only one cluster).

        Returns
        -------
        VGraph
            The modified graph.
        """  # noqa: E501
        if not inplace:
            return self.copy().merge_nodes(
                clusters, inplace=True, assume_reduced=assume_reduced, nodes_weight=nodes_weight
            )

        branches_to_remove = []
        updated_branches = []
        nodes_to_remove = []
        resulting_nodes = []
        nodes_lookup = np.arange(self.nodes_count, dtype=int)

        if not assume_reduced:
            clusters = reduce_clusters(clusters)

        if len(clusters) == 0:
            return self

        for cluster in clusters:
            nodes = np.asarray(cluster, dtype=int)
            nodes.sort()
            n0 = nodes[0]
            resulting_nodes.append(n0)

            # 1. Handle branches connections
            branches_connections = np.isin(self._branch_list, nodes)
            branches_connections_count = np.sum(branches_connections, axis=1)
            #   - Mark every branches inside the cluster (with 2/2 connections in cluster) to be removed
            branches_to_remove.extend(np.argwhere(branches_connections_count == 2).flatten())
            #   - Connect every branches incident to the cluster to the kept node
            branches_to_connect = np.argwhere(branches_connections_count == 1).flatten()
            if len(branches_to_connect) == 0:
                nodes_to_remove.append(nodes)
                continue
            else:
                self._branch_list[branches_connections] = n0
                updated_branches.append(branches_to_connect)

            # 2. Handle Geometric data
            weight = None
            if nodes_weight is not None:
                weight = nodes_weight[nodes]
            for gdata in self._geometric_data:
                gdata._merge_nodes(nodes, weight=weight)

            # 3. Mark nodes to be removed
            nodes_to_remove.append(nodes[1:])
            nodes_lookup[nodes[1:]] = n0

        # 4. Update nodes references
        if self._nodes_refs:
            resulting_nodes = np.unique(resulting_nodes)
            nodes_lookup = add_empty_to_lookup(nodes_lookup, increment_index=False)
            for node in self._nodes_refs:
                node._id = nodes_lookup[node._id + 1]
                if node._id in resulting_nodes:
                    node.clear_incident_branch_cache()

        # 5. Remove branches and nodes
        if len(branches_to_remove) > 0:
            branches_to_remove = np.unique(np.asarray(branches_to_remove, dtype=int))
            self._delete_branches(branches_to_remove)

        if len(nodes_to_remove) > 0:
            nodes_to_remove = np.unique(np.concatenate(nodes_to_remove, dtype=int))
            self._delete_nodes(nodes_to_remove, update_refs=False)

        # 6. Update branches references
        if self._branches_refs and updated_branches:
            updated_branches = np.unique(np.concatenate(updated_branches, dtype=int))
            for branch in self._branches_refs:
                if branch._id in updated_branches:
                    branch._nodes_id = self.branch_list[branch._id]

        return self

    ####################################################################################################################
    #  === BRANCH AND NODE ACCESSORS ===
    ####################################################################################################################
    NODE_ACCESSOR_TYPE: Type[VGraphNode] = VGraphNode
    BRANCH_ACCESSOR_TYPE: Type[VGraphBranch] = VGraphBranch

    def node(self, node_id: int, /) -> VGraphNode:
        """Create a reference to a given node of the graph.

        Parameters
        ----------
        node_id : int
            The index of the node.

        Returns
        -------
        VGraphNode
            A reference to the node.
        """
        assert 0 <= node_id < self.nodes_count, f"Node index {node_id} is out of range."
        return self.__class__.NODE_ACCESSOR_TYPE(self, node_id)

    def nodes(
        self,
        ids: Optional[int | npt.ArrayLike[int]] = None,
        /,
        *,
        only_degree: Optional[int | npt.ArrayLike[np.int_]] = None,
        dynamic_iterator: bool = False,
    ) -> Generator[VGraphNode]:
        """Iterate over the nodes of a graph, encapsulated in :class:`VGraphNode` objects.

        Parameters
        ----------
        ids : int or npt.ArrayLike[int], optional
            The indexes of the nodes to iterate over. If None, iterate over all nodes.

        only_degree : int or npt.ArrayLike[int], optional
            If not None, iterate only over the nodes with the given degree.

        dynamic_iterator : bool, optional
            If True, iterate over all the nodes present in the graph when this method is called.
            All nodes added during the iteration will be ignored. Nodes reindexed during the iteration will be visited in their original order at the time of the call. Deleted nodes will not be visited.

            Enable this option if you plan to modify the graph during the iteration. If you only plan to read the graph, disable this option for better performance.

        Returns
        -------
        Generator[VGraphNode]
            A generator that yields nodes.
        """  # noqa: E501
        nodes_ids = range(self.nodes_count) if ids is None else np.atleast_1d(ids).astype(int).flatten()
        if only_degree is not None:
            only_degree = np.asarray(only_degree, dtype=int)
            node_degree = self.nodes_degree()
            nodes_ids = np.argwhere(np.isin(node_degree, only_degree)).flatten()
        assert isinstance(nodes_ids, range) or (
            np.all(0 <= nodes_ids) and np.all(nodes_ids < self.nodes_count)
        ), "Node index is out of range."

        if dynamic_iterator:
            nodes = [self.__class__.NODE_ACCESSOR_TYPE(self, i) for i in nodes_ids]
            while len(nodes) > 0:
                node = nodes.pop(0)
                if node.is_valid():
                    yield node
        else:
            for i in nodes_ids:
                yield self.__class__.NODE_ACCESSOR_TYPE(self, i)

    def branch(self, branch_id: int, /) -> VGraphBranch:
        """Create a reference to a given branch of the graph.

        Parameters
        ----------
        branch_id : int
            The index of the branch.

        Returns
        -------
        VGraphBranch
            A reference to the branch.
        """
        assert 0 <= branch_id < self.branches_count, f"Branch index {branch_id} is out of range."
        return self.__class__.BRANCH_ACCESSOR_TYPE(self, branch_id)

    def branches(
        self, ids: Optional[int | npt.ArrayLike[int]] = None, /, *, only_terminal=False, dynamic_iterator: bool = False
    ) -> Generator[VGraphBranch]:
        """Iterate over the branches of a graph, encapsulated in :class:`VGraphBranch` objects.

        Parameters
        ----------
        ids : int or npt.ArrayLike[int], optional
            The indexes of the branches to iterate over. If None, iterate over all branches.

        only_terminal : bool, optional
            If True, iterate only over the terminal branches.

        dynamic_iterator : bool, optional
            If True, iterate over all the branches present in the graph when this method is called.
            All branches added during the iteration will be ignored. Branches reindexed during the iteration will be visited in their original order at the time of the call. Deleted branches will not be visited.

            Enable this option if you plan to modify the graph during the iteration. If you only plan to read the graph, disable this option for better performance.

        Returns
        -------
        Generator[VVGraphBranch]
            A generator that yields branches.
        """  # noqa: E501
        branches_ids = range(self.branches_count) if ids is None else np.atleast_1d(ids).astype(int).flatten()
        if only_terminal:
            branches_ids = np.intersect1d(branches_ids, self.endpoints_branches())
        assert isinstance(branches_ids, range) or (
            np.all(0 <= branches_ids) and np.all(branches_ids < self.branches_count)
        ), "Branch index is out of range."

        if dynamic_iterator:
            branches = [self.__class__.BRANCH_ACCESSOR_TYPE(self, i) for i in branches_ids]
            while len(branches) > 0:
                branch = branches.pop(0)
                if branch.is_valid():
                    yield branch
        else:
            for i in branches_ids:
                yield self.__class__.BRANCH_ACCESSOR_TYPE(self, i)

    ####################################################################################################################
    #  === VISUALISATION UTILITIES ===
    ####################################################################################################################
    def jppype_layer(
        self,
        edge_labels=False,
        node_labels=False,
        edge_map=False,
        bspline=None,
        max_colored_node_id: Optional[int] = None,
        boundaries=None,
    ):
        from jppype.layers import LayerGraph
        from jppype.utils.color import colormap_by_name

        from ..utils.bezier import BezierCubic
        from .vgeometric_data import VBranchGeoData

        if bspline is False:
            bspline = None
        elif bspline is not None:
            edge_map = False
        if boundaries is False:
            boundaries = None

        domain = self.geometric_data().domain

        layer = LayerGraph(
            self.branch_list,
            self.nodes_coord() - np.array(domain.top_left)[None, :],
            self.geometric_data().branches_label_map(calibre_attr=boundaries),
        )
        layer.set_options(
            {
                "edges_labels_visible": bool(edge_labels),
                "nodes_labels_visible": bool(node_labels),
                "edges_labels": None if isinstance(edge_labels, bool) else edge_labels,
                "nodes_labels": None if isinstance(node_labels, bool) else node_labels,
                "edge_map_visible": edge_map,
            }
        )
        if bspline is not None:
            if bspline is True:
                bspline = VBranchGeoData.Fields.BSPLINE
            branches_geodata = self.branches_geo_data(bspline)
            nodes_coord = self.nodes_coord()

            bsplines_path = []
            filler_paths = []
            for i, d in enumerate(branches_geodata):
                n1, n2 = [Point(*nodes_coord[_]) for _ in self.branch_list[i]]
                if isinstance(d, VBranchGeoData.BSpline):
                    bsplines_path.append(d.data.to_path())
                    filler_paths.append([_.to_path() for _ in d.data.filling_curves(n1, n2, smoothing=0.5)])
                else:
                    bsplines_path.append("")
                    filler_paths.append([BezierCubic(n1, n1, n2, n2).to_path()])

            layer.edges_path = bsplines_path
            layer.dotted_edges_paths = filler_paths

        if bspline or edge_map:
            layer.edges_labels_coord = self.geometric_data().branch_midpoint()

        if max_colored_node_id:
            node_cmap = {None: colormap_by_name()} | {
                _: "#444" for _ in range(max_colored_node_id + 1, self.nodes_count + 1)
            }
            layer.nodes_cmap = node_cmap
            incidents_branches = np.setdiff1d(
                np.arange(self.branches_count),
                self.incident_branches(np.arange(max_colored_node_id + 1)),
            )
            branch_cmap = {None: colormap_by_name()} | {int(_): "#444" for _ in incidents_branches}
            layer.edges_cmap = branch_cmap

        return layer

    def branch_normals_map(self, segmentation, only_tips=False):
        import cv2

        from ..utils.graph.measures import branch_boundaries

        def eval_point(curve):
            return [0, len(curve) - 1] if only_tips else np.arange(len(curve), step=4)

        out_map = np.zeros_like(segmentation, dtype=np.uint8)

        yx_curves = {k: curve for k, curve in enumerate(self.geometric_data().branch_curve()) if len(curve) > 1}
        boundaries = {id: branch_boundaries(curve, segmentation, eval_point(curve)) for id, curve in yx_curves.items()}

        for i, curve in yx_curves.items():
            bounds = boundaries[i]
            curve = curve[eval_point(curve)]
            for (y, x), ((byL, bxL), (byR, bxR)) in zip(curve, bounds, strict=True):
                dL = np.linalg.norm([x - bxL, y - byL])
                dR = np.linalg.norm([x - bxR, y - byR])
                if not only_tips and abs(dL - dR) > 1.5:
                    continue
                if byL != y or bxL != x:
                    cv2.line(out_map, (x, y), (bxL, byL), int(i + 1), 1)
                if byR != y or bxR != x:
                    cv2.line(out_map, (x, y), (bxR, byR), int(i + 1), 1)
                out_map[y, x] = 0

        return out_map
