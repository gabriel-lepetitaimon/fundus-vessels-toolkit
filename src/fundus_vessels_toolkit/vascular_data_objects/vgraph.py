from __future__ import annotations

__all__ = ["VGraph"]

import itertools
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Literal, Optional, Tuple, Type, TypeAlias, Union, overload
from weakref import WeakSet

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.bezier import BSpline
from ..utils.cluster import reduce_chains, reduce_clusters
from ..utils.cpp_optimized import first_index_of, first_two_index_of
from ..utils.data_io import NumpyDict, load_numpy_dict, pandas_to_numpy_dict, save_numpy_dict
from ..utils.fundus_projections import FundusProjection
from ..utils.geometric import Point, Rect
from ..utils.lookup_array import add_empty_to_lookup, complete_lookup, create_removal_lookup, invert_complete_lookup
from ..utils.pandas import DFSetterAccessor
from .vgeometric_data import VBranchGeoData, VBranchGeoDataKey, VGeometricData

IndexLike: TypeAlias = Union[int, npt.ArrayLike, pd.Series, None]


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
        graph._node_refs.add(self)

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
        return DFSetterAccessor(self.__graph._node_attr, self._id)

    def coord(self, geodata: VGeometricData | int = 0) -> Point:
        if not self.is_valid():
            raise RuntimeError("The node has been removed from the graph.")
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(geodata)
        return Point(*geodata.node_coord(self._id))

    #  __ INCIDENT BRANCHES __
    def _update_adjacent_branch_cache(self):
        self._ibranch_ids, self._ibranch_dirs = self.graph.adjacent_branches(self._id, return_branch_direction=True)

    def clear_adjacent_branch_cache(self):
        self._ibranch_ids = None
        self._ibranch_dirs = None

    @property
    def adjacent_branch_ids(self) -> List[int]:
        if not self.is_valid():
            return []
        if self._ibranch_ids is None:
            self._update_adjacent_branch_cache()
        return [int(_) for _ in self._ibranch_ids]

    @property
    def adjacent_branches_first_node(self) -> List[bool]:
        if not self.is_valid():
            return []
        if self._ibranch_ids is None:
            self._update_adjacent_branch_cache()
        return [bool(_) for _ in self._ibranch_dirs]

    def adjacent_branches(self) -> Iterable[VGraphBranch]:
        """Return the branches incident to this node.

        ..warning::
            The returned :class:`VGraphBranch` objects are linked to their branch and not to this node. After merging or splitting this node or its branches, the original returned branches may not be incident to this node anymore.
        """  # noqa: E501
        if not self.is_valid():
            return ()
        if self._ibranch_ids is None:
            self._update_adjacent_branch_cache()
        return (VGraphBranch(self.__graph, i) for i in self._ibranch_ids)

    @property
    def degree(self) -> int:
        if not self.is_valid():
            return 0
        if self._ibranch_ids is None:
            self._update_adjacent_branch_cache()
        return len(self._ibranch_ids)

    def adjacent_nodes(self) -> Iterable[VGraphNode]:
        """Return the nodes connected to this node by an incident branch.

        ..warning::
            The returned :class:`VGraphNode` objects are linked to their node and not to this node adjacency. After merging or splitting this node or its incident branches, the original returned nodes may not be connected to this node anymore.
        """  # noqa: E501
        if not self.is_valid():
            return ()
        if self._ibranch_ids is None:
            self._update_adjacent_branch_cache()
        return (
            VGraphNode(self.__graph, self.__graph._branch_list[branch][1 if dir else 0])
            for branch, dir in zip(self._ibranch_ids, self._ibranch_dirs, strict=True)
        )

    @overload
    def tips_geodata(self, attrs: VBranchGeoDataKey, geodata: VGeometricData | int = 0) -> np.ndarray: ...
    @overload
    def tips_geodata(
        self, attrs: Optional[List[VBranchGeoDataKey]], geodata: VGeometricData | int = 0
    ) -> Dict[str, np.ndarray]: ...
    def tips_geodata(
        self,
        attrs: Optional[VBranchGeoDataKey | List[VBranchGeoDataKey]] = None,
        geodata: VGeometricData | int = 0,
    ) -> np.ndarray | Dict[str, np.ndarray]:
        if not self.is_valid():
            raise RuntimeError("The node has been removed from the graph.")
        if self._ibranch_ids is None:
            self._update_adjacent_branch_cache()
        if not isinstance(geodata, VGeometricData):
            geodata = self.__graph.geometric_data(geodata)
        return geodata.tip_data(attrs, self._ibranch_ids, self._ibranch_dirs)

    def tips_tangent(self, geodata: VGeometricData | int = 0, infer_from_nodes_if_missing=True) -> np.ndarray:
        if not self.is_valid():
            raise RuntimeError("The node has been removed from the graph.")
        if self._ibranch_ids is None:
            self._update_adjacent_branch_cache()
        if not isinstance(geodata, VGeometricData):
            geodata = self.__graph.geometric_data(geodata)
        return geodata.tip_tangent(
            self._ibranch_ids,
            self._ibranch_dirs,
            attr=VBranchGeoData.Fields.TIPS_TANGENT,
            infer_from_nodes_if_missing=infer_from_nodes_if_missing,
        )


class VGraphBranch:
    """A class representing a branch in a vascular network graph.
    :class:`VGraphBranch` objects are only references to graph's branches and do not store any data.
    All their method and properties makes calls to :class:`VGraph` methods to generate their results.
    """

    def __init__(self, graph: VGraph, id: int, nodes_id: Optional[npt.NDArray[np.int_]] = None):
        self.__graph = graph
        self._id = id
        graph._branch_refs.add(self)

        self._nodes_id = graph._branch_list[id] if nodes_id is None else nodes_id

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
        return 0 <= self._id < self.__graph.branch_count

    @property
    def id(self) -> int:
        if self._id < 0:
            raise RuntimeError("The branch has been removed from the graph.")
        return int(self._id)

    @property
    def graph(self) -> VGraph:
        return self.__graph

    @property
    def node_ids(self) -> List[int]:
        """The indices of the nodes connected by the branch as a tuple."""
        return [int(_) for _ in self._nodes_id]

    def nodes(self) -> Tuple[VGraphNode, VGraphNode]:
        """The two nodes connected by this branch.  # noqa: E501

        ..warning::
            The returned :class:`VGraphNode` objects are linked to their node and not to this branch tip. After merging, splitting or flipping this branch, the original returned nodes may not be connected by this branch anymore.
        """  # noqa: E501
        assert self.is_valid(), "The branch has been removed from the graph."
        n1, n2 = self._nodes_id
        return VGraphNode(self.__graph, n1), VGraphNode(self.__graph, n2)

    def adjacent_branch_ids(self) -> List[int]:
        if not self.is_valid():
            return []
        n1, n2 = self.node_ids

        adj_branches = np.argwhere(
            np.any(self.__graph._branch_list == n1, axis=1) | np.any(self.__graph._branch_list == n2, axis=1)
        ).flatten()
        return [int(_) for _ in adj_branches if _ != self._id]

    def adjacent_branches(self) -> Iterable[VGraphBranch]:
        """Return the branches adjacent to this branch.

        ..warning::
            The returned :class:`VGraphBranch` objects are linked to their branch and not to this branch. After merging, splitting or flipping this branch, the original returned branches may not be adjacent to this branch anymore.
        """  # noqa: E501
        return (VGraphBranch(self.__graph, i) for i in self.adjacent_branch_ids())

    @property
    def attr(self) -> DFSetterAccessor:
        assert self.is_valid(), "The branch has been removed from the graph."
        return DFSetterAccessor(self.__graph._branch_attr, self._id)

    def curve(self, geodata: VGeometricData | int = 0) -> npt.NDArray[np.int_]:
        """Return the indices of the pixels forming the curve of the branch as a 2D array of shape (n, 2).

        This method is a shortcut to :meth:`VGeometricData.branch_curve`.

        Parameters
        ----------
        geo_data : int, optional
            The index of the geometrical_data, by default 0

        Returns
        -------
        np.ndarray
            The indices of the pixels forming the curve of the branch.
        """
        assert self.is_valid(), "The branch has been removed from the graph."
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(geodata)
        return geodata.branch_curve(self._id)

    def midpoint(self, geodata: VGeometricData | int = 0) -> Point:
        """Return the middle point of the branch.

        This method is a shortcut to :meth:`VGeometricData.branch_midpoint`.

        Parameters
        ----------
        geo_data : int, optional
            The index of the geometrical_data, by default 0

        Returns
        -------
        Point
            The middle point of the branch.
        """
        assert self.is_valid(), "The branch has been removed from the graph."
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(geodata)
        return Point(*geodata.branch_midpoint(self._id))

    def bspline(self, geodata: VGeometricData | int = 0) -> BSpline:
        assert self.is_valid(), "The branch has been removed from the graph."
        if not isinstance(geodata, VGeometricData):
            geodata = self.__graph.geometric_data(geodata)
        return geodata.branch_bspline(self._id)

    def geodata(
        self, attr_name: VBranchGeoDataKey, geodata: VGeometricData | int = 0
    ) -> VBranchGeoData.Base | Dict[str, VBranchGeoData.Base]:
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
        if not isinstance(geodata, VGeometricData):
            geodata = self.__graph.geometric_data(geodata)
        return geodata.branch_data(attr_name, self._id)

    def node_to_node_length(self, geodata: VGeometricData | int = 0) -> float:
        assert self.is_valid(), "The branch has been removed from the graph."
        if not isinstance(geodata, VGeometricData):
            geodata = self.graph.geometric_data(geodata)
        yx1, yx2 = geodata.node_coord(self._nodes_id)
        return np.linalg.norm(yx1 - yx2)

    def arc_length(self, geodata: VGeometricData | int = 0) -> float:
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
        branch_list: np.ndarray | str,
        geometric_data: VGeometricData | Iterable[VGeometricData] = (),
        node_attr: Optional[pd.DataFrame] = None,
        branch_attr: Optional[pd.DataFrame] = None,
        node_count: Optional[int] = None,
        check_integrity: bool = True,
    ):
        """Create a Graph object from the given data.

        Parameters
        ----------
        branch_list :
            A 2D array of shape (B, 2) where B is the number of branches. Each row contains the indices of the nodes
            connected by each branch.

        geometric_data : VGeometricData or Iterable[VGeometricData]
            The geometric data associated with the graph.

        node_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each node. The index of the dataframe must be the nodes indices.

        branch_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each branch. The index of the dataframe must be the branches indices.

        node_count : int, optional
            The number of nodes in the graph. If not provided, the number of nodes is inferred from the branch list and the integrity of the data is checked with :meth:`VGraph.check_integrity`.


        Raises
        ------
        ValueError
            If the input data does not match the expected shapes.
        """  # noqa: E501
        # === Check and store branches list ===
        branch_list = np.asarray(branch_list)
        assert (
            branch_list.ndim == 2 and branch_list.shape[1] == 2
        ), "branch_list must be a 2D array of shape (B, 2) where B is the number of branches"
        self._branch_list = branch_list
        B = branch_list.shape[0]

        # === Infer node_count ===
        if node_count is None:
            nodes_indexes = np.unique(branch_list)
            assert len(nodes_indexes) == nodes_indexes[-1] + 1, (
                f"The branches list must contain every node id at least once (from 0 to {len(nodes_indexes)-1}).\n"
                f"\t Nodes {sorted([int(_) for _ in np.setdiff1d(np.arange(len(nodes_indexes)), nodes_indexes)]) }"
                " are missing."
            )
            N = len(nodes_indexes)
        else:
            N = node_count
        self._nodes_count = N

        if node_attr is None:
            node_attr = pd.DataFrame(index=pd.RangeIndex(N))
        self._node_attr = node_attr
        if branch_attr is None:
            branch_attr = pd.DataFrame(index=pd.RangeIndex(B))
        self._branch_attr = branch_attr

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

        if check_integrity or node_count is None:
            self.check_integrity()

        self._node_refs: WeakSet[VGraphNode] = WeakSet()
        self._branch_refs: WeakSet[VGraphBranch] = WeakSet()

    def __getstate__(self) -> object:
        d = self.__dict__.copy()
        d.drop("_node_refs", None)
        d.drop("_branch_refs", None)
        return d

    def check_integrity(self):
        """Check the integrity of the graph data.

        This method checks that all branches and nodes index are consistent.

        Raises
        ------
        ValueError
            If the graph data is not consistent.
        """
        N, B = self.node_count, self.branch_count

        assert N > 0, "The graph must contain at least one node."
        assert B > 0, "The graph must contain at least one branch."

        # --- Check geometric data ---
        branches_idx = set()
        nodes_idx = set()
        for gdata in self._geometric_data:
            # Check that each node has a distinct position
            if (np.diff(gdata._nodes_coord[np.lexsort(gdata._nodes_coord.T)], axis=0) == 0).all(axis=1).any():
                warnings.warn("The geometric data contains duplicated nodes coordinates.", stacklevel=2)

            branches_idx.update(gdata.branch_ids)
            nodes_idx.update(gdata.node_ids)

        branches_idx.difference_update(np.arange(B))
        assert (
            len(branches_idx) == 0
        ), f"Geometric data contains branches indices that are not in the branch list: {branches_idx}."

        nodes_idx.difference_update(np.arange(N))
        assert (
            len(nodes_idx) == 0
        ), f"Geometric data contains nodes indices that are above the nodes count: {nodes_idx}."

        # --- Check nodes attributes ---
        assert (
            self._node_attr.index.inferred_type == "integer"
        ), "The index of nodes_attr dataframe must be nodes Index."
        assert self._node_attr.index.max() < N and self._node_attr.index.min() >= 0, (
            "The maximum value in nodes_attr index must be lower than the number of nodes."
            f" Got {self._node_attr.index.max()} instead of {N}"
        )
        if len(self._node_attr.index) != N:
            self._node_attr.reindex(np.arange(N))

        # --- Check branches attributes ---
        assert (
            self._branch_attr.index.inferred_type == "integer"
        ), "The index of branches_attr dataframe must be branches Index."
        assert self._branch_attr.index.max() < B and self._branch_attr.index.min() >= 0, (
            "The maximum value in branches_attr index must be lower than the number of branches."
            f" Got {self._branch_attr.index.max()} instead of {B}"
        )
        if len(self._branch_attr.index) != B:
            self._branch_attr.reindex(np.arange(B))

    def copy(self) -> VGraph:
        """Create a copy of the current Graph object."""
        return VGraph(
            self._branch_list.copy(),
            [gdata.copy(None) for gdata in self._geometric_data],
            self._node_attr.copy() if self._node_attr is not None else None,
            self._branch_attr.copy() if self._branch_attr is not None else None,
            self._nodes_count,
            check_integrity=False,
        )

    def save(self, filename: Optional[str | Path] = None) -> NumpyDict:
        """Save the graph data to a file.

        The graph is saved as a dictionary with the following keys:
            - ``branch_list``: The list of branches in the graph as a 2D array of shape (B, 2) where B is the number of branches. Each row contains the indices of the nodes connected by each branch.
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
            nodes_attr=pandas_to_numpy_dict(self._node_attr),
            branches_attr=pandas_to_numpy_dict(self._branch_attr),
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

    @classmethod
    def empty(cls) -> VGraph:
        """Create an empty graph."""
        return cls(np.empty((0, 2), dtype=int), [VGeometricData.empty()], node_count=0, check_integrity=False)

    @classmethod
    def empty_like(cls, other: VGraph) -> VGraph:
        """Create an empty graph with the same attributes as another graph."""
        assert isinstance(other, VGraph), "The other object must be a VGraph object."
        return cls(**cls._empty_like_kwargs(other))

    @classmethod
    def _empty_like_kwargs(cls, other: VGraph) -> Dict[str, Any]:
        branch_attr = pd.DataFrame(columns=other._branch_attr.columns)
        node_attr = pd.DataFrame(columns=other._node_attr.columns)

        return dict(
            branch_list=np.empty((0, 2), dtype=int),
            geometric_data=[VGeometricData.empty_like(_, parent_graph=None) for _ in other._geometric_data],
            node_attr=node_attr,
            branch_attr=branch_attr,
            node_count=0,
            check_integrity=False,
        )

    @classmethod
    def parse(cls, branch_list: str) -> VGraph:
        """Parse a branch list from a string.

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
        VGraph
            The graph object created from the branch list.

        Examples
        --------
        >>> VGraph.parse("0->1;3➔2➔1").branch_list.tolist()
        [[0, 1], [3, 2], [2, 1]]

        Raises
        ------
        ValueError
            If the branch list is not correctly formatted.
        """  # noqa: E501
        branches = []
        branch_list = re.sub(r"\s+", "", branch_list)

        # TODO: Add support for node labelling (e.g. "A➔B➔C")

        for branch in re.split(r";", branch_list):
            nodes = re.split(r"->|➔", branch)
            for n1, n2 in itertools.pairwise(nodes):
                try:
                    branches.append([int(n1), int(n2)])
                except ValueError:
                    raise ValueError(
                        f"Invalid branch definition: {branch}: {n1} or {n2} is not a valid node index."
                    ) from None
        return cls(branches)

    ####################################################################################################################
    #  === PROPERTIES ===
    ####################################################################################################################
    @property
    def node_count(self) -> int:
        """The number of nodes in the graph.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2➔3 ; 1➔4 ; 2➔5").node_count
        6
        """
        return self._nodes_count

    @property
    def branch_count(self) -> int:
        """The number of branches in the graph.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2➔3 ; 1➔4 ; 2➔5").branch_count
        5
        """
        return self._branch_list.shape[0]

    @property
    def branch_list(self) -> npt.NDArray[np.int_]:
        """The list of branches in the graph as a 2D array of shape (B, 2) where B is the number of branches. Each row contains the indices of the nodes connected by each branch.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2➔0").branch_list
        array([[0, 1],
               [1, 2],
               [2, 0]])
        """  # noqa: E501
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

        See Also
        --------
        VGeometricData : The class representing the geometric data of a vascular network.
        """
        return self._geometric_data[id]

    @property
    def node_attr(self) -> pd.DataFrame:
        """The attributes of the nodes in the graph as a pandas.DataFrame."""
        return self._node_attr

    @property
    def branch_attr(self) -> pd.DataFrame:
        """The attributes of the branches in the graph as a pandas.DataFrame."""
        return self._branch_attr

    def as_node_ids(self, ids: IndexLike | VGraphNode, /, *, check=True) -> npt.NDArray[np.int_]:
        """Parse objects assimilable to indices into valid node indices.

        Parameters
        ----------
        ids : int | VGraphNode | npt.ArrayLike | pd.Series
            An object assimilable to node indices.

            Valid indices are: integers, iterables of integers, :class:`VGraphNode` objects, or boolean masks.

        Returns
        -------
            The corresponding indices of the nodes.

        Raises
        ------
        ValueError
            If the input object is not assimilable to node indices.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4 ; 2➔5")

        >>> graph.as_node_ids([2, 3])
        array([2, 3])

        >>> graph.as_node_ids([True, False, False, True, False, False])
        array([0, 3])

        >>> graph.as_node_ids(6)
        Traceback (most recent call last):
        ValueError: Invalid node index: out of range

        """  # noqa: E501
        if ids is None:
            return np.arange(self.node_count, dtype=int)
        elif isinstance(ids, int):
            if check and not (0 <= ids < self.node_count):
                raise ValueError("Invalid node index: out of range")
            return np.array([ids], dtype=int)
        elif isinstance(ids, self.__class__.NODE_ACCESSOR_TYPE):
            if check and not ids.is_valid():
                raise ValueError("Invalid node index: the VGraphNode references a deleted node.")
            return np.array([ids._id], dtype=int)
        elif isinstance(ids, pd.Series):
            if ids.dtype != bool:
                raise ValueError("Invalid node index: pd.Series must be a boolean mask.")
            if check and ids.index is not self._node_attr.index and np.any(ids.index != self._node_attr.index):
                raise ValueError("Invalid node index: pd.Series must have the same index as the nodes attributes.")
            return self._node_attr.index[ids].to_numpy()

        ids_array = np.atleast_1d(ids).flatten()
        if ids_array.dtype == bool:
            if check and ids_array.shape != (self.node_count,):
                raise ValueError(
                    "Invalid node index: "
                    "Boolean mask must have the same length as the number of nodes: "
                    f"{self.node_count}, instead of {len(ids_array)}."
                )
            return np.argwhere(ids_array).flatten()
        else:
            ids_array = ids_array.astype(int)
            if check and not np.all(np.logical_and(0 <= ids_array, ids_array < self.node_count)):
                raise ValueError("Invalid node index: out of range.")
            return ids_array

    def as_branch_ids(self, ids: IndexLike | VGraphBranch, /, *, check=True) -> npt.NDArray[np.int_]:
        """Parse objects assimilable to indices into valid branch indices.

        Parameters
        ----------
        ids : int | VGraphBranch | npt.ArrayLike | pd.Series
            An object assimilable to branch indices. Valid indices are: integers, iterables of integers, :class:`VGraphBranch` objects, or boolean masks.

        Returns
        -------
            The corresponding indices of the branches.

        Raises
        ------
        ValueError
            If the input object is not assimilable to branch indices.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4 ; 2➔5")

        >>> graph.as_branch_ids([1, 2])
        array([1, 2])

        >>> graph.as_branch_ids([True, False, False, True, False])
        array([0, 3])

        >>> graph.as_branch_ids(6)
        Traceback (most recent call last):
        ValueError: Invalid branch index: out of range

        """  # noqa: E501
        if ids is None:
            return np.arange(self.branch_count, dtype=int)
        elif isinstance(ids, int):
            if check and not (0 <= ids < self.branch_count):
                raise ValueError("Invalid branch index: out of range")
            return np.array([ids], dtype=int)
        elif isinstance(ids, self.__class__.NODE_ACCESSOR_TYPE):
            if check and not ids.is_valid():
                raise ValueError("Invalid branch index: the VGraphBranch references a deleted branch.")
            return np.array([ids._id], dtype=int)
        elif isinstance(ids, pd.Series):
            if ids.dtype != bool:
                raise ValueError("Invalid branch index: pd.Series must be a boolean mask.")
            if check and ids.index is not self._branch_attr.index and np.any(ids.index != self._branch_attr.index):
                raise ValueError("Invalid branch index: pd.Series must have the same index as the branches attributes.")
            return self._branch_attr.index[ids].to_numpy()

        ids_array = np.atleast_1d(ids).flatten()
        if ids_array.dtype == bool:
            if check and ids_array.shape != (self.branch_count,):
                raise ValueError(
                    "Invalid branch index: "
                    "Boolean mask must have the same length as the number of branches: "
                    f"{self.branch_count}, instead of {ids_array.shape}."
                )
            return np.argwhere(ids_array).flatten()
        else:
            ids_array = ids_array.astype(int)
            if check and not np.all(np.logical_and(0 <= ids_array, ids_array < self.branch_count)):
                raise ValueError("Invalid branch index: out of range.")
            return ids_array

    ####################################################################################################################
    #  === COMPUTABLE GRAPH PROPERTIES ===
    ####################################################################################################################
    def branch_by_node_matrix(self) -> npt.NDArray[np.bool_]:
        """Compute the branches-by-nodes connectivity matrix from this graph branch list.

        The branches-by-nodes connectivity matrix is a 2D boolean matrix of shape (B, N) where B is the number of branches and N is the number of nodes. Each True value in the array indicates that the branch is connected to the corresponding node. This matrix therefore contains exactly two True values per row.

        Returns
        -------
        np.ndarray
            A 2D boolean array of shape (B, N) where N is the number of nodes and B is the number of branches.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2 ; 1➔3").branch_by_node_matrix()
        array([[ True,  True, False, False],
               [False,  True,  True, False],
               [False,  True, False,  True]])
        """  # noqa: E501
        n_branch = self.branch_count
        branch_by_node = np.zeros((n_branch, self.node_count), dtype=bool)
        branch_by_node[np.arange(n_branch)[:, None], self._branch_list] = True
        return branch_by_node

    def branch_adjacency_matrix(self) -> npt.NDArray[np.bool_]:
        """Compute the branch adjacency matrix from this graph branches-by-nodes connectivity matrix.

        The branch adjacency matrix is a 2D boolean matrix of shape (B, B) where B is the number of branches. Each True value in the array indicates that the two branches are connected by a node. This matrix is symmetric.

        Returns
        -------
        np.ndarray
            A 2D boolean array of shape (B, B) where B is the number of branches.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2➔3").branch_adjacency_matrix()
        array([[False,  True, False],
                [ True, False,  True],
                [False,  True, False]])
        """  # noqa: E501
        branch_by_node = self.branch_by_node_matrix()
        adj = (branch_by_node @ branch_by_node.T) > 0
        self_loops = self._branch_list[:, 0] == self._branch_list[:, 1]
        adj[np.diag_indices(self.branch_count)] = self_loops
        return adj

    def node_adjacency_matrix(self) -> npt.NDArray[np.bool_]:
        """Compute the node adjacency matrix from this graph branch list.

        The node adjacency matrix is a 2D boolean matrix of shape (N, N) where N is the number of nodes. Each True value in the array indicates that the two nodes are connected by a branch. This matrix is symmetric.

        Returns
        -------
        np.ndarray
            A 2D boolean array of shape (N, N) where N is the number of nodes.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2➔3").node_adjacency_matrix()
        array([[False,  True, False, False],
                [ True, False,  True, False],
                [False,  True, False,  True],
                [False, False,  True, False]])
        """  # noqa: E501
        node_adj = np.zeros((self.node_count, self.node_count), dtype=bool)
        node_adj[self._branch_list[:, 0], self._branch_list[:, 1]] = True
        node_adj[self._branch_list[:, 1], self._branch_list[:, 0]] = True
        return node_adj

    @overload
    def adjacent_branches(
        self, node_idx: IndexLike, return_branch_direction: Literal[False]
    ) -> npt.NDArray[np.int_]: ...
    @overload
    def adjacent_branches(
        self, node_idx: IndexLike, return_branch_direction: Literal[True]
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]: ...
    def adjacent_branches(
        self,
        node_idx: IndexLike,
        return_branch_direction: bool = False,
    ) -> npt.NDArray[np.int_] | Tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
        """Return the indices of the branches adjacent to the given node(s).

        Parameters
        ----------
        node_idx : IndexLike
            The indices of the node(s) to get the incident branches from.
            Valid indices are the same as for :meth:`VGraph.as_node_ids`.

        return_branch_direction : bool, optional
            If True, also return whether the branch is outgoing from the node. Default is False.

        Returns
        -------
        np.ndarray
            The indices of the branches incident to the given nodes.

        np.ndarray
            The direction of the branches incident to the given nodes (only returned if ``return_branch_direction`` is True).

            True indicates that the branch is outgoing from the node.

        See Also
        --------
        adjacent_branches_per_node : Perform the same operation but individually for each node.
        adjacent_nodes : Get the indices of the nodes adjacent to the given nodes.

        Examples
        --------
        >>> # Branch id:           0 1 2     3 4     5
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4➔5 ; 2➔6")

        >>> graph.adjacent_branches(0)
        array([0])

        >>> graph.adjacent_branches(1)
        array([0, 1, 3])

        >>> graph.adjacent_branches(1, return_branch_direction=True)
        (array([0, 1, 3]), array([False, True, True]))

        >>> graph.adjacent_branches([1, 6])
        array([0, 1, 3, 5])

        """  # noqa: E501
        node_idx = self.as_node_ids(node_idx)
        return self._adjacent_branches(
            node_idx=node_idx, return_branch_direction=return_branch_direction, individual_nodes=False
        )

    def adjacent_nodes(self, node_id: IndexLike) -> npt.NDArray[np.int_]:
        """Compute the indices of the nodes adjacent to the given node.

        Parameters
        ----------
        node_idx : IndexLike
            The indices of the nodes to get the adjacent nodes from.
            Valid indices are the same as for :meth:`VGraph.as_node_ids`.

        Returns
        -------
        npt.NDArray[np.int_]
            The indices of the nodes adjacent to the given nodes.

        See Also
        --------
        adjacent_branches : Get the indices of the branches adjacent to the given nodes.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2 ; 1➔3").adjacent_nodes(1)
        array([0, 2, 3])

        >>> VGraph.parse("0➔1➔2 ; 1➔3➔4➔5").adjacent_nodes([1,3])
        array([0, 2, 4])

        """
        node_id = np.unique(self.as_node_ids(node_id))
        adjacents = np.unique(self._branch_list[np.any(np.isin(self._branch_list, node_id), axis=1)])
        return np.setdiff1d(adjacents, node_id, assume_unique=True)

    @overload
    def adjacent_branches_per_node(
        self, node_idx: npt.ArrayLike[int], return_branch_direction: Literal[False]
    ) -> List[npt.NDArray[np.int_]]: ...
    @overload
    def adjacent_branches_per_node(
        self, node_idx: npt.ArrayLike[int], return_branch_direction: Literal[True]
    ) -> Tuple[List[npt.NDArray[np.int_], List[npt.NDArray[np.bool_]]]]: ...
    def adjacent_branches_per_node(
        self,
        node_idx: npt.ArrayLike[int],
        return_branch_direction: bool = False,
    ) -> List[npt.NDArray[np.int_]] | Tuple[List[npt.NDArray[np.int_], List[npt.NDArray[np.bool_]]]]:
        """Compute the indices of the branches incident to multiple nodes.
        In contrast to :meth:`VGraph.incident_branches`, this method returns the incident branches for each node separately.

        Parameters
        ----------
        node_idx : int or Iterable[int]
            The indices of the nodes to get the incident branches from.
            Valid indices are the same as for :meth:`VGraph.as_node_ids`.

        return_branch_direction : bool, optional
            If True, also return whether the branch is outgoing from the node. Default is False.

        Returns
        -------
        List[np.ndarray]
            The indices of the branches incident to the given nodes.

        List[np.ndarray]
            The direction of the branches incident to the given nodes. Only returned if ``return_branch_direction`` is True. True indicates that the branch is outgoing from the node.

        See Also
        --------
        adjacent_branches : Perform the same operation but for all node indistinctly.

        Examples
        --------
        >>> # Branch id:           0 1 2     3 4     5
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4➔5 ; 2➔6")

        >>> graph.adjacent_branches_per_node([1, 6])
        [array([0, 1, 3]), array([5])]

        >>> graph.adjacent_branches_per_node([1, 6], return_branch_direction=True)
        ([array([0, 1, 3]), array([5])], [array([False, True, True]), array([False])])

        """  # noqa: E501
        return self._adjacent_branches(
            node_idx=node_idx, return_branch_direction=return_branch_direction, individual_nodes=True
        )

    def _adjacent_branches(
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
        elif len(node_idx) == 0:
            if individual_nodes:
                return [], [] if return_branch_direction else []
            return (
                (np.empty((0,), dtype=int), np.empty((0,), dtype=bool))
                if return_branch_direction
                else np.empty((0,), dtype=int)
            )

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

    def node_degree(
        self, node_id: Optional[int | npt.ArrayLike] = None, /, *, count_loop_branches_once: bool = False
    ) -> npt.NDArray[np.int_]:
        """Compute the degree of each node in the graph.

        The degree of a node is its number of adjacent (or incident) branches.

        Parameters
        ----------
        node_id : int or Iterable[int], optional
            The indices of the nodes to compute the degree from. If None, the degree of all nodes is computed.

        count_loop_branches_only_once : bool, optional
            If True, a branch connecting a node to itself are counted as one incident branch (instead of two). Default is False.

        Returns
        -------
        np.ndarray
            An array of shape (N,) containing the degree of each node.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4")

        >>> graph.node_degree(0)
        1

        >>> graph.node_degree([1, 2])
        array([3, 2])

        >>> graph.node_degree()
        array([1, 3, 2, 1, 1])

        """  # noqa: E501
        if np.isscalar(node_id):
            return (
                np.sum(np.any(self._branch_list == node_id, axis=1))
                if count_loop_branches_once
                else np.sum(self._branch_list == node_id)
            )
        elif node_id is not None:
            nodes_id = np.asarray(node_id, dtype=int)
            if count_loop_branches_once:
                return np.sum(np.any(self._branch_list[None, :, :] == nodes_id[:, None, None], axis=2), axis=1)
            else:
                return np.sum(self._branch_list.flatten()[None, :] == nodes_id[:, None], axis=1)
        else:
            node_count = np.bincount(self._branch_list.flatten(), minlength=self.node_count)
            if count_loop_branches_once:
                loop_nodes, loops_count = np.unique(
                    self._branch_list[self._branch_list[:, 0] == self._branch_list[:, 1], 0], return_counts=True
                )
                node_count[loop_nodes] -= loops_count
            return node_count

    @overload
    def endpoint_nodes(self, as_mask: Literal[False] = False) -> npt.NDArray[np.int_]: ...
    @overload
    def endpoint_nodes(self, as_mask: Literal[True]) -> npt.NDArray[np.bool_]: ...
    def endpoint_nodes(self, as_mask=False) -> npt.NDArray[np.int_ | np.bool_]:
        """Return the indices of the endpoint nodes in the graph.

        The endpoint nodes are the nodes connected to exactly one branch (i.e. of degree 1).

        .. note::
            If a node is connected to itself, it is **not** considered as an endpoint node.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the endpoint nodes instead of their indices.

        Returns
        -------
        np.ndarray
            The indices of the endpoint nodes (or if ``as_mask`` is True, a boolean mask of shape (N,) where N is the number of nodes).

        See Also
        --------
        junction_nodes : Get the indices of the junctions (non-endpoints) nodes.

        passing_nodes : Get the indices of the nodes connected to exactly two branches.

        endpoint_nodes_with_branch_id : Get the indices of the endpoint nodes along with their incident branches.

        Examples
        --------
        >>> VGraph.parse("0➔1➔2➔3 ; 1➔4").endpoint_nodes()
        array([0, 3, 4])

        >>> VGraph.parse("0➔1➔2➔3 ; 1➔4").endpoint_nodes(as_mask=True)
        array([ True, False, False,  True,  True])
        """  # noqa: E501
        mask = np.bincount(self._branch_list.flatten(), minlength=self.node_count) == 1
        return mask if as_mask else np.argwhere(mask).flatten()

    @overload
    def endpoint_nodes_with_branch_id(
        self, *, return_branch_direction: Literal[False] = False
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: ...
    @overload
    def endpoint_nodes_with_branch_id(
        self, *, return_branch_direction: Literal[True]
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.bool_]]: ...
    def endpoint_nodes_with_branch_id(
        self, *, return_branch_direction: bool = False
    ) -> (
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]
        | Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.bool_]]
    ):
        """Return the indices of the nodes that are connected to exactly one branch along with the indices of their incident branches.

        Parameters
        ----------
        return_branch_direction : bool, optional
            If True, also return whether the branch is outgoing from the node. Default is False.

        Returns
        -------
        nodes_index: npt.NDArray[np.int_]
            The indices of the endpoints nodes.

        incident_branch_index: npt.NDArray[np.int_]
            The indices of the branches connected to the passing nodes.

        branch_direction: npt.NDArray[np.bool_] (optional)
            An array of shape (N,) indicating the direction of the branches according to :attr:`VGraph.branch_list`: True indicates that the passing node is the second node of the branch.

            (This is only returned if ``return_branch_direction`` is True.)


        Examples
        --------
        >>> # Branch id:           0 1 2     3
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4")

        >>> graph.endpoint_nodes_with_branch_id()
        (array([0, 3, 4]), array([0, 2, 3], dtype=int32))

        >>> graph.endpoint_nodes_with_branch_id(return_branch_direction=True)
        (array([0, 3, 4]), array([0, 2, 3], dtype=int32), array([ True, False, False]))



        """  # noqa: E501
        endpoint_nodes = self.endpoint_nodes(as_mask=False)
        incident_branches = first_index_of(self._branch_list.flatten(), endpoint_nodes)
        branch_index = incident_branches // 2
        if return_branch_direction:
            branch_dirs = incident_branches % 2 == 0
            return endpoint_nodes, branch_index, branch_dirs
        else:
            return endpoint_nodes, branch_index

    @overload
    def junction_nodes(self, as_mask: Literal[False] = False) -> npt.NDArray[np.int_]: ...
    @overload
    def junction_nodes(self, as_mask: Literal[True]) -> npt.NDArray[np.bool_]: ...
    def junction_nodes(self, as_mask=False) -> npt.NDArray[np.int_ | np.bool_]:
        """Return the indices of the junctions (non-endpoints) nodes in the graph.

        The junction nodes are the nodes connected to at least two branches.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the junction nodes instead of their indices.

        Returns
        -------
        np.ndarray
            The indices of the junction nodes (or if ``as_mask`` is True, a boolean mask of shape (N,) where N is the number of nodes).

        See Also
        --------
        endpoint_nodes : Get the indices of the nodes connected to exactly one branch.

        passing_nodes : Get the indices of the nodes connected to exactly two branches.


        Examples
        --------
        >>> VGraph.parse("0➔1➔2➔3 ; 1➔4").junction_nodes()
        array([1, 2])

        >>> VGraph.parse("0➔1➔2➔3 ; 1➔4").junction_nodes(as_mask=True)
        array([False,  True,  True, False, False])

        """  # noqa: E501
        mask = np.bincount(self._branch_list.flatten(), minlength=self.node_count) > 1
        return mask if as_mask else np.argwhere(mask).flatten()

    @overload
    def passing_nodes(self, *, as_mask: Literal[False] = False, exclude_loop: bool = True) -> npt.NDArray[np.int_]: ...
    @overload
    def passing_nodes(self, *, as_mask: Literal[True], exclude_loop: bool = True) -> npt.NDArray[np.bool_]: ...
    def passing_nodes(
        self, *, as_mask=False, exclude_loop: bool = True
    ) -> npt.NDArray[np.int_ | np.bool_] | Tuple[npt.NDArray[np.int_ | np.bool_], List[npt.NDArray[np.int_]]]:
        """Return the indices of the nodes that are connected to exactly two branches.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the passing nodes instead of their indices.
        exclude_loop : bool, optional
            If True (by default), exclude the nodes connected to themselves.

        Returns
        -------
        passing_nodes_index: np.ndarray
            The indices of the passing nodes (or if ``as_mask`` is True, a boolean mask of shape (N,) where N is the number of nodes).

        incident_branch_index: List[np.ndarray]
            The indices of the branches connected to the passing nodes. Only returned if ``return_branch_index`` is True.

        See Also
        --------
        passing_nodes_with_branch_index : Get the indices of the passing nodes along with their incident branches.

        Examples
        --------
        >>> # Branch id:           0 1 2     3 4     5
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4➔5 ; 6➔6")

        >>> graph.passing_nodes()
        array([2, 4])

        >>> graph.passing_nodes(as_mask=True)
        array([False, False,  True, False,  True, False, False])

        >>> graph.passing_nodes(exclude_loop=False)
        array([2, 4, 6])

        """  # noqa: E501
        branch_list = self._branch_list
        if exclude_loop:
            branch_list = branch_list[branch_list[:, 0] != branch_list[:, 1]]
        mask = np.bincount(branch_list.flatten(), minlength=self.node_count) == 2

        return mask if as_mask else np.argwhere(mask).flatten()

    @overload
    def passing_nodes_with_branch_index(
        self, *, return_branch_direction: Literal[False] = False, exclude_loop: bool = True
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]: ...
    @overload
    def passing_nodes_with_branch_index(
        self, *, return_branch_direction: Literal[True], exclude_loop: bool = True
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.bool_]]: ...
    def passing_nodes_with_branch_index(
        self, *, return_branch_direction: bool = False, exclude_loop: bool = True
    ) -> (
        Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]
        | Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.bool_]]
    ):
        """Return the indices of the nodes that are connected to exactly two branches along with the indices of these branches.

        Parameters
        ----------
        return_branch_direction : bool, optional
            If True, also return the branch direction. Default is False.

        exclude_loop : bool, optional
            If True (by default), exclude the nodes connected to themselves.

        Returns
        -------
        passing_nodes_index: npt.NDArray[np.int_]
            The indices of the passing nodes as a array of shape (N,) where N is the number of passing nodes.

        incident_branch_index: npt.NDArray[np.int_]
            The indices of the branches connected to the passing nodes as an array of shape (N, 2)

        branch_direction: npt.NDArray[np.bool_] (optional)
            An array of shape (N,2) indicating the direction of the branches according to :attr:`VGraph.branch_list`:
            - For each first branch, True indicates that the branch is directed towards the passing node
            - For each second branch, True indicates that the branch is directed away from the passing node.

            (This is only returned if ``return_branch_direction`` is True.)

        See Also
        --------
        passing_nodes : Get the indices of the nodes connected to exactly two branches.

        Examples
        --------
        >>> # Branch id:           0 1 2     3 4     5
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 5➔4➔1 ; 6➔6")

        >>> graph.passing_nodes_with_branch_index()
        (array([2, 4]), array([[1, 2], [3, 4]], dtype=int32))

        >>> graph.passing_nodes_with_branch_index(return_branch_direction=True)
        (array([2, 4]), array([[1, 2], [3, 4]], dtype=int32), array([[ True,  True], [False, False]]))

        """  # noqa: E501
        passing_nodes = self.passing_nodes(as_mask=False, exclude_loop=exclude_loop)
        incident_branches = first_two_index_of(self._branch_list.flatten(), passing_nodes)
        branch_index = incident_branches // 2
        if return_branch_direction:
            branch_dirs = incident_branches < self._branch_list.shape[0]
            return passing_nodes, branch_index, branch_dirs
        else:
            return passing_nodes, branch_index

    @overload
    def endpoint_branches(
        self, *, as_mask=False, return_node_mask: Literal[False] = False
    ) -> npt.NDArray[np.int_ | np.bool_]: ...
    @overload
    def endpoint_branches(
        self, *, as_mask=False, return_node_mask: Literal[True]
    ) -> Tuple[npt.NDArray[np.int_ | np.bool_], npt.NDArray[np.bool_]]: ...
    def endpoint_branches(
        self, *, as_mask=False, return_node_mask=False
    ) -> npt.NDArray[np.int_ | np.bool_] | Tuple[npt.NDArray[np.int_ | np.bool_], npt.NDArray[np.bool_]]:
        """Return the indices of the terminal branches in the graph.

        The terminal branches are the branches connected to endpoint nodes.

        Parameters
        ----------

        as_mask : bool, optional
            If True, return a mask of the terminal branches instead of their indices.

        return_node_mask : bool, optional
            If True, return a mask of the nodes of the terminal branches indicating which are endpoints.

        Returns
        -------
        endpoints_branches : npt.NDArray[np.int_ | np.bool_]
            An array of length (B,) containing the indices of the terminal branches (or a mask of the terminal branches if ``as_mask`` is True).

        node_mask : npt.NDArray[np.bool_]
            If ``return_node_mask`` is True, a 2D boolean array of shape (B, 2) indicating for each branch if its first and second node are endpoints.

        See Also
        --------
        endpoint_nodes : Get the indices of the nodes connected to exactly one branch.

        Examples
        --------
        >>> # Branch id:           0 1 2     3
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔4")

        >>> graph.endpoint_branches()
        array([0, 2, 3])

        >>> graph.endpoint_branches(as_mask=True)
        array([ True, False,  True, True])

        >>> graph.endpoint_branches(return_node_mask=True)
        (array([0, 2, 3]), array([[ True, False], [False,  True], [False,  True]]))

        """  # noqa: E501
        endpoint_nodes = self.endpoint_nodes(as_mask=True)
        endpoints_mask = endpoint_nodes[self._branch_list]
        endpoints_branches = np.any(endpoints_mask, axis=1)
        if not as_mask:
            endpoints_branches = np.argwhere(endpoints_branches).flatten()
            endpoints_mask = endpoints_mask[endpoints_branches]
        return (endpoints_branches, endpoints_mask) if return_node_mask else endpoints_branches

        ## Legacy implementation (slower?)
        # _, branch_index, node_count = np.unique(self._branch_list, return_counts=True, return_index=True)
        # return np.unique(branch_index[node_count == 1] // 2)

    def orphan_branches(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Return the indices of the branches connected to no other branch in the graph.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the orphan branches instead of their indices.

        Returns
        -------
        np.ndarray
            The indices of the branches connected to a single node.

        Examples
        --------
        >>> # Branch id:           0 1     2    3
        >>> graph = VGraph.parse("0➔1➔2 ; 3➔4; 5➔5")

        >>> graph.orphan_branches()
        array([2, 3])

        >>> graph.orphan_branches(as_mask=True)
        array([False, False,  True,  True])

        """
        terminal_nodes = self.endpoint_nodes(as_mask=True)
        orphan_branches_mask = np.all(terminal_nodes[self._branch_list], axis=1) | (
            self._branch_list[:, 0] == self._branch_list[:, 1]
        )
        return orphan_branches_mask if as_mask else np.argwhere(orphan_branches_mask).flatten()

    def self_loop_branches(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indices of the branches connecting a node to itself in the graph.

        Parameters
        ----------
        as_mask : bool, optional
            If True, return a mask of the self-loop branches instead of their indices.

        Returns
        -------
        np.ndarray
            The indices of the self-loop branches.

        Examples
        --------
        >>> # Branch id:   0 1     2    3
        >>> VGraph.parse("0➔1➔2 ; 3➔4; 5➔5").self_loop_branches()
        array([3])
        """
        self_loop_mask = self._branch_list[:, 0] == self._branch_list[:, 1]
        return self_loop_mask if as_mask else np.argwhere(self_loop_mask).flatten()

    def twin_branches(self) -> List[npt.NDArray[np.int_]]:
        """Compute the indices of the branches that are twins in the graph.

        Returns
        -------
        List[np.ndarray]
            A list of arrays containing the indices of the twin branches.

        Examples
        --------
        >>> # Branch id:           0 1 2     3     4 5
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 1➔2 ; 2➔3➔2")

        >>> graph.twin_branches()
        [array([1, 3]), array([2, 4, 5])]

        """  # noqa: E501
        branch_list = np.sort(self._branch_list, axis=1)
        _, inv, counts = np.unique(branch_list, return_inverse=True, return_counts=True, axis=0)
        return [np.argwhere(inv == twin_id).flatten() for twin_id in np.argwhere(counts > 1).flatten()]

    def node_connected_components(self) -> List[npt.NDArray[np.int_]]:
        """Compute the connected components of the graph and return, for each of them, its nodes indices.

        Returns
        -------
        List[npt.NDArray[np.int_]]
            A tuple of arrays containing the indices of the nodes included in each connected component.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 4➔4 ; 5➔6")

        >>> graph.node_connected_components()
        [array([0, 1, 2, 3]), array([4]), array([5, 6])]
        """
        return [np.asarray(cc, dtype=int) for cc in reduce_clusters(self._branch_list, drop_singleton=False)]

    ####################################################################################################################
    #  === COMBINE GEOMETRIC DATA ===
    ####################################################################################################################
    def node_coord(self) -> npt.NDArray[np.float32]:
        """Compute the coordinates of the nodes in the graph (averaged from all geometric data).

        Returns
        -------
        np.ndarray
            An array of shape (N, 2) containing the (y, x) positions of the nodes in the graph.
        """
        assert len(self._geometric_data) > 0, "No geometric data available to retrieve the nodes coordinates."
        if len(self._geometric_data) == 1:
            # Use the only geometric data
            return self.geometric_data().node_coord()

        # Average nodes coordinates from all geometric data
        coord = np.zeros((self.node_count, 2), dtype=float)
        count = np.zeros(self.node_count, dtype=int)
        for gdata in self._geometric_data:
            nodes_id = gdata.node_ids
            coord[nodes_id] += gdata.node_coord()
            count[nodes_id] += 1
        return coord / count[:, None]

    def branch_node2node_length(self, branch_id: Optional[int | Iterable[int]] = None) -> npt.NDArray[np.float64]:
        """Compute the length of the branches in the graph (averaged from all geometric data).

        Parameters
        ----------
        branch_id : int or Iterable[int], optional
            The indices of the branches to compute the length from. If None, the length of all branches is computed.

        Returns
        -------
        np.ndarray
            An array of shape (B,) containing the length of the branches.
        """
        if branch_id is None:
            branches = self._branch_list
        else:
            if isinstance(branch_id, int):
                branch_id = [branch_id]
            branches = self._branch_list[branch_id]

        return np.linalg.norm(self.node_coord()[branches[:, 0]] - self.node_coord()[branches[:, 1]], axis=1)

    def branch_arc_length(
        self, ids: Optional[int | Iterable[int]] = None, fast_approximation=True
    ) -> npt.NDArray[np.float64]:
        """Compute the arc length of the branches in the graph (averaged from all geometric data).

        Parameters
        ----------
        ids : int or Iterable[int], optional
            The indices of the branches to compute the arc length from. If None, the arc length of all branches is computed.

        Returns
        -------
        np.ndarray
            An array of shape (B,) containing the arc length of the branches.
        """  # noqa: E501
        assert len(self._geometric_data) > 0, "No geometric data available to compute the arc length."
        if len(self._geometric_data) == 1:
            # Use the only geometric data
            return self.geometric_data().branch_arc_length(ids, fast_approximation=fast_approximation)

        # Average arc length from all geometric data...
        if ids is None:
            # ... for all branches
            total_arc_length = np.zeros(self.branch_count, dtype=float)
            count = np.zeros(self.branch_count, dtype=int)
            for gdata in self._geometric_data:
                id = gdata.branch_ids
                total_arc_length[id] += gdata.branch_arc_length(fast_approximation=fast_approximation)
                count[id] += 1
        else:
            if isinstance(ids, int):
                ids = [ids]
            # ... for a subset of branches
            total_arc_length = np.zeros(len(ids), dtype=float)
            count = np.zeros(len(ids), dtype=int)
            for gdata in self._geometric_data:
                id = np.intersect1d(ids, gdata.branch_ids)
                total_arc_length[id] += gdata.branch_arc_length(id, fast_approximation=fast_approximation)
                count[id] += 1

        return total_arc_length / count

    def branch_chord_length(self, branch_id: Optional[int | Iterable[int]] = None) -> npt.NDArray[np.float64]:
        """Compute the chord length of the branches in the graph (averaged from all geometric data).

        Parameters
        ----------
        branch_id : int or Iterable[int], optional
            The indices of the branches to compute the chord length from. If None, the chord length of all branches is computed.

        Returns
        -------
        np.ndarray
            An array of shape (B,) containing the chord length of the branches.
        """  # noqa: E501
        assert len(self._geometric_data) > 0, "No geometric data available to compute the chord length."
        if len(self._geometric_data) == 1:
            # Use the only geometric data
            return self.geometric_data().branch_chord_length(branch_id)

        # Average chord length from all geometric data...
        if branch_id is None:
            # ... for all branches
            total_chord_length = np.zeros(self.branch_count, dtype=float)
            count = np.zeros(self.branch_count, dtype=int)
            for gdata in self._geometric_data:
                id = gdata.branch_ids
                total_chord_length[id] += gdata.branch_chord_length()
                count[id] += 1
        else:
            # ... for a subset of branches
            total_chord_length = np.zeros(len(branch_id), dtype=float)
            count = np.zeros(len(branch_id), dtype=int)
            for gdata in self._geometric_data:
                id = np.intersect1d(branch_id, gdata.branch_ids)
                total_chord_length[id] += gdata.branch_chord_length(id)
                count[id] += 1

        return total_chord_length / count

    def branch_label_map(self, geometrical_data_priority: Optional[int | Iterable[int]] = None) -> npt.NDArray[np.int_]:
        """Compute the label map of the branches in the graph.

        In case of overlapping branches, the label of the branches is determined by the priority of the geometrical data.

        Parameters
        ----------
        geometrical_data_priority : int or Iterable[int], optional
            The indices of the geometrical data to use to determine the label of the branches. The first geometrical data in the list has the highest priority.

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
            label_map[domain.slice()] = gdata.branch_label_map()
        return label_map

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
    def reindex_nodes(self, indices, *, inverse_lookup=False, inplace=False) -> VGraph:
        """Reindex the nodes of the graph.

        Parameters
        ----------
        new_node_indexes : npt.NDArray
            The new indices of the nodes.

        inverse_lookup : bool, optional

            - If False, indices is sorted by old indices and contains the new one: indices[old_index] -> new_index.
            - If True, indices is sorted by new indices and contains the old one: indices[new_index] -> old_index.

            By default: False.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise (by default), a modified copy of the graph is returned.

        Returns
        -------
        VGraph
            The modified graph.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2 ; 1➔3")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [1, 3]]

        Reindex the nodes as: 0➔2, 1➔0, 2➔1, 3➔3

        >>> g1 = graph.reindex_nodes([2, 0, 1, 3])
        >>> g1.branch_list.tolist()
        [[2, 0], [0, 1], [0, 3]]

        Reindex the nodes back to their original indices: 2➔0, 0➔1, 1➔2, 3➔3
        >>> g2 = g1.reindex_nodes([2, 0, 1, 3], inverse_lookup=True)
        >>> g2.branch_list.tolist()
        [[0, 1], [1, 2], [1, 3]]
        """
        graph = self if inplace else self.copy()

        indices = complete_lookup(indices, max_index=graph.node_count - 1)
        if inverse_lookup:
            indices = invert_complete_lookup(indices)

        # Update nodes indices in ...
        graph._branch_list = indices[graph._branch_list]  # ... branch list
        graph._node_attr = graph._node_attr.set_index(indices)  # ... nodes attributes
        for gdata in graph._geometric_data:  # ... geometric data
            gdata._reindex_nodes(indices)

        # Update nodes indices in ...
        indices = add_empty_to_lookup(indices, increment_index=False)  # Insert -1 in lookup for missing nodes
        for node_ref in graph._node_refs:  # ... nodes references
            if node_ref._id > -1:
                node_ref._id = indices[node_ref._id + 1]
        for branch_ref in graph._branch_refs:  # ... nodes indices stored in branches references
            if branch_ref.is_valid():
                branch_ref._nodes_id = indices[branch_ref._nodes_id]
        return graph

    def reindex_branches(self, indices, inverse_lookup=False) -> VGraph:
        """Reindex the branches of the graph.

        Parameters
        ----------
        indices : npt.NDArray
            A lookup table to reindex the branches.

        inverse_lookup : bool, optional

            - If False, indices is sorted by old indices and contains the new one: indices[old_index] -> new_index.
            - If True, indices is sorted by new indices and contains the old one: indices[new_index] -> old_index.

            By default: False.

        Returns
        -------
        VGraph
            The modified graph.

        Examples
        --------
        >>> # Branch id:           0 1 2 3
        >>> graph = VGraph.parse("0➔1➔2➔3➔4")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3], [3, 4]]

        Reindex the branches as: 0➔2, 1➔0, 2➔1, 3➔3

        >>> g1 = graph.reindex_branches([2, 0, 1, 3])
        >>> g1.branch_list.tolist()
        [[1, 2], [2, 3], [0, 1], [3, 4]]

        Reindex the branches back to their original indices: 2➔0, 0➔1, 1➔2, 3➔3
        >>> g2 = g1.reindex_branches([2, 0, 1, 3], inverse_lookup=True)
        >>> g2.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3], [3, 4]]

        """
        indices = complete_lookup(indices, max_index=self.branch_count - 1)
        if inverse_lookup:
            indices = invert_complete_lookup(indices)

        # Update branches indices in ...
        self._branch_list[indices, :] = self._branch_list.copy()  # ... branch list
        self._branch_attr = self._branch_attr.set_index(indices).reindex(
            pd.RangeIndex(self.branch_count), copy=False
        )  # ... branches attributes
        for gdata in self._geometric_data:
            gdata._reindex_branches(indices)

        # Update branches indices in ...
        indices = add_empty_to_lookup(indices, increment_index=False)  # Insert -1 in lookup for missing branches
        for branch_ref in self._branch_refs:  # ... branches references
            branch_ref._id = indices[branch_ref._id + 1]
        for node_ref in self._node_refs:  # ... incident branches stored in nodes references
            if node_ref._ibranch_ids is not None:
                node_ref._ibranch_ids = indices[node_ref._ibranch_ids + 1]
        return self

    def flip_branch_direction(self, branch_id: IndexLike, inplace=False) -> VGraph:
        """Flip the direction of the branches in the graph.

        In ``branch_list``, the first node of the branch becomes the second and vice versa. The branch curve and all geometric data are also flipped.

        Parameters
        ----------
        branch_id : int or Iterable[int]
            The indices of the branches to flip.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise (by default), a modified copy of the graph is returned.

        Returns
        -------
        VGraph
            The modified graph.

        Examples
        --------
        >>> # Branch id:           0 1 2
        >>> graph = VGraph.parse("0➔1➔2➔3")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3]]

        Flip the direction of the branches 0 and 2

        >>> g1 = graph.flip_branch_direction([0, 2])
        >>> g1.branch_list.tolist()
        [[1, 0], [1, 2], [3, 2]]
        """  # noqa: E501
        graph = self if inplace else self.copy()

        branches_id = graph.as_branch_ids(branch_id)

        # Flip branches in ...
        graph._branch_list[branches_id] = graph._branch_list[branches_id, ::-1]  # ... branch list
        for gdata in graph._geometric_data:  # ... geometric data
            gdata._flip_branch_direction(branches_id)
        for branch_ref in graph._branch_refs:  # ... branches references
            if branch_ref._id in branches_id:
                branch_ref._nodes_id = graph._branch_list[branch_ref._id]
        for node_ref in graph._node_refs:  # ... incident branches stored in nodes references
            if node_ref._ibranch_ids is not None and np.any(np.isin(node_ref._ibranch_ids, branches_id)):
                node_ref._ibranch_dirs = node_ref._ibranch_dirs[node_ref._ibranch_ids][:, 0] == node_ref._id
        return graph

    def sort_branches_by_nodesID(self, descending=False, inplace: bool = False) -> VGraph:
        """Flip and sort the branches of the graph by the ID of their first node.

        This method is mainly useful for visualization and debug purposes.

        Parameters
        ----------
        descending : bool, optional
            If True, sort the branches in descending order of their first node ID. By default: False.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise (by default), a modified copy of the graph is returned.

        Returns
        -------
        VGraph
            The modified graph.
        """
        flip = self._branch_list[:, 0] > self._branch_list[:, 1]
        self.flip_branch_direction(np.argwhere(flip).flatten(), inplace=True)

        nodesID = self._branch_list[:, 0]
        new_order = np.argsort(nodesID)
        if descending:
            new_order = new_order[::-1]
        return self.reindex_branches(new_order)

    # --- Private base edition ---
    def _delete_branch(self, branch_id: npt.NDArray[np.int_], update_refs: bool = True) -> npt.NDArray[np.int_]:
        if len(branch_id) == 0:
            return np.arange(self.branch_count)

        branches_reindex = create_removal_lookup(branch_id, replace_value=-1, length=self.branch_count)
        for gdata in self._geometric_data:
            gdata._reindex_branches(branches_reindex)

        self._branch_list = np.delete(self._branch_list, branch_id, axis=0)
        self._branch_attr = self._branch_attr.drop(branch_id).reset_index(drop=True)

        if update_refs:
            reindex_refs = add_empty_to_lookup(branches_reindex, increment_index=False)
            for branch in self._branch_refs:
                branch._id = reindex_refs[branch._id + 1]

        return branches_reindex

    def _delete_node(self, node_id: npt.NDArray[np.int_], update_refs: bool = True) -> npt.NDArray[np.int_]:
        if len(node_id) == 0:
            return
        self._node_attr = self._node_attr.drop(node_id).reset_index(drop=True)

        nodes_reindex = create_removal_lookup(node_id, replace_value=-1, length=self.node_count)
        self._branch_list = nodes_reindex[self._branch_list]
        for gdata in self._geometric_data:
            gdata._reindex_nodes(nodes_reindex)

        self._nodes_count -= len(node_id)

        if update_refs:
            nodes_reindex = add_empty_to_lookup(nodes_reindex, increment_index=False)
            for node in self._node_refs:
                node._id = nodes_reindex[node._id + 1]
            for branch in self._branch_refs:
                if branch.is_valid():
                    branch._nodes_id = self._branch_list[branch._id]

        return nodes_reindex

    # --- Branches edition ---
    def delete_branch(self, branch_id: IndexLike, delete_orphan_nodes=True, *, inplace=False) -> VGraph:
        """Remove the branches with the given indices from the graph.

        Parameters
        ----------
        branch_indexes : int | npt.ArrayLike[int]
            The indices of the branches to remove from the graph.

        delete_orphan_nodes : bool, optional
            If True (by default), the nodes that are not connected to any branch after the deletion are removed from the graph.

            .. warning::
                Several method of this FVT assumes that the graphes contains no orphan nodes. Disabling this option is not recommended.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.

        Examples
        --------
        >>> # Branch id:           0 1     2 3     4
        >>> graph = VGraph.parse("0➔1➔2 ; 0➔3➔4 ; 5➔5")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [0, 3], [3, 4], [5, 5]]

        >>> g1 = graph.delete_branch([0, 3, 4])
        >>> g1.branch_list.tolist()
        [[1, 2], [0, 3]]

        When a removing a branch connected to an endpoint node, the node is also removed. The index of the remaining nodes are shifted.

        >>> g2 = graph.delete_branch(1)
        >>> g2.branch_list.tolist()
        [[0, 1], [0, 2], [2, 3], [4, 4]]

        """  # noqa: E501
        graph = self if inplace else self.copy()
        branch_indexes = graph.as_branch_ids(branch_id)

        # Find connected nodes
        connected_nodes = np.array([])
        if delete_orphan_nodes or graph._node_refs:
            connected_nodes = np.unique(graph._branch_list[branch_indexes])

            # Clear the incident branch cache of nodes connected to the deleted branches
            for node in graph._node_refs:
                if node._id in connected_nodes:
                    node.clear_adjacent_branch_cache()

        # Remove branches and orphan nodes
        graph._delete_branch(branch_indexes)
        if delete_orphan_nodes:
            orphan_nodes = connected_nodes[np.isin(connected_nodes, graph._branch_list, invert=True)]
            graph._delete_node(orphan_nodes)

        return graph

    def add_branch(
        self, branch_nodes: npt.ArrayLike[int], *, return_branch_id=False, inplace=True
    ) -> VGraph | Tuple[VGraph, npt.NDArray[np.int_]]:
        """Add branch(es) to the graph.

        Parameters
        ----------
        branch_nodes : npt.ArrayLike[int]
            A 2D array of shape (N, 2) containing the indices of the nodes connected by the new branches.

        return_branch_id : bool, optional
            If True, return the indices of the added branches.

        inplace : bool, optional
            If True (by default), the graph is modified in place. Otherwise, a new graph is returned.
        Returns
        -------
        VGraph
            The modified graph.

        new_branch_ids : np.ndarray
            The indices of the added branches. Only returned if ``return_branch_id`` is True.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2➔3")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3]]

        Add a new branch connecting nodes 1 and 3

        >>> g1 = graph.add_branch([1, 3])
        >>> g1.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3], [1, 3]]

        Add two new branches connecting nodes 0 and 2, and 0 and 3:

        >>> g2, new_branch_id = graph.add_branch([[0, 2], [0, 3]], return_branch_id=True)
        >>> new_branch_id
        array([4, 5])

        >>> g2.branch_list[new_branch_id]
        array([[0, 2], [0, 3]])
        """
        graph = self if inplace else self.copy()

        branch_nodes = np.atleast_2d(branch_nodes).astype(int)
        assert branch_nodes.shape[1] == 2, "branch_nodes must be a 2D array of shape (N, 2)."
        assert np.all(np.logical_and(branch_nodes >= 0, branch_nodes < graph.node_count)), "Invalid node indices."

        graph._branch_list = np.concatenate((graph._branch_list, branch_nodes), axis=0)
        graph._branch_attr = graph._branch_attr.reindex(pd.RangeIndex(len(graph._branch_list)), copy=False)

        for gdata in graph._geometric_data:
            gdata._append_empty_branches(len(branch_nodes))

        return (
            graph
            if not return_branch_id
            else (graph, np.arange(graph.branch_count - len(branch_nodes), graph.branch_count))
        )

    def duplicate_branch(
        self, branch_id: IndexLike, *, return_branch_id=False, inplace=False
    ) -> VGraph | Tuple[VGraph, npt.NDArray[np.int_]]:
        """Duplicate branches in the graph.

        Parameters
        ----------
        branch_indexes : int or Iterable[int]
            The indices of the branches to duplicate.

        return_branch_id : bool, optional
            If True, return the indices of the duplicated branches.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.

        new_branch_ids : np.ndarray
            The indices of the duplicated branches. Only returned if ``return_branch_id`` is True.

        Examples
        --------
        >>> # Branch id:           0 1 2
        >>> graph = VGraph.parse("0➔1➔2➔3")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3]]

        Duplicate the branch 1

        >>> g1 = graph.duplicate_branch(1)
        >>> g1.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3], [1, 2]]

        Duplicate the branches 0 and 2

        >>> g2, new_branch_id = graph.duplicate_branch([0, 2], return_branch_id=True)
        >>> new_branch_id
        array([3, 4])

        >>> g2.branch_list[new_branch_id]
        array([[0, 1], [2, 3]])

        """
        branches_id = self.as_branch_ids(branch_id)
        if len(branches_id) == 0:
            return self if not return_branch_id else (self, np.empty(0, dtype=int))
        graph = self if inplace else self.copy()

        branch_nodes = graph._branch_list[branches_id]
        oldB = graph.branch_count
        newB = oldB + len(branches_id)
        new_branches_id = np.arange(oldB, newB)

        graph._branch_list = np.concatenate((graph._branch_list, branch_nodes), axis=0)
        graph._branch_attr = graph._branch_attr.reindex(pd.RangeIndex(newB))
        graph._branch_attr.iloc[oldB:] = graph._branch_attr.iloc[branches_id]

        for gdata in graph._geometric_data:
            gdata._append_branch_duplicates(branches_id, new_branches_id)

        return graph if not return_branch_id else (graph, new_branches_id)

    def split_branch(
        self,
        branchID: int,
        split_curve_id: npt.ArrayLike[int] | npt.ArrayLike[float],
        split_coord: Optional[npt.ArrayLike[int]] = None,
        *,
        return_branch_ids=False,
        return_node_ids=False,
        inplace=False,
    ) -> VGraph | Tuple[VGraph, npt.NDArray[np.int_]]:
        """Split a branch into two branches by adding a new node near the given coordinates.

        The new node is added to the nodes coordinates and the two new branches are added to the branch list.

        Parameters
        ----------
        branchID : int
            The index of the branch to split.
        split_curve_id : npt.ArrayLike[int] | npt.ArrayLike[float]
            An 1D array of size N specifying where to split the branch curve. The values are either the index of the curve point or the relative position along the curve (between 0 and 1).

        split_coord : npt.ArrayLike[int], optional
            An 2D array of shape (N, 2) containing the coordinates of the new nodes. If None, the new nodes are added on the split points of the branch curve.

        return_branch_ids : bool, optional
            If True, return the indices of the split and new branches. By default: False.

        return_node_ids : bool, optional
            If True, return the indices of the new nodes. By default: False.

        Returns
        -------
        graph: VGraph
            The modified graph.

        branch_ids : np.ndarray
            An array of shape (N+1,) with the indices of the modified branches: the splitted branch and the new branches.

            Only returned if ``return_branch_id`` is True.

        new_node_ids : np.ndarray
            An array of shape (N,) with the indices of the new nodes. Only returned if ``return_node_ids`` is True.

        Examples
        --------
        >>> # Branch id:           0 1
        >>> graph = VGraph.parse("0➔1➔2")

        Split the branch ``0`` in its middle adding the node ``3`` in the graph.

        >>> g1 = graph.split_branch(0, 0.5)
        >>> g1.branch_list.tolist()
        [[0, 3], [1, 2], [3, 1]]

        Split the branch ``1`` of the original graph at its thirds:

        >>> g2, branch_ids, new_nodes = graph.split_branch(1, [1/3, 2/3], return_branch_ids=True, return_node_ids=True)
        >>> branch_ids
        array([1, 2, 3])
        >>> new_nodes
        array([3, 4])
        >>> g2.branch_list.tolist()
        [[0, 1], [1, 3], [3, 4], [4, 2]]

        """  # noqa: E501
        graph = self if inplace else self.copy()

        # === Check coordinates or index of the splits ===
        split_curve_id = np.atleast_1d(split_curve_id)
        assert split_curve_id.ndim == 1, "split_curve_id must be a 1D array."
        assert split_curve_id.shape[0] > 0, "split_curve_id must contain at least one split point."
        n_split = len(split_curve_id)

        if split_coord is not None:
            split_coord = np.atleast_2d(split_coord)
            assert split_coord.ndim == 2 and split_coord.shape[1] == 2, "split_coord must be a 2D array of shape (N, 2)"
            assert len(split_coord) == n_split, "split_coord and split_curve_id must have the same length."

        # === Compute index for the new nodes and branches ===
        _, nEnd = graph._branch_list[branchID]
        new_nodeIds = np.arange(n_split) + graph.node_count
        new_branchIds = np.arange(n_split) + graph.branch_count

        # === Split the branches in the geometric data ===
        for gdata in graph._geometric_data:
            gdata._split_branch(branchID, split_curve_id, split_coord, new_branchIds, new_nodeIds)

        # === Insert new nodes and branches ... ===
        # 1. ... in the branch list
        graph._branch_list[branchID, 1] = new_nodeIds[0]
        new_branches = []
        for nPrev, nNext in itertools.pairwise(np.concatenate([new_nodeIds, [nEnd]])):
            new_branches.append((nPrev, nNext))
        graph._branch_list = np.concatenate((graph._branch_list, new_branches), axis=0)
        # 2. ... in the branches attributes
        new_df = graph._branch_attr.reindex(pd.RangeIndex(len(graph._branch_list)), copy=False)
        new_df.iloc[new_branchIds] = new_df.iloc[branchID]
        graph._branch_attr = new_df

        # 3. ... in the nodes attributes
        graph._nodes_count += len(new_nodeIds)
        graph._node_attr = graph._node_attr.reindex(pd.RangeIndex(graph.node_count), copy=False)

        # === Update the nodes and branches references ===
        for node in graph._node_refs:
            if node._id == nEnd:
                node.clear_adjacent_branch_cache()
        for branch in graph._branch_refs:
            if branch._id == branchID:
                branch._nodes_id = graph._branch_list[branchID]

        out = [graph]
        if return_branch_ids:
            out.append(np.concatenate([[branchID], new_branchIds]))
        if return_node_ids:
            out.append(new_nodeIds)

        return out if len(out) > 1 else out[0]

    def bridge_nodes(self, node_pairs: npt.ArrayLike[int], *, fuse_nodes=False, check=True, inplace=False) -> VGraph:
        """Fuse pairs of endpoints nodes by linking them together.

        The nodes are removed from the graph and their corresponding branches are merged.

        Parameters
        ----------
        node_pairs : npt.NDArray
            Array of shape (P, 2) containing P pairs of indices of the nodes to link.

        fuse_nodes : bool, optional
            If True, the nodes are fused together instead of being linked by a new branch.

        check : bool, optional
            If True, check the pairs to ensure that the nodes are not already connected by a branch.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.

        Examples
        --------
        >>> graph = VGraph.parse("0➔2 ; 3➔1")
        >>> graph.branch_list.tolist()
        [[0, 2], [3, 1]]

        Bridge the nodes 2 and 3:
        >>> g1 = graph.bridge_nodes([2, 3])
        >>> g1.branch_list.tolist()
        [[0, 2], [3, 1], [2, 3]]

        Bridge the nodes 2 and 3 and fuse them:
        >>> g2 = graph.bridge_nodes([2, 3], fuse_nodes=True)
        >>> g2.branch_list.tolist()
        [[0, 1]]

        """  # noqa: E501
        graph = self.copy() if not inplace else self

        node_pairs = np.atleast_2d(node_pairs).astype(int)
        assert node_pairs.ndim == 2 and node_pairs.shape[1] == 2, "node_pairs must be a 2D array of shape (P, 2)."

        if check:
            N = self.node_count
            node_pairs = node_pairs[:, 0] + node_pairs[:, 1] * N
            node_pairs = np.unique(node_pairs)
            branch_list = graph._branch_list[:, 0] + graph._branch_list[:, 1] * N
            node_pairs = node_pairs[np.isin(node_pairs, branch_list, invert=True)]
            if len(node_pairs) == 0:
                return graph

            node_pairs = np.stack([node_pairs % N, node_pairs // N], axis=1)

        # === Insert new branches between the nodes ===
        graph.add_branch(node_pairs, inplace=True)

        # === Fuse the nodes together if needed ===
        if fuse_nodes:
            graph.fuse_node(node_pairs, inplace=True)
        else:
            # or update the nodes references
            updated_nodes = np.unique(node_pairs)
            for node_ref in graph._node_refs:
                if node_ref._id in updated_nodes:
                    node_ref.clear_adjacent_branch_cache()
        return graph

    # --- Nodes edition ---
    def delete_node(self, node_id: IndexLike, *, inplace=False) -> VGraph:
        """Remove the nodes with the given indices from the graph as well as their incident branches.

        Any nodes left without incident branches after this deletions are also removed from the graph.

        Parameters
        ----------
        node_id : npt.NDArray

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.

        See Also
        --------
        fuse_node : Fuse nodes connected to exactly two branches, removing the node and merging the branches.

        delete_branch : Remove the branches connected to the node and the node itself.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2 ; 1➔3 ; 2➔3")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [1, 3], [2, 3]]

        Remove the node ``3`` and its incident branches:

        >>> g1 = graph.delete_node(3)
        >>> g1.branch_list.tolist()
        [[0, 1], [1, 2]]

        Remove the node ``2`` and ``3``:

        >>> g2 = graph.delete_node([2, 3])
        >>> g2.branch_list.tolist()
        [[0, 1]]

        Remove the node ``0``, the subsequent node indices are shifted (i.e. ``1``➔``0``, ``2``➔``1``, ``3``➔``2``):

        >>> g2 = graph.delete_node(0)
        >>> g2.branch_list.tolist()
        [[0, 1], [0, 2], [1, 2]]

        Remove the node ``1``. Because the node ``0`` would be an orphan node, it is also removed. This leave a single branch connecting the nodes ``2`` and ``3`` (reindexed as ``0`` and ``1``):

        >>> g3 = graph.delete_node(1)
        >>> g3.branch_list.tolist()
        [[0, 1]]

        """  # noqa: E501
        incident_branches = self.adjacent_branches(node_id)
        return self.delete_branch(incident_branches, delete_orphan_nodes=True, inplace=inplace)

    def fuse_node(
        self,
        node_id: IndexLike,
        *,
        quietly_ignore_invalid_nodes=False,
        inplace=False,
        incident_branches: Optional[npt.ArrayLike[np.int_]] = None,
    ) -> VGraph:
        """Fuse nodes connected to exactly two branches.

        The nodes are removed from the graph and their corresponding branches are merged.

        Parameters
        ----------
        node_id : npt.NDArray
            Array of indices of the nodes to fuse.

        quietly_ignore_invalid_nodes : bool, optional
            Choose the behavior when a node from ``node_id`` is not connected to exactly two branches:
            - If True, simply ignore the invalid nodes;
            - If False (by default), raise an error;
            - If None, assumes that the provided nodes are connected to exactly two branches.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

            By default: False.

        incident_branches : npt.ArrayLike, optional
            The incident branches of each node as a 2D array of shape (N, 2).
            If None, the incident branches are inferred from the branches graph.

            .. warning::
                If ``incident_branches`` is provided, the validity of the nodes is not checked.

        Returns
        -------
        VGraph
            The modified graph.`

        See Also
        --------

        delete_node : Remove the nodes and their incident branches.

        merge_consecutive_branches : Merge consecutive branches in the graph.

        Examples
        --------
        >>> # Branch id:           0 1
        >>> graph = VGraph.parse("0➔2➔1")
        >>> graph.branch_list.tolist()
        [[0, 2], [2, 1]]

        Fuse the node ``2``:

        >>> g1 = graph.fuse_node(2)
        >>> g1.branch_list.tolist()
        [[0, 1]]

        Attempt to fuse the node ``0`` and ``2``:

        >>> g2 = graph.fuse_node([0, 2], quietly_ignore_invalid_nodes=True)
        >>> g2.branch_list.tolist()
        [[0, 1]]

        Fuse the node ``2`` by providing its incident branches:

        >>> g3 = graph.fuse_node(2, incident_branches=[[0, 1]])
        >>> g3.branch_list.tolist()
        [[0, 1]]

        Attempt to fuse an invalid node:

        >>> VGraph.parse("0➔1➔2 ; 1➔3").fuse_node(1)
        Traceback (most recent call last):
        ValueError: Node 1 is not connected to exactly two branches.
        """
        graph = self.copy() if not inplace else self
        node_id = graph.as_node_ids(node_id)
        graph._fuse_nodes(
            node_id, quietly_ignore_invalid_nodes=quietly_ignore_invalid_nodes, incident_branches=incident_branches
        )
        return graph

    def _fuse_nodes(
        self,
        node_id: npt.ArrayLike[int],
        *,
        quietly_ignore_invalid_nodes: Optional[bool] = False,
        incident_branches: Optional[npt.NDArray[np.int_]] = None,
    ) -> Tuple[List[npt.NDArray[np.int_]], npt.NDArray[np.bool_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Fuse nodes connected to exactly two branches. (See :meth:`fuse_nodes` for the public method)

        Parameters
        ----------
        node_id : npt.ArrayLike[int]
            Array of indices of N nodes to fuse.

        quietly_ignore_invalid_nodes : bool, optional
            Choose the behavior when a node from ``node_id`` is not connected to exactly two branches:
            - If True, simply ignore the invalid nodes;
            - If False (by default), raise an error;
            - If None, assumes that the provided nodes are connected to exactly two branches.

        incident_branches : npt.NDArray, optional
            The incident branches of each node as a 2D array of shape (N, 2).
            If None, the incident branches are inferred from the branches graph.

            .. warning::
                If ``incident_branches`` is provided, the validity of the nodes is not checked.

        Returns
        -------
        consecutive_branches : List[np.ndarray]
            The list of the N clusters of merged branches. The first branch now contains all the branches of the group. The index correspond the branches indices before the merge.

        main_branch_flipped : np.ndarray
            A boolean array of shape (N,) indicating if the main branch of each group was flipped.

        branch_merge_lookup : np.ndarray
            A lookup table to reindex the branches after the merge.

        branches_to_delete : np.ndarray
            The indices of the branches that were deleted. The index correspond the branches indices before the merge.

        """  # noqa: E501
        assert node_id.ndim == 1, "node_id must be a 1D array."
        if incident_branches is None:
            branch_pairs = self.adjacent_branches_per_node(node_id)
            if quietly_ignore_invalid_nodes is not None:
                ignore_nodes = []
                for i, (nodes_id, pair) in enumerate(zip(node_id, branch_pairs, strict=True)):
                    if len(pair) != 2:
                        if not quietly_ignore_invalid_nodes:
                            raise ValueError(f"Node {nodes_id} is not connected to exactly two branches.")
                        else:
                            ignore_nodes.append(i)
                if ignore_nodes:
                    node_id = np.delete(node_id, ignore_nodes)
                    for i in sorted(ignore_nodes, reverse=True):
                        del branch_pairs[i]
        else:
            branch_pairs = np.atleast_2d(incident_branches)
            assert (
                branch_pairs.ndim == 2 and branch_pairs.shape[1] == 2
            ), "incident_branches must be a 2D array of shape (N, 2)."
            assert branch_pairs.shape[0] == len(
                node_id
            ), "The number of incident branches must match the number of nodes."

        return self._merge_consecutive_branches(
            branch_pairs=branch_pairs, junction_nodes=node_id, remove_orphan_nodes=True
        )

    def merge_consecutive_branches(
        self,
        branch_pairs: npt.ArrayLike[int],
        junction_nodes: Optional[npt.ArrayLike[int]] = None,
        *,
        remove_orphan_nodes: bool = True,
        quietly_ignore_invalid_pairs: Optional[bool] = False,
        inplace=False,
    ) -> VGraph:
        """Merge consecutive branches in the graph.

        In opposition to :meth:`fuse_nodes`, this method can merge branches connected through node of degree higher than 2.

        Parameters
        ----------
        branch_pairs : npt.ArrayLike[int]
            An array of shape (N, 2) containing the pairs of consecutive branches to merge.

            .. note::
                To merge more than two consecutive branches together, simply provide the pairs of branches to merge in order: ``[[0, 1], [1, 2], [2, 3]]``...

        junction_nodes : npt.ArrayLike[int], optional
            The indices of the junction nodes connecting each pair of branches. If None, the junction nodes are inferred from the branches graph.

        remove_orphan_nodes : bool, optional
            If True, the junction nodes that are not connected to any branch after the deletion are removed from the graph.

        quietly_ignore_invalid_pairs : bool, optional
            If True, ignore any branch pairs not consecutive or connecting the same nodes or containing looping branches.
            If False, raise an error if such branches are found.
            If None, assumes that the provided branches are valid.

        inplace : bool, optional
            If True, the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.

        See Also
        --------
        fuse_nodes : Fuse nodes connected to exactly two branches, removing the node and merging the branches.

        delete_branch : Remove the branches connected to the node and the node itself.

        Examples
        --------
        >>> # Branch id:           0 1 2     3 4     5
        >>> graph = VGraph.parse("0➔1➔2➔3 ; 0➔3➔4 ; 5➔5")
        >>> graph.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3], [0, 3], [3, 4], [5, 5]]

        Merge the branches ``0`` and ``1``. The junction node ``1`` is removed (shifting the subsequent node indices):

        >>> g1 = graph.merge_consecutive_branches([0, 1])
        >>> g1.branch_list.tolist()
        [[0, 1], [1, 2], [0, 2], [2, 3], [4, 4]]

        Merge the branches ``0``, ``1``, and ``2``. Because they are consecutive, their merging results in a single branch:

        >>> g2 = graph.merge_consecutive_branches([[0, 1], [1, 2]])
        >>> g2.branch_list.tolist()
        [[0, 1], [0, 1], [1, 2], [3, 3]]

        Merge the branches ``3`` and ``4``. Note that, because the junction node ``3`` has a third adjacent branch, it is not removed:

        >>> g3 = graph.merge_consecutive_branches([3, 4])
        >>> g3.branch_list.tolist()
        [[0, 1], [1, 2], [2, 3], [0, 4], [5, 5]]

        """  # noqa: E501
        graph = self.copy() if not inplace else self

        branch_pairs = np.asarray(branch_pairs, dtype=int).reshape(-1, 2)

        if junction_nodes is None:
            junction_nodes = []
            ignore_pairs = []
            for i, pair in enumerate(branch_pairs):
                branch_nodes = graph._branch_list[pair]
                nodes, count = np.unique(branch_nodes, return_counts=True)
                junction = nodes[count == 2]
                if any(branch_nodes[:, 0] == branch_nodes[:, 1]) or junction.size != 1:
                    ignore_pairs.append(i)
                else:
                    junction_nodes.append(junction[0])

            if ignore_pairs:
                if quietly_ignore_invalid_pairs:
                    branch_pairs = np.delete(branch_pairs, ignore_pairs)
                else:
                    raise ValueError(f"Invalid consecutive branches: {branch_pairs[ignore_pairs].tolist()}.")
        else:
            # Check that the junction nodes are valid
            junction_nodes = np.asarray(junction_nodes, dtype=int)
            if quietly_ignore_invalid_pairs is not None:
                assert len(junction_nodes) == len(branch_pairs), (
                    "The number of junction nodes must match the number of branch pairs. \n"
                    f"({len(junction_nodes)} junction nodes where provided and {len(branch_pairs)} branch pairs)"
                )

                invalid_pairs = [
                    i
                    for i, (node, b_pair) in enumerate(zip(junction_nodes, branch_pairs, strict=True))
                    if node not in graph._branch_list[b_pair[0]] or node not in graph._branch_list[b_pair[1]]
                ]
                if len(invalid_pairs):
                    if quietly_ignore_invalid_pairs:
                        branch_pairs = np.delete(branch_pairs, invalid_pairs)
                        junction_nodes = np.delete(junction_nodes, invalid_pairs)
                    else:
                        raise ValueError(f"Invalid consecutive branches: {branch_pairs[invalid_pairs].tolist()}.")

        graph._merge_consecutive_branches(branch_pairs, junction_nodes, remove_orphan_nodes)
        return graph

    def _merge_consecutive_branches(
        self, branch_pairs: npt.ArrayLike[int], junction_nodes: npt.ArrayLike[int], remove_orphan_nodes: bool = True
    ) -> Tuple[List[npt.NDArray[np.int_]], npt.NDArray[np.bool_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Merge consecutive branches in the graph. (See :meth:`merge_consecutive_branches` for the public method)

        .. warning::
            This method assumes that the branches are consecutive and that the junction nodes are correctly provided.
            Attempting to merge non-consecutive branches will result in unknown behaviors.

        Parameters
        ----------
        branches_pairs : npt.ArrayLike[int]
            The pairs of consecutive branches to merge.

        junction_nodes : npt.ArrayLike[int]
            The indices of the junction nodes connecting each pair of branches.

        remove_orphan_nodes : bool, optional
            If True, the nodes that are not connected to any branch after the deletion are removed from the graph.

        Returns
        -------
        consecutive_branches : List[npt.NDArray]
            The list of the N clusters of merged branches. The first branch now contains all the branches of the group. The index correspond the branches indices before the merge.

        main_branch_flipped : npt.NDArray
            A boolean array of shape (N,) indicating if the main branch of each group was flipped.

        branch_merge_lookup : npt.NDArray
            A lookup table to reindex the branches after the merge.

        branches_to_delete : npt.NDArray
            The indices of the branches that were deleted. The index correspond the branches indices before the merge.
        """  # noqa: E501

        # 1. Cluster consecutive branches
        junction_nodes = np.asarray(junction_nodes, dtype=int)
        branch_pairs = np.asarray(branch_pairs, dtype=int).reshape(-1, 2)

        consecutive_branches, consecutive_pair_ids = reduce_chains(branch_pairs.tolist(), return_index=True)
        consecutive_branches = [np.array(c, dtype=int) for c in consecutive_branches]
        consecutive_pair_ids = [np.array(c, dtype=int) for c in consecutive_pair_ids]

        # 2. For each group of consecutive branches flip them if needed and merge them
        branches_to_delete = []
        main_branch_flipped = np.empty(len(consecutive_branches), dtype=bool)
        merged_branches_summary = {}
        for i, (branches, pairs_id) in enumerate(zip(consecutive_branches, consecutive_pair_ids, strict=True)):
            junctions = junction_nodes[abs(pairs_id) - 1]
            b0, b_last = branches[[0, -1]]

            b0_flipped = self._branch_list[b0, 1] != junctions[0]
            flip_branches = np.concatenate([[b0_flipped], self._branch_list[branches[1:], 0] != junctions])
            main_branch_flipped[i] = b0_flipped

            # Update the nodes of the first branch with the non-junction node of the cluster's first and last branches
            self._branch_list[b0, 0] = self._branch_list[b0, 1 if b0_flipped else 0]
            self._branch_list[b0, 1] = self._branch_list[b_last, 0 if flip_branches[-1] else 1]

            # Remember the merged branches
            for b in branches[1:]:
                merged_branches_summary[b] = b0

            # Flip and merge the branches
            branches_to_delete.extend(branches[1:])
            for gdata in self._geometric_data:
                if flip_branches.any():
                    gdata._flip_branch_direction(branches[flip_branches])
                gdata._merge_branches(branches)

        branches_to_delete = np.array(branches_to_delete, dtype=int)

        # 3a. Update the branch references
        if self._branch_refs:
            for branch in self._branch_refs:
                if (redirected_to := merged_branches_summary.get(branch._id, -1)) != -1:
                    # Redirect the merged branches to the first ones
                    branch._id = redirected_to
                    branch._nodes_id = self._branch_list[branch._id]

        # 3b. Update the nodes references of the reconnected nodes
        if self._node_refs:
            reconnected_nodes = np.unique(self._branch_list[[c[0] for c in consecutive_branches]])
            for node in self._node_refs:
                if node._id in reconnected_nodes:
                    node.clear_incident_branch_cache()

        # 3c. Remove the branches and create the merge lookup
        branch_merge_lookup = self._delete_branch(branches_to_delete, update_refs=False)
        for c in consecutive_branches:
            branch_merge_lookup[c[1:]] = branch_merge_lookup[c[0]]

        # 4. Remove orphan nodes
        if remove_orphan_nodes:
            orphans = junction_nodes[~np.isin(junction_nodes, self._branch_list)]
            self._delete_node(orphans)

        return (consecutive_branches, main_branch_flipped, branch_merge_lookup, branches_to_delete)

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
            The indices of the nodes to merge.

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
        nodes_lookup = np.arange(self.node_count, dtype=int)

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
        if self._node_refs:
            resulting_nodes = np.unique(resulting_nodes)
            nodes_lookup = add_empty_to_lookup(nodes_lookup, increment_index=False)
            for node in self._node_refs:
                node._id = nodes_lookup[node._id + 1]
                if node._id in resulting_nodes:
                    node.clear_incident_branch_cache()

        # 5. Remove branches and nodes
        if len(branches_to_remove) > 0:
            branches_to_remove = np.unique(np.asarray(branches_to_remove, dtype=int))
            self._delete_branch(branches_to_remove)

        if len(nodes_to_remove) > 0:
            nodes_to_remove = np.unique(np.concatenate(nodes_to_remove, dtype=int))
            self._delete_node(nodes_to_remove, update_refs=False)

        # 6. Update branches references
        if self._branch_refs and updated_branches:
            updated_branches = np.unique(np.concatenate(updated_branches, dtype=int))
            for branch in self._branch_refs:
                if branch._id in updated_branches:
                    branch._nodes_id = self._branch_list[branch._id]

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
        assert 0 <= node_id < self.node_count, f"Node index {node_id} is out of range."
        return self.__class__.NODE_ACCESSOR_TYPE(self, node_id)

    def nodes(
        self,
        ids: IndexLike = None,
        /,
        *,
        only_degree: Optional[int | npt.ArrayLike[np.int_]] = None,
        dynamic_iterator: bool = False,
    ) -> Generator[VGraphNode]:
        """Iterate over the nodes of a graph, encapsulated in :class:`VGraphNode` objects.

        Parameters
        ----------
        ids : int or npt.ArrayLike[int], optional
            The indices of the nodes to iterate over. If None, iterate over all nodes.

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
        nodes_ids = self.as_node_ids(ids)
        if only_degree is not None:
            only_degree = np.asarray(only_degree, dtype=int)
            node_degree = self.node_degree()
            nodes_ids = np.argwhere(np.isin(node_degree, only_degree)).flatten()

        if dynamic_iterator:
            nodes = [self.__class__.NODE_ACCESSOR_TYPE(self, int(i)) for i in nodes_ids]
            while len(nodes) > 0:
                node = nodes.pop(0)
                if node.is_valid():
                    yield node
        else:
            for i in nodes_ids:
                yield self.__class__.NODE_ACCESSOR_TYPE(self, int(i))

    def walk_nodes(
        self, root_id: IndexLike, /, depth: int = 1, traversal: Literal["bfs", "dfs"] = "bfs"
    ) -> Generator[VGraphNode]:
        """Walk the graph from a given node and yield the indices of the nodes.

        Parameters
        ----------
        root : int
            The index of the starting node.

        depth : int, optional
            The maximum depth of the walk.

        traversal : str, optional
            The traversal strategy:

            - 'bfs': Breadth-first search.
            - 'dfs': Depth-first search.

        Returns
        -------
        Generator[VGraphNode]
            A generator that yields the indices of the nodes.

        Examples
        --------
        >>> graph = VGraph.parse("0➔1➔2➔3; 1➔4; 2➔5;")
        >>> [node.id for node in graph.walk_nodes(0, traversal="bfs")]
        [0, 1, 2, 4, 3, 5]
        >>> [node.id for node in graph.walk_nodes(0, traversal="dfs")]
        [0, 1, 2, 3, 5, 4]
        """
        stack = self.as_node_ids(root_id).astype(int).flatten()
        stack = stack.tolist()
        visited_nodes = np.zeros(self.node_count, dtype=bool)

        depth_first = traversal == "dfs"

        while stack:
            node_id = stack.pop() if depth_first else stack.pop(0)
            yield self.node(node_id)
            visited_nodes[node_id] = True

            successors = self.adjacent_nodes(node_id)
            successors = successors[~visited_nodes[successors]]
            if depth_first:
                successors = successors[::-1]
            stack.extend(successors)

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
        assert 0 <= branch_id < self.branch_count, f"Branch index {branch_id} is out of range."
        return self.__class__.BRANCH_ACCESSOR_TYPE(self, branch_id)

    def branches(
        self,
        ids: IndexLike = None,
        /,
        *,
        filter: Optional[Literal["orphan", "endpoint", "non-endpoint"]] = None,
        dynamic_iterator: bool = False,
    ) -> Generator[VGraphBranch]:
        """Iterate over the branches of a graph, encapsulated in :class:`VGraphBranch` objects.

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
            If True, iterate over all the branches present in the graph when this method is called.
            All branches added during the iteration will be ignored. Branches reindexed during the iteration will be visited in their original order at the time of the call. Deleted branches will not be visited.

            Enable this option if you plan to modify the graph during the iteration. If you only plan to read the graph, disable this option for better performance.

        Returns
        -------
        Generator[VVGraphBranch]
            A generator that yields branches.
        """  # noqa: E501
        branches_ids = self.as_branch_ids(ids)
        if filter == "orphan":
            branches_ids = np.intersect1d(branches_ids, self.orphan_branches())
        elif filter == "endpoint":
            branches_ids = np.intersect1d(branches_ids, self.endpoint_branches())
        elif filter == "non-endpoint":
            branches_ids = np.setdiff1d(branches_ids, self.endpoint_branches())

        if dynamic_iterator:
            branches = [self.__class__.BRANCH_ACCESSOR_TYPE(self, int(i)) for i in branches_ids]
            while len(branches) > 0:
                branch = branches.pop(0)
                if branch.is_valid():
                    yield branch
        else:
            for i in branches_ids:
                yield self.__class__.BRANCH_ACCESSOR_TYPE(self, int(i))

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
        boundaries_only_tip=False,
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

        geodata = self.geometric_data()
        domain = geodata.domain

        layer = LayerGraph(
            self._branch_list,
            geodata.node_coord() - np.array(domain.top_left)[None, :],
            geodata.branch_label_map(calibre_attr=boundaries, only_tip=boundaries_only_tip),
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

        if self.branch_count > 0:
            if bspline is not None:
                if bspline is True:
                    bspline = VBranchGeoData.Fields.BSPLINE
                branch_bspline = geodata.branch_data(bspline)
                node_coord = geodata.node_coord()

                bsplines_path = []
                filler_paths = []
                for i, d in enumerate(branch_bspline):
                    n1, n2 = [Point(*node_coord[_]) for _ in self._branch_list[i]]
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
                _: "#444" for _ in range(max_colored_node_id + 1, self.node_count + 1)
            }
            layer.nodes_cmap = node_cmap
            incidents_branches = np.setdiff1d(
                np.arange(self.branch_count),
                self.adjacent_branches(np.arange(max_colored_node_id + 1)),
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
        #
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
