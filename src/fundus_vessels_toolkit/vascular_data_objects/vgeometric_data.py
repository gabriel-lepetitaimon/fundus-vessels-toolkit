from __future__ import annotations

import itertools
import warnings
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Tuple, overload
from weakref import ref

import numpy as np
import numpy.typing as npt

from fundus_toolkits import FundusData
from fundus_toolkits.transform import Transform, Translation
from fundus_toolkits.utils.geometric import Point, Rect
from fundus_toolkits.utils.typing import (
    Bool1DArrayLike,
    Float1DArray,
    Float2DArray,
    FloatPairArray,
    Indices,
    IndicesLike,
    Int1DArray,
    Int1DArrayLike,
    Int2DArray,
    PointArray,
    PointArrayLike,
    as_float_pairs,
)

from ..utils import if_none
from ..utils.bezier import BSpline
from ..utils.cluster import remove_consecutive_duplicates
from ..utils.cpp_optimized import discontiguous_index
from ..utils.data_io import NumpyDict, load_numpy_dict, save_numpy_dict
from ..utils.lookup_array import invert_lookup, reorder_array
from ..utils.math import nearest_point_on_segment
from ..utils.numpy import array_is_equal, array_list_is_equal, as_1d_array, np_find_sorted, readonly
from ..utils.rasterization import rasterize_line
from .vbranch_geodata import (
    BranchGeoDataEditContext,
    T_VBranchGeoData,
    VBranchGeoData,
    VBranchGeoDataBase,
    VBranchGeoDataKey,
    VBranchGeoDataLike,
    VBranchGeoDescriptor,
    VBranchGeoDict,
    VBranchGeoDictUncurated,
)

if TYPE_CHECKING:
    from .vgraph import NodeIndices, VGraph

    # T_VBranchGeoData = TypeVar("T_VBranchGeoData", bound=VBranchGeoData)

EMPTY_CURVE = readonly(np.empty((0, 2), dtype=np.int_))
INTEGRITY_CHECK: Literal[False, "raise", "warn"] = False


class VGeometricData:
    """``VGeometricData`` is a class that stores the geometric data of a vascular graph."""

    def __init__(
        self,
        nodes_coord: Float2DArray,
        branches_curve: List[npt.NDArray[np.uint32]],
        domain: Rect,
        branches_attr: Optional[VBranchGeoDictUncurated] = None,
        nodes_id: Optional[npt.NDArray[np.uint32]] = None,
        branches_id: Optional[npt.NDArray[np.uint32]] = None,
        parent_graph: Optional[VGraph] = None,
        fundus_data: Optional[FundusData] = None,
    ):
        """Create a VGeometricData object.

        Parameters
        ----------
        image_shape : Point
            The shape (height, width) in pixel of the image from which was extracted the geometric data.

        node_coord : npt.NDArray[np.uint32] of shape (n_nodes, 2)
            The coordinates (y, x) of the nodes in the graph.

        branch_curves_yx : List[npt.NDArray[np.uint32]]
            The coordinates of the pixels that compose the branches of the graph. The list is of length n_branches and all of its elements are 2D array with shape (n_points, 2) storing the (y, x) coordinates of the pixels. The number of points of each branch can be different.

        branch_curves_attr : Optional[BranchesCurvesAttributes], optional
            The attributes of the branches, by default None. The keys are the attribute names and the values are a dictionary where the keys are the branch ids and the values are the attribute values.

        nodes_id : Optional[npt.NDArray[np.uint32]], optional
            The ids of the nodes in the graph, by default None. If None, the index of the nodes are assumed to be all integer values from 0 to n_nodes - 1.

        branches_id : Optional[npt.NDArray[np.uint32]], optional
            The ids of the branches in the graph, by default None. If None, the index of the branches are assumed to be all integer values from 0 to n_branches - 1.

        Raises
        ------
        ValueError
            If the length of the curves attributes is not the same as the length of the branch curve.
        """  # noqa: E501
        # Define domain
        self._parent_graph: Optional[ref[VGraph]] = None
        self.parent_graph = parent_graph
        self.fundus_data = fundus_data.to_immutable() if fundus_data is not None else None
        self._domain: Rect = Rect.from_tuple(domain)

        # Check and define nodes coordinates and index
        nodes_coord = np.asarray(nodes_coord, dtype=np.float64)
        if not nodes_coord.ndim == 2 and nodes_coord.shape[1] == 2:
            raise ValueError("The node coordinates should be a 2D array with shape (n_nodes, 2).")
        self._nodes_coord: npt.NDArray[np.float64] = nodes_coord.astype(np.float64)

        if nodes_id is not None:
            nodes_id = np.asarray(nodes_id, dtype=np.uint32)
            if not (nodes_id.ndim == 1 and nodes_id.shape[0] == nodes_coord.shape[0]):
                raise ValueError(
                    "The node ids should be a 1D array with the same length as the number of nodes. "
                    f"Got shape {nodes_id.shape} instead of {nodes_coord.shape}."
                )
        self._nodes_id = nodes_id

        # Check and define branches curves and index
        if not all(isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] == 2 for c in branches_curve):
            raise ValueError("The branch curves should be a list of 2D arrays with shape (n_points, 2).")
        self._branch_curve: List[npt.NDArray[np.int_]] = [readonly(c) for c in branches_curve]

        if branches_id is not None:
            branches_id = np.asarray(branches_id, dtype=np.uint32)
            assert branches_id.ndim == 1 and branches_id.shape[0] == len(branches_curve), (
                "The branch ids should be a 1D array with the same length as the number of branches. "
                f"Got shape {branches_id.shape} instead of {len(branches_curve)}."
            )
        self._branches_id = branches_id

        # Check and define branches curves attributes
        self._branch_data_dict: VBranchGeoDict = {}
        self._branches_attrs_descriptors: Dict[str, VBranchGeoDescriptor] = {}

        if branches_attr is not None:
            self._branch_data_dict, self._branches_attrs_descriptors = VBranchGeoData.parse_geo_dict(
                branches_attr, self
            )

        # Ensure that the nodes and branches are sorted by their index in the graph
        self._sort_internal_node_ids()
        self._sort_internal_branch_ids()

        if INTEGRITY_CHECK:
            self.check_integrity()

    @classmethod
    def from_dict(
        cls,
        nodes_coord: Dict[int, Point],
        branches_curve: Dict[int, npt.NDArray[np.uint32]],
        domain: Rect,
        branches_attr: Optional[Dict[str, Dict[int, VBranchGeoDataLike]]] = None,
        parent_graph=None,
    ) -> VGeometricData:
        """Create a VGeometricData object from dictionaries.

        Parameters
        ----------
        nodes_coord : Dict[int, Point]
            The coordinates of the nodes in the graph. The keys are the node ids and the values are the coordinates.

        branches_curve : Dict[int, np.ndarray]
            The coordinates of the pixels that compose the branches of the graph. The keys are the branch ids and the values are 2D array with shape (n_points, 2) storing the (y, x) coordinates of the pixels. The number of points of each branch can be different.

        domain : Rect
            The domain of the geometric data.

        branches_attr : Optional[Dict[str, Dict[int, np.ndarray | VBranchGeoAttr]]], optional
            The attributes of the branches, by default None. The keys are the attribute names and the values are a dictionary where the keys are the branch ids and the values are the attribute values.

        Returns
        -------
        VGeometricData
            The geometric data object.
        """  # noqa: E501
        nodes_id = np.array(list(nodes_coord.keys()), dtype=np.uint32)
        nodes_coord_array = np.array(list(nodes_coord.values()), dtype=np.float32)
        branches_id = np.array(list(branches_curve.keys()), dtype=np.uint32)
        branches_curve_array = list(branches_curve.values())

        if branches_attr is not None:
            branch_attr_array = {
                attr_name: [attr.get(i, None) for i in branches_id] for attr_name, attr in branches_attr.items()
            }
        else:
            branch_attr_array = None

        return cls(
            nodes_coord_array, branches_curve_array, domain, branch_attr_array, nodes_id, branches_id, parent_graph
        )

    def save(self, filename: Optional[str | Path] = None) -> NumpyDict:
        """Save the geometric data to a numpy dictionary.

        The geometric data is saved as a dictionary with the following keys:
        - ``nodes_coord``: The coordinates of the nodes.
        - ``nodes_id``: The ids of the nodes.
        - ``branches_curve``: The coordinates of the pixels that compose the branches.
        - ``branches_id``: The ids of the branches.
        - ``domain``: The domain of the geometric data: (height, width, top, left).
        - ``branches_attr``: The attributes of the branches.


        Parameters
        ----------
        filename : Optional[str | Path], optional
            The path to the file where to save the data. If None, the data is not saved to a file.

        Returns
        -------
        NumpyDict
            The geometric data as a dictionary of numpy arrays.
        """
        data: NumpyDict = dict(
            nodes_coord=self._nodes_coord,
            nodes_id=self._nodes_id,
            branches_curve=self._branch_curve,
            branches_id=self._branches_id,
            domain=np.asarray(self._domain),
            branches_attr=VBranchGeoData.save(self._branch_data_dict),
        )  # type: ignore

        if filename is not None:
            save_numpy_dict(data, filename)
        return data

    @classmethod
    def load(cls, filename: str | Path | NumpyDict, parent_graph=None, fundus_data=None) -> VGeometricData:
        """Load the geometric data from a numpy dictionary.
        Parameters
        ----------
        filename : str | Path | NumpyDict
            The path to the file where to save the data or a dictionary generated by :meth:`save`.

        Returns
        -------
        VGeometricData
            The geometric data object.
        """
        if isinstance(filename, (str, Path)):
            data = load_numpy_dict(filename)
        else:
            data = filename

        return cls(
            nodes_coord=data["nodes_coord"],
            nodes_id=data["nodes_id"],
            branches_curve=data["branches_curve"],
            branches_id=data["branches_id"],
            domain=Rect(*data["domain"]),
            branches_attr=VBranchGeoData.load(data["branches_attr"]),
            parent_graph=parent_graph,
            fundus_data=fundus_data,
        )

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if (
            not isinstance(other, VGeometricData)
            or len(self._branch_curve) != len(other._branch_curve)
            or self._domain != other._domain
            or not array_is_equal(self._nodes_coord, other._nodes_coord)
            or not array_is_equal(self._nodes_id, other._nodes_id)
            or not array_is_equal(self._branches_id, other._branches_id)
            or not array_list_is_equal(self._branch_curve, other._branch_curve)
        ):
            return False

        if set(self._branch_data_dict.keys()) != set(other._branch_data_dict.keys()):
            return False
        for key in self._branch_data_dict.keys():
            data1 = self._branch_data_dict[key]
            data2 = other._branch_data_dict[key]
            if not data1 == data2:
                return False
        return True

    @classmethod
    def empty(cls, parent_graph: Optional[VGraph] = None) -> VGeometricData:
        """Create an empty geometric data object."""
        return cls(
            nodes_coord=np.empty((0, 2), dtype=np.float32),
            branches_curve=[],
            domain=Rect(0, 0, 0, 0),
            parent_graph=parent_graph,
        )

    @classmethod
    def empty_like(
        cls, other: VGeometricData, parent_graph: Optional[VGraph | Literal["other"]] = "other"
    ) -> VGeometricData:
        """Create an empty geometric data object with the same domain and nodes as another geometric data object."""
        return cls(
            nodes_coord=np.empty((0, 2), dtype=np.float32),
            branches_curve=[],
            domain=other.domain,
            parent_graph=other.parent_graph if parent_graph == "other" else parent_graph,
        )

    def copy(self, parent_graph: Optional[VGraph | Literal["same"]] = "same") -> VGeometricData:
        """Return a copy of the object."""
        other = copy(self)
        other._nodes_coord = self._nodes_coord.copy()
        other._branch_curve = copy(self._branch_curve)
        other._branch_data_dict = {k: copy(v) for k, v in self._branch_data_dict.items()}
        other._branches_attrs_descriptors = {k: copy(v) for k, v in self._branches_attrs_descriptors.items()}
        other._branches_id = self._branches_id.copy() if self._branches_id is not None else None
        if parent_graph == "same":
            other._parent_graph = ref(self._parent_graph()) if self._parent_graph is not None else None
        else:
            other._parent_graph = ref(parent_graph) if parent_graph is not None else None
        return other

    def __getstate__(self) -> Dict[str, Any]:
        # Swap the weakref to a regular reference for serialization
        state = self.__dict__.copy()
        parent_graph = state.pop("_parent_graph", None)
        state["parent_graph"] = parent_graph() if parent_graph is not None else None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # Swap the regular reference to a weakref after deserialization
        parent_graph = state.pop("parent_graph", None)
        state["_parent_graph"] = ref(parent_graph) if parent_graph is not None else None
        self.__dict__.update(state)

    ####################################################################################################################
    #  === BASE PROPERTIES ===
    ####################################################################################################################
    @property
    def parent_graph(self) -> VGraph:
        """Return the parent graph of the geometric data."""
        if self._parent_graph is None:
            raise AttributeError("The parent graph of this geometric data is not set.")
        return self._parent_graph()  # type: ignore

    @parent_graph.setter
    def parent_graph(self, graph: VGraph | None) -> None:
        """Set the parent graph of the geometric data."""
        if getattr(self, "_parent_graph", None) is not None:
            raise AttributeError("The parent graph of this geometric data is already set.")
        self._parent_graph = ref(graph) if graph is not None else None

    @property
    def domain(self) -> Rect:
        """Return the domain of the geometric data."""
        return self._domain

    @property
    def node_count(self) -> int:
        """Return the number of nodes whose geometric data are stored in this object."""
        return self._nodes_coord.shape[0]

    @property
    def node_ids(self) -> np.ndarray:
        """Return the indices of the nodes as indexed in the graph.
        In other word, a mapping of nodes index from they index in this VGeometricData object to their index in the original graph."""  # noqa: E501
        return np.arange(self.node_count) if self._nodes_id is None else self._nodes_id

    def _graph_to_internal_nodes_index(
        self, index: npt.ArrayLike, graph_index=True, *, index_sorted=False, sort_index=False, check_valid=False
    ) -> npt.NDArray[np.int_]:
        """Convert the index of the nodes in the graph to their index in the internal representation."""
        index = np.array(index, copy=True, dtype=np.int32)
        if not index_sorted and sort_index:
            index.sort()
            index_sorted = True
        if graph_index and self._nodes_id is not None:
            internal_index = np_find_sorted(index, self._nodes_id, assume_keys_sorted=index_sorted)
            assert check_valid or np.all(internal_index >= 0), f"Nodes {index[internal_index < 0]} not found."
            return internal_index
        else:
            assert check_valid or np.all([0 <= index, index < self.node_count]), f"Invalid node index {index}."
            return index

    def node_coord(
        self, ids: Optional[int | npt.NDArray[np.int_]] = None, *, graph_index=True, apply_domain=False
    ) -> npt.NDArray[np.float64]:
        """Return the coordinates of the nodes in the graph."""

        if ids is None:
            return self._nodes_coord

        ids, is_single = as_1d_array(ids)

        if isinstance(ids, np.ndarray):
            internal_id = self._graph_to_internal_nodes_index(ids, graph_index=graph_index)
            coord: npt.NDArray[np.float64] = self._nodes_coord[internal_id]
            if apply_domain:
                coord += np.array(self.domain.top_left)[None, :]
            return coord if not is_single else coord[0]
        else:
            raise TypeError("Invalid type for node id.")

    def set_node_coord(
        self, coord: npt.ArrayLike, ids: Optional[int | npt.NDArray[np.int32]], *, graph_index=True
    ) -> None:
        """Set the coordinates of the nodes in the graph."""
        coord = np.atleast_2d(coord)
        assert coord.ndim == 2, "The coordinates should be a 2D array."
        if ids is None:
            if graph_index is False:
                assert coord.shape[0] == self.node_count, (
                    "The number of coordinates should be the same as the number of nodes."
                )
                self._nodes_coord = np.asarray(coord, dtype=np.float64)
                return
            else:
                ids = np.arange(coord.shape[0])
        else:
            ids, _ = as_1d_array(ids)
            assert coord.shape[0] == len(ids), "The number of coordinates should be the same as the number of nodes."

        internal_id = self._graph_to_internal_nodes_index(ids, graph_index=graph_index)
        self._nodes_coord[internal_id] = coord

    @property
    def branch_count(self) -> int:
        """Return the number of branches whose geometric data are stored in this object."""
        return len(self._branch_curve)

    @property
    def branch_ids(self) -> np.ndarray:
        """Return the indices of the branches as indexed in the graph.
        In other word, a mapping of branches index from they index in this VGeometricData object to their index in the original graph."""  # noqa: E501
        return np.arange(self.branch_count) if self._branches_id is None else self._branches_id

    def _graph_to_internal_branch_ids(
        self,
        index: Optional[npt.ArrayLike],
        *,
        graph_index=True,
        index_sorted=False,
        sort_index=False,
        check_valid=True,
    ) -> npt.NDArray[np.int_]:
        """Convert the index of the branches in the graph to their index in the internal representation."""
        if index is None:
            return np.arange(self.branch_count)
        index = np.asarray(index, dtype=np.int32)
        if not index_sorted and sort_index:
            index = np.sort(index)
            index_sorted = True
        if graph_index and self._branches_id is not None:
            internal_index = np_find_sorted(index, self._branches_id, assume_keys_sorted=index_sorted)
            assert not check_valid or np.all(internal_index >= 0), f"Branch {index[internal_index < 0]} not found."
            return internal_index
        else:
            assert not check_valid or np.all([0 <= index, index < self.branch_count]), f"Invalid branch index {index}."
            return index

    @overload
    def has_branch_curve(self, ids: int, *, graph_index=True) -> bool: ...
    @overload
    def has_branch_curve(
        self, ids: Optional[npt.NDArray[np.int32]] = None, *, graph_index=True
    ) -> npt.NDArray[np.bool_]: ...
    def has_branch_curve(
        self, ids: Optional[int | npt.NDArray[np.int32]] = None, *, graph_index=True
    ) -> bool | npt.NDArray[np.bool_]:
        """Return whether the branches have a curve defined.

        Parameters
        ----------
        ids : Optional[int | np.ndarray], optional
            The id of the branch(es). If None (by default), the function returns whether all branches have a curve defined.

        Returns
        -------
        bool | np.ndarray
        """  # noqa: E501
        if ids is None:
            return np.array([c is not None and len(c) > 0 for c in self._branch_curve], dtype=bool)

        ids, is_single = as_1d_array(ids)
        internal_id = self._graph_to_internal_branch_ids(ids, graph_index=graph_index, check_valid=False)
        if is_single:
            return (c := self._branch_curve[int(internal_id)]) is not None and len(c) > 0
        else:
            return np.array(
                [(c := self._branch_curve[int(i)]) is not None and len(c) > 0 for i in internal_id], dtype=bool
            )

    @overload
    def branch_curve(
        self,
        ids: int,
        *,
        fill_with_nodes=False,
        min_length=1,
    ) -> PointArray: ...
    @overload
    def branch_curve(
        self,
        ids: Optional[Int1DArrayLike] = None,
        *,
        fill_with_nodes=False,
        min_length=1,
    ) -> list[PointArray]: ...
    def branch_curve(
        self,
        ids: Optional[int | Int1DArrayLike] = None,
        *,
        fill_with_nodes=False,
        min_length=1,
    ) -> PointArray | list[PointArray]:
        """Return the coordinates of the pixels that compose the branches of the graph.

        Parameters
        ----------
        ids : Optional[int | np.ndarray], optional
            The id of the branch(es) whose curve is requested. If None (by default), the function returns the curves of all branches.

        fill_with_nodes : bool, optional
            If True, the curve is filled with the coordinates of the nodes of the branch if the curve is not available. By default False.

        min_length : int, optional
            The minimum length of the curve to be considered as valid. If the curve is shorter than this length, it is considered as not available and the function returns an empty array or the nodes coordinates if ``fill_with_nodes`` is True.

        Returns
        -------
        np.ndarray | list[np.ndarray]

            - If ``ids`` is a scalar: a 2D array of shape (n_points, 2) containing the coordinates of the pixels defining the skeleton of the requested branch.
            - If ``ids`` is an iterable of int: a list of such arrays.
        """  # noqa: E501

        def from_node(b_id: int) -> PointArray:
            nodes = self.parent_graph.branch_list[b_id]
            return as_float_pairs(rasterize_line(*self.node_coord(nodes).astype(np.int_)))

        if ids is None:
            ids = np.arange(self.branch_count)
            is_single = False
        else:
            ids, is_single = as_1d_array(ids)

        if isinstance(ids, np.ndarray):
            curves = [as_float_pairs(if_none(self._branch_curve[i], EMPTY_CURVE)) for i in ids]
            if fill_with_nodes:
                curves = [from_node(i) if len(c) < min_length else c for i, c in zip(ids, curves, strict=True)]
            return curves[0] if is_single else curves

        else:
            raise TypeError("Invalid type for branches index.")

    @overload
    def branch_midpoint(self, ids: int, pos: float = 0.5, *, graph_index=True) -> Point | None: ...
    @overload
    def branch_midpoint(
        self, ids: Optional[npt.NDArray[np.int32]] = None, pos: float = 0.5, *, graph_index=True
    ) -> list[Point | None]: ...
    def branch_midpoint(
        self,
        ids: Optional[int | npt.NDArray[np.int32]] = None,
        pos: float = 0.5,
        *,
        infer_from_nodes=True,
        graph_index=True,
    ) -> Point | None | list[Point | None]:
        """Return the coordinates of a point from each branch skeleton.

        Parameters
        ----------
        ids : Optional[int | np.ndarray], optional
            The id of the branch(es), by default None

        p : float, optional
            The position of the midpoint on the branch. If p=0.5, the midpoint is the middle of the branch. If p=0, the midpoint is the first point of the branch. If p=1, the midpoint is the last point of the branch.

        infer_from_nodes : bool, optional
            If True, the midpoint is inferred from the nodes of the branch if the branch curve is not available. By default True.

        Returns
        -------
        np.ndarray | List[np.ndarray]
            - If ``ids`` is a scalar: a 1D array of shape (2,) containing the coordinate of the midpoint of the requested branch.
            - If ``ids`` is an iterable of int: a list of such arrays.
        """  # noqa: E501

        def midpoint(internal_id: int) -> Point | None:
            if internal_id < 0:
                return None
            curve = self._branch_curve[internal_id]
            if curve is None or curve.shape[0] == 0:
                if not infer_from_nodes or self._parent_graph is None:
                    return None
                else:
                    nodes = self.parent_graph.branch_list[internal_id]
                    return Point.from_array(self.node_coord(nodes).mean(axis=0))
            else:
                return Point.from_array(curve[int(pos * curve.shape[0])])

        if ids is None:
            return [midpoint(c) for c in range(self.branch_count)]

        ids, is_single = as_1d_array(ids)
        if isinstance(ids, np.ndarray):
            internal_ids = self._graph_to_internal_branch_ids(ids, graph_index=graph_index, check_valid=False)

            if is_single:
                return midpoint(internal_ids[0])
            return [midpoint(i) for i in internal_ids]

        else:
            raise TypeError("Invalid type for branches index.")

    @overload
    def branch_closest_index(
        self,
        yx: PointArrayLike,
        branch_ids: Optional[int | npt.NDArray[np.int32]] = None,
        *,
        return_distance: Literal[False] = False,
        interpolate: bool = False,
    ) -> Int1DArray: ...
    @overload
    def branch_closest_index(
        self,
        yx: PointArrayLike,
        branch_ids: Optional[int | npt.NDArray[np.int32]] = None,
        *,
        return_distance: Literal[True],
        interpolate: bool = False,
    ) -> Tuple[Int1DArray, Float1DArray]: ...
    def branch_closest_index(
        self,
        yx: PointArrayLike,
        branch_ids: Optional[int | npt.NDArray[np.int32]] = None,
        *,
        return_distance=False,
        interpolate: bool = False,
    ) -> Int1DArray | Tuple[Int1DArray, Float1DArray]:
        """Return the closest point(s) on the branch(es) to a set of point(s).

        Parameters
        ----------
        points : np.ndarray
            The coordinates of the point(s) as a 2D array of shape (N, 2).

        branch_ids : Optional[int | np.ndarray], optional
            The id of the branch(es), by default None

        return_distance : bool, optional
            Whether to return the distance to the closest point(s) as well, by default False.

        interpolate : bool, optional
            Whether to interpolate the branch curve when it is composed of several disjoint segments, or when no curve points are present, by default False.
            The interpolation is a linear interpolation between the disjoint segments, or between the branch nodes if no curve points are present.

        Returns
        -------
        np.ndarray
            The index of the closest point on the branch(es) to the input point(s).

            - If ``branch_ids`` is a scalar, the output is a 1D array of shape (N) containing the indices of the closest points on the branch.
            - If ``branch_ids`` is an iterable of int, the output is a 2D array of shape (len(ids), N,) containing the indices of the closest points on each branch.
        """  # noqa: E501
        yx = np.atleast_2d(yx)
        assert yx.ndim == 2 and yx.shape[1] == 2, "The yx points should be a 2D array of shape (N, 2)."

        def closest_index(branch_id, curve, points):
            if curve is None or curve.shape[0] == 0:
                if interpolate:
                    nodes_yx = self.node_coord(self.parent_graph.branch_list[branch_id])
                    _, d = nearest_point_on_segment(points, *nodes_yx, return_distance=True)
                    return np.full(points.shape[0], -1), d[:, 0]
                else:
                    return np.full(points.shape, np.nan), np.full(points.shape[0], np.inf)
            distances = np.linalg.norm(curve[:, None, :] - points[None, :, :], axis=2)
            curve_ids = np.argmin(distances, axis=0)
            distances = distances[curve_ids, np.arange(points.shape[0])]

            if interpolate:
                nodes_yx = self.node_coord(self.parent_graph.branch_list[branch_id])
                discontinuity = np.array([0] + discontiguous_index(curve) + [len(curve)], dtype=np.int_)
                discontinuity_yx_a = np.concatenate([nodes_yx[0, None], curve[discontinuity[1:] - 1]])
                discontinuity_yx_b = np.concatenate([curve[discontinuity[:-1]], nodes_yx[1, None]])

                _, dist = nearest_point_on_segment(points, discontinuity_yx_a, discontinuity_yx_b, return_distance=True)
                closest_discontinuity = np.argmin(dist, axis=1)
                closest_discontinuity_dist = dist[np.arange(points.shape[0]), closest_discontinuity]
                mask = closest_discontinuity_dist < distances
                curve_ids[mask] = discontinuity[closest_discontinuity[mask]]
                distances[mask] = closest_discontinuity_dist[mask]

            return curve_ids, distances

        if branch_ids is None:
            branch_ids = np.arange(self.branch_count)
            is_single = False
        else:
            branch_ids, is_single = as_1d_array(branch_ids)

        branches_closest_index = []
        branches_closest_distance = []
        for i in branch_ids:
            curve_id, dist = closest_index(i, self._branch_curve[i], yx)
            branches_closest_index.append(curve_id)
            branches_closest_distance.append(dist)

        if not return_distance:
            return branches_closest_index[0] if is_single else np.stack(branches_closest_index)

        return (
            branches_closest_index[0] if is_single else np.stack(branches_closest_index),
            branches_closest_distance[0] if is_single else np.stack(branches_closest_distance),
        )

    @overload
    def closest_branches(
        self, yx: PointArrayLike, *, return_distance: Literal[False] = False, interpolate: bool = False
    ) -> Int2DArray: ...
    @overload
    def closest_branches(
        self, yx: PointArrayLike, *, return_distance: Literal[True], interpolate: bool = False
    ) -> Tuple[Int2DArray, Float1DArray]: ...
    def closest_branches(
        self, yx: PointArrayLike, *, return_distance=False, interpolate: bool = False
    ) -> Int2DArray | Tuple[Int2DArray, Float1DArray]:
        """Return the closest branch for each point.

        Parameters
        ----------
        points : Float2DArrayLike
            The coordinates of the point(s) as a 2D array of shape (N, 2).

        Returns
        -------
        np.ndarray
            A (N, 2) array containing the id of the closest branch and the index of the closest point on the branch for each input point.

        np.ndarray
            If ``return_distance`` is True, also return a 1D array of shape (N,) containing the distance to the closest point on the closest branch for each input point.

        """  # noqa: E501
        yx = np.asarray(yx)
        if is_single := yx.ndim == 1:
            yx = yx[None, :]
        assert yx.ndim == 2 and yx.shape[1] == 2, "The yx points should be a 2D array of shape (N, 2)."

        closest_index, closest_distance = self.branch_closest_index(
            yx, branch_ids=None, return_distance=True, interpolate=interpolate
        )
        closest_branch = np.argmin(closest_distance, axis=0)
        closest_index = closest_index[closest_branch, np.arange(yx.shape[0])]
        closest_distance = closest_distance[closest_branch, np.arange(yx.shape[0])]

        closest = np.stack((closest_branch, closest_index), axis=1)
        if is_single:
            closest = closest[0]
            closest_distance = closest_distance[0]
        return closest if not return_distance else (closest, closest_distance)

    @overload
    def closest_nodes(self, yx: PointArrayLike, *, return_distance: Literal[False] = False) -> npt.NDArray[np.int_]: ...
    @overload
    def closest_nodes(
        self, yx: PointArrayLike, *, return_distance: Literal[True]
    ) -> Tuple[npt.NDArray[np.int_], Float1DArray]: ...
    def closest_nodes(
        self, yx: PointArrayLike, *, return_distance=False
    ) -> npt.NDArray[np.int_] | Tuple[npt.NDArray[np.int_], Float1DArray]:
        """Return the closest node for each point.

        Parameters
        ----------
        points : Float2DArrayLike
            The coordinates of the point(s) as a 2D array of shape (N, 2).

        Returns
        -------
        np.ndarray
            A (N, 2) array containing the id of the closest branch and the index of the closest point on the branch for each input point.
        """  # noqa: E501
        yx = np.asarray(yx)
        if is_single := yx.ndim == 1:
            yx = yx[None, :]

        assert yx.ndim == 2 and yx.shape[1] == 2, "The yx points should be a 2D array of shape (N, 2)."

        dist = np.linalg.norm(self._nodes_coord[:, None, :] - yx[None, :, :], axis=2)
        closest_node = np.argmin(dist, axis=0)
        closest_distance = dist[closest_node, np.arange(yx.shape[0])]

        if is_single:
            closest_node = closest_node[0]
            closest_distance = closest_distance[0]

        return closest_node if not return_distance else (closest_node, closest_distance)

    def set_branch_curve(
        self,
        curve: None | npt.NDArray | List[Optional[npt.NDArray]],
        branch_id: Optional[int | npt.NDArray[np.int32]] = None,
        *,
        graph_index=True,
    ) -> None:
        """Set a geometric attribute of one or more branch.

        Parameters
        ----------
        curve : None | np.ndarray | List[np.ndarray]
            The coordinates of the pixels that compose the branches. If None, the branch curve is set to an empty array.
        branch_id : Optional[int  |  npt.NDArray[np.int32]], optional
            The id of the branch(es), by default None
        version : Optional[str], optional
            The version of the attribute, by default ''
        graph_index : bool, optional
            Whether the branch_id is indexed according to the graph (True) or to the internal representation (False), by default True.
        no_check : bool, optional
            If True, do not check the validity of the attribute data, by default False

        """  # noqa: E501
        if branch_id is not None and np.isscalar(branch_id):
            curve = [curve]
        branch_id = self._graph_to_internal_branch_ids(branch_id, graph_index=graph_index)

        assert len(branch_id) == len(curve), "The number of branches and curves should be the same."
        for i, c in zip(branch_id, curve, strict=True):
            if c is None:
                self.clear_branch_gdata(i)
            else:
                self._branch_curve[i] = readonly(c)

    def _geodata_edit_ctx(self, internal_branch_id: int, attr_name: str = "") -> BranchGeoDataEditContext:
        """Return the context of a branch."""
        return BranchGeoDataEditContext(
            attr_name=attr_name,
            branch_id=internal_branch_id,
            curve=self._branch_curve[internal_branch_id],
            geodata_attrs=self.list_branch_data(internal_branch_id, graph_index=False),
            info={},
        )

    ####################################################################################################################
    #  === COMPUTABLE GEOMETRIC PROPERTIES ===
    ####################################################################################################################
    def skeleton_label_map(
        self,
        skeleton: bool = True,
        boundaries: bool = False,
        calibre: bool | Literal["tip"] = False,
        *,
        boundaries_attr=None,
        connect_nodes: bool = False,
        interpolate: bool = False,
    ) -> Int2DArray:
        """Return a label map of the branches.

        Returns
        -------
        np.ndarray
            An array of the same shape as the image where each pixel is labeled with the id of the branch it belongs to.
        """
        import torch

        from ..utils.cpp_extensions.fvt_cpp import draw_skeleton_labels

        domain_shape = self.domain.shape
        top_left = np.array([self.domain.top, self.domain.left])
        curves = [torch.from_numpy(c - top_left).int() for c in self.branch_curve()]
        branch_label_map = np.zeros(domain_shape, dtype=np.int32)

        if skeleton:
            if connect_nodes:
                nodes_coord = torch.from_numpy(self.node_coord(graph_index=False) - top_left).int()
                branch_list = torch.from_numpy(self.parent_graph.branch_list).int()
            else:
                nodes_coord = torch.empty(0, 2, dtype=torch.int)
                branch_list = torch.empty(0, 2, dtype=torch.int)
            branch_label_map_torch = torch.from_numpy(branch_label_map)
            draw_skeleton_labels(curves, branch_label_map_torch, nodes_coord, branch_list, interpolate)

        if boundaries or calibre:
            from skimage.draw import line

            if boundaries_attr is None:
                try:
                    calibre_desc = self._fetch_branch_data_descriptor(VBranchGeoData.Fields.BOUNDARIES)
                except KeyError:
                    try:
                        calibre_desc = self._fetch_branch_data_descriptor(VBranchGeoData.Fields.TIPS_BOUNDARIES)
                    except KeyError:
                        raise KeyError("No calibre attribute found.") from None
            else:
                calibre_desc = self._fetch_branch_data_descriptor(boundaries_attr)
            assert issubclass(calibre_desc.geo_type, (VBranchGeoData.TipsData, VBranchGeoData.Curve)), (
                f"Invalid attribute for boundaries of branches tips: {calibre_desc.name}."
            )
            boundaries_data = self.branch_data(calibre_desc)

            if calibre:
                only_tip = calibre == "tip"
                lines = []
                for branch_id, b_bound_data in zip(self.branch_ids, boundaries_data, strict=True):
                    if isinstance(b_bound_data, VBranchGeoData.TipsData):
                        for boundL, boundR in b_bound_data.data.astype(np.int_):
                            lines += [(Point(*boundL), Point(*boundR), branch_id + 1)]
                    elif isinstance(b_bound_data, VBranchGeoData.Curve):
                        bounds_lr = b_bound_data.data.astype(np.int_)
                        if bounds_lr.shape[0] == 0:
                            continue
                        for boundL, boundR in [bounds_lr[0]] if only_tip else bounds_lr[::4]:
                            lines += [(Point(*boundL), Point(*boundR), branch_id + 1)]
                        lines += [(Point(*bounds_lr[-1, 0]), Point(*bounds_lr[-1, 1]), branch_id + 1)]

                for p0, p1, color in lines:
                    if p0 != p1 and p0 in self.domain and p1 in self.domain:
                        branch_label_map[line(*p0, *p1)] = color

            if boundaries:
                bounds_lr = [_.data.astype(np.int32) for _ in boundaries_data if _.data.size > 0]
                bounds_l = [torch.from_numpy(bound[:, 0]) for bound in bounds_lr]
                bounds_r = [torch.from_numpy(bound[:, 1]) for bound in bounds_lr]
                empty = torch.empty(0, 2, dtype=torch.int)
                branch_label_map_torch = torch.from_numpy(branch_label_map)
                draw_skeleton_labels(bounds_l, branch_label_map_torch, empty, empty, interpolate)
                draw_skeleton_labels(bounds_r, branch_label_map_torch, empty, empty, interpolate)

        return branch_label_map  # type: ignore

    @overload
    def branch_arc_length(self, graph_ids: int, fast_approximation=True) -> float: ...
    @overload
    def branch_arc_length(
        self, graph_ids: Optional[Iterable[int]] = None, fast_approximation=True
    ) -> npt.NDArray[np.float32]: ...
    def branch_arc_length(
        self, graph_ids: Optional[int | Iterable[int]] = None, fast_approximation=True
    ) -> float | npt.NDArray[np.float32]:
        """Return the arc length of the branches.

        Parameters
        ----------
        graph_ids : Optional[int | Iterable[int]], optional
            The id of the branch(es) to retrieve the length.
            If None (by default), the length of all branches is returned.

        fast_approximation : bool, optional
            If True (by default), the arc length is approximated by the number of pixels in the branch curve.
            If False, the arc length is computed as the sum of the Euclidean distances between consecutive points in the branch curve.

        Returns
        -------
        Dict[int, float]
            The arc length of the branches. The keys are the branch ids and the values are the arc length of the    branches.
        """  # noqa: E501
        if graph_ids is None:
            is_single = False
        else:
            graph_ids, is_single = as_1d_array(graph_ids)

        arc = []
        for curve in self.branch_curve(graph_ids):
            if curve is None:
                arc.append(0)
            elif fast_approximation:
                arc.append(curve.shape[0])
            else:
                arc.append(np.linalg.norm(curve[-1] - curve[0]))
        return arc[0] if is_single else np.asarray(arc, dtype=np.float32)

    def branch_chord_length(self, graph_ids: Optional[int | Iterable[int]] = None) -> float | npt.NDArray[np.float32]:
        """Return the chord length of the branches."""
        if graph_ids is None:
            is_single = False
        else:
            graph_ids, is_single = as_1d_array(graph_ids)
        chord = []

        for curve in self.branch_curve(graph_ids):
            if curve is None:
                chord.append(0)
            else:
                chord.append(np.linalg.norm(curve[-1] - curve[0]))
        return chord[0] if is_single else np.asarray(chord, dtype=np.float32)

    def distance_to_branch(
        self, point: Point, branch_id: Optional[int] = None, return_closest_point=False
    ) -> float | Tuple[float, Point]:
        """Return the distance between a point and a branch.

        Parameters
        ----------
        point : Point
            The point from which to measure the distance.

        branch_id : Optional[int], optional
            The id of the branch, by default None. If None, the distance to the closest branch is returned.

        return_point : bool, optional
            If True, return the closest point on the branch, by default False.

        Returns
        -------
        float
            The distance between the point and the branch.

        Point
            The closest point on the branch.
        """
        if branch_id is None:
            min_distance = np.inf
            closest_point = None
            for branchID in self.branch_ids:
                distance, closest_point = self.distance_to_branch(point, branchID, True)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = closest_point
            return min_distance, closest_point if return_closest_point else min_distance

        curve = self.branch_curve(branch_id)
        point = np.asarray(point)[None, :]
        distances = np.linalg.norm(point - curve, axis=1)
        min_i = np.argmin(distances)
        return distances[min_i], curve[min_i] if return_closest_point else distances[min_i]

    def branch_with_unknown_curve(self) -> npt.NDArray[np.int_]:
        """Return the index of the branches without geometric data."""
        return np.where([c is None or len(c) == 0 for c in self._branch_curve])[0]

    ####################################################################################################################
    #  === BRANCH GEOMETRIC ATTRIBUTES ===
    ####################################################################################################################
    def list_branch_data(self, branch_id: int, *, graph_index=True) -> Dict[str, VBranchGeoDataBase]:
        """Return the attributes of a branch.

        Parameters
        ----------
        branch_id : int
            The id of the branch.

        Returns
        -------
        Dict[str, VBranchGeoAttr]
            The attributes of the branch. The keys are the attribute names and the values are the attribute values.
        """
        if graph_index:
            internal_id = self._graph_to_internal_branch_ids(branch_id, graph_index=True, check_valid=False)
        else:
            internal_id = branch_id

        return {
            attr: data
            for attr, attr_data in self._branch_data_dict.items()
            if (data := attr_data[internal_id]) is not None
        }

    def has_branch_data(self, attr_name: VBranchGeoDataKey) -> bool:
        """Return True if the branch has the attribute."""
        try:
            desc = self._fetch_branch_data_descriptor(attr_name)
            return desc is not None
        except KeyError:
            return False

    @overload
    def branch_data(
        self, attr_name: VBranchGeoDescriptor[T_VBranchGeoData], branch_id: int, *, graph_index=True
    ) -> T_VBranchGeoData | None: ...
    @overload
    def branch_data(
        self, attr_name: VBranchGeoDataKey, branch_id: int, *, graph_index=True
    ) -> VBranchGeoDataBase | None: ...
    @overload
    def branch_data(
        self,
        attr_name: VBranchGeoDescriptor[T_VBranchGeoData],
        branch_id: Optional[Int1DArray] = None,
        *,
        graph_index=True,
    ) -> List[T_VBranchGeoData | None]: ...
    @overload
    def branch_data(
        self, attr_name: VBranchGeoDataKey, branch_id: Optional[Int1DArray] = None, *, graph_index=True
    ) -> List[T_VBranchGeoData | None]: ...
    @overload
    def branch_data(self, *, branch_id: int, graph_index=True) -> Dict[str, T_VBranchGeoData | None]: ...
    @overload
    def branch_data(
        self, *, branch_id: Optional[Int1DArray] = None, graph_index=True
    ) -> Dict[str, List[T_VBranchGeoData | None]]: ...
    def branch_data(
        self,
        attr_name: Optional[VBranchGeoDataKey] = None,
        branch_id: Optional[int | Int1DArray] = None,
        *,
        graph_index=True,
    ) -> (
        VBranchGeoDataBase
        | None
        | List[T_VBranchGeoData | None]
        | Dict[str, T_VBranchGeoData | None]
        | Dict[str, List[T_VBranchGeoData | None]]
    ):
        """Return the attribute of a branch.

        Parameters
        ----------
        branch_id : int
            The id of the branch.

        attr_name : str
            The name of the attribute.

        Returns
        -------
        VBranchGeoAttr
            The attribute of the branch.
        """
        if attr_name is None:
            return self._branch_data_dict if branch_id is None else self.list_branch_data(branch_id, graph_index=True)  # type: ignore

        attr = self._fetch_branch_data(attr_name)
        if branch_id is None:
            return attr  # type: ignore

        branch_id, is_single = as_1d_array(branch_id)

        internal_id = self._graph_to_internal_branch_ids(branch_id, graph_index=graph_index, check_valid=False)
        if is_single:
            return attr[internal_id[0]] if internal_id[0] >= 0 else None  # type: ignore
        return [attr[i] if i >= 0 else None for i in internal_id]  # type: ignore

    @overload
    def tip_data(
        self,
        attrs: VBranchGeoDataKey,
        branch_id: Optional[IndicesLike] = None,
        first_tip: Optional[Bool1DArrayLike] = None,
        *,
        graph_index=True,
    ) -> npt.NDArray: ...
    @overload
    def tip_data(
        self,
        attrs: Optional[List[VBranchGeoDataKey]] = None,
        branch_id: Optional[IndicesLike] = None,
        first_tip: Optional[Bool1DArrayLike] = None,
        *,
        graph_index=True,
    ) -> Dict[str, npt.NDArray]: ...
    def tip_data(
        self,
        attrs: Optional[VBranchGeoDataKey | List[VBranchGeoDataKey]] = None,
        branch_id: Optional[IndicesLike] = None,
        first_tip: Optional[Bool1DArrayLike] = None,
        *,
        graph_index=True,
    ) -> npt.NDArray | Dict[str, npt.NDArray]:
        """Return the geometric data of the tips of the given branch(es).

        Parameters
        ----------
        branch_id : Optional[int | np.ndarray], optional
            The id of the branch(es), by default None

        attrs : Optional[str | List[str]], optional
            The name of the attribute(s) to return. If None, all the attributes are returned, by default None

        first_tip : Optional[bool | np.ndarray], optional

            - If None (by default), return both tips of the branch(es).
            - If True, return the first tip of the branch(es).
            - If False, return the second tip of the branch(es).
            - If an array of bool, return the first or the second tip of each branch according to the value of the array.

        Returns
        -------
        np.ndarray
            When ``attrs`` is a name of a single :class:`VBranchTipsData` attribute, only its data is returned.
            - If first_tip is not None, the data is an array of shape (b, ...) where b is the length of ``branch_ids`` or (, ...) if ``branch_ids`` is a scalar.
            - If first_tip is None, the data is an array of shape (b, 2, ...) where b is the length of ``branch_ids`` or (2, ...) if ``branch_ids`` is a scalar.

        Dict[str, np.ndarray]
            When ``attrs`` is a list of names of :class:`VBranchTipsData` attributes, a dictionary is returned with the following entries:
            - "yx": the coordinates of the curve tip (or nan if the curve is empty).
            - The requested tips data as described above.

        """  # noqa: E501
        attr_single = False
        if attrs is None:
            attrs_desc = self._branches_attrs_descriptors.values()
        elif isinstance(attrs, list):
            attrs_desc = [self._fetch_branch_data_descriptor(attr) for attr in attrs]
        else:
            attrs_desc = [self._fetch_branch_data_descriptor(attrs)]
            attr_single = True
        attrs_desc = [
            attr
            for attr in attrs_desc
            if attr.geo_type is not None and issubclass(attr.geo_type, VBranchGeoData.TipsData)
        ]

        if branch_id is None:
            branch_id = np.arange(self.branch_count)
        branch_ids, is_single = as_1d_array(branch_id)
        branch_ids = self._graph_to_internal_branch_ids(branch_ids, graph_index=graph_index)

        if attr_single:
            attr_data = self._branch_data_dict[attrs_desc[0]]
            data = [attr_data[_].data for _ in branch_ids]
            if first_tip is not None:
                if isinstance(first_tip, (bool, np.bool_)):
                    data = [d[0] for d in data] if first_tip else [d[1] for d in data]
                else:
                    data = [d[0 if first else 1] for d, first in zip(data, first_tip, strict=True)]
            return data[0] if is_single else np.stack(data)

        else:
            curves = self.branch_curve(branch_ids)
            out = {attr: [] for attr in attrs_desc}

            nan = np.array([np.nan, np.nan], dtype=np.float32)
            if first_tip is not None:
                if type(first_tip) in (bool, np.bool_):
                    out["yx"] = [(curve[0] if first_tip else curve[-1]) if len(curve) else nan for curve in curves]
                else:
                    out["yx"] = [
                        (curve[0 if first else -1]) if len(curve) else nan
                        for curve, first in zip(curves, first_tip, strict=True)
                    ]
            else:
                out["yx"] = [curve[[0, -1]] if len(curve) else np.array((nan, nan)) for curve in curves]

            for attr in attrs_desc:
                attr_data = self._branch_data_dict[attr]
                data = [attr_data[_].data for _ in branch_ids]

                if first_tip is not None:
                    if type(first_tip) in (bool, np.bool_):
                        out[attr] = [d[0] if first_tip else d[-1] for d in data]
                    else:
                        out[attr] = [d[0 if first else -1] for d, first in zip(data, first_tip, strict=True)]
                else:
                    out[attr] = data

            return {k: v[0] for k, v in out.items()} if is_single else {k: np.stack(v) for k, v in out.items()}

    @overload
    def tip_data_around_node(self, attrs: VBranchGeoDataKey, node_id: int, *, graph_index: bool) -> npt.NDArray: ...
    @overload
    def tip_data_around_node(
        self, attrs: VBranchGeoDataKey, node_id: Optional[Int1DArray], *, graph_index: bool
    ) -> List[npt.NDArray]: ...
    @overload
    def tip_data_around_node(
        self, attrs: List[VBranchGeoDataKey], node_id: int, *, graph_index: bool
    ) -> Dict[str, npt.NDArray]: ...
    @overload
    def tip_data_around_node(
        self,
        attrs: Optional[List[VBranchGeoDataKey]] = None,
        node_id: Optional[Int1DArray] = None,
        *,
        graph_index: bool,
    ) -> Dict[str, List[npt.NDArray]]: ...
    def tip_data_around_node(
        self,
        attrs: Optional[VBranchGeoDataKey | List[VBranchGeoDataKey]] = None,
        node_id: Optional[int | Int1DArray] = None,
        *,
        graph_index=True,
    ) -> npt.NDArray | List[npt.NDArray] | Dict[str, npt.NDArray] | Dict[str, List[npt.NDArray]]:
        """Return the geometric data of the tips of the branches incident to a node.

        Parameters
        ----------
        attrs : Optional[str | List[str]], optional
            The name of the attribute(s) to return. If None, all the attributes are returned, by default None

        node_id : Optional[int | np.ndarray], optional
            The id of the node(s), by default None

        Returns
        -------
        np.ndarray | List[np.ndarray]
            If ``attrs`` is a name of a single :class:`VBranchTipsData` attribute, only its data is returned.

            - If ``nodes_id`` is a scalar, the data is an array of shape (b, ...) where b is the number of branches incident to the node.
            - If ``nodes_id`` is an iterable of int, the data is a list of such arrays.

        Dict[str, np.ndarray] | Dict[str, List[np.ndarray]]
            A dictionary storing the requested tips data and a "branches" entry containing the ids of the branches incident to the nodes.

            - If ``nodes_id`` is a scalar, the "branches" entry is a 1D array of length b, and all other entries are arrays of shape (b, ...) where b is the number of branches incident to the node.
            - If ``nodes_id`` is an iterable of int, all entries are lists of such arrays.
        """  # noqa: E501
        attr_single = False
        if attrs is None:
            attrs = self._branches_attrs_descriptors.values()
        elif isinstance(attrs, list):
            attrs = [self._fetch_branch_data_descriptor(attr) for attr in attrs]
        else:
            attrs = [self._fetch_branch_data_descriptor(attrs)]
            attr_single = True
        attrs = [attr.name for attr in attrs if issubclass(attr.geo_type, VBranchGeoData.TipsData)]

        if node_id is None:
            node_id = np.arange(self.node_count)
        node_id, is_single = as_1d_array(node_id)

        branch_ids, branch_dirs = self.parent_graph.adjacent_branches_per_node(node_id, return_branch_direction=True)
        branch_ids = [self._graph_to_internal_branch_ids(bid, graph_index=graph_index) for bid in branch_ids]

        if attr_single:
            attr_data = self._branch_data_dict[attrs[0]]
            out = []
            for bids, bdirs in zip(branch_ids, branch_dirs, strict=True):
                data = [attr_data[_] for _ in bids]
                data = np.stack([d.data[0 if bdir else 1] for d, bdir in zip(data, bdirs, strict=True)])
                out.append(data)
            return out[0] if is_single else out

        else:
            curves = [self.branch_curve(bids) for bids in branch_ids]
            out = {"branches": branch_ids} | {attr: [] for attr in attrs}
            nan = np.array([np.nan, np.nan], dtype=np.float32)
            out["yx"] = [
                np.stack([c[0 if d else -1] if len(c) else nan for c, d in zip(node_curves, dirs, strict=True)])
                for node_curves, dirs in zip(curves, branch_dirs, strict=True)
            ]

            for attr in attrs:
                attr_data = self._branch_data_dict[attr]
                for bids, bdirs in zip(branch_ids, branch_dirs, strict=True):
                    data = [attr_data[_] for _ in bids]
                    data = np.stack([bdata.data[0 if bdir else 1] for bdata, bdir in zip(data, bdirs, strict=True)])
                    out[attr].append(data)
            return {k: v[0] for k, v in out.items()} if is_single else out

    def tip_coord(
        self,
        branch_id: Optional[int | Int1DArrayLike] = None,
        first_tip: Optional[bool | Bool1DArrayLike] = None,
        *,
        use_nodes_if_missing: bool = True,
        graph_index=True,
    ) -> np.ndarray:
        """Return the coordinates of the tips of the branches.

        Parameters
        ----------
        branch_id : Optional[int | np.ndarray], optional
            The id of the branch(es), by default None

        first_tip : Optional[bool | np.ndarray], optional
            - If None, return the coordinates of both tips of the branch(es).
            - If True, return the coordinates of the first tip of the branch(es).
            - If False, return the coordinates of the second tip of the branch(es).
            - If an array of bool, return the coordinates of the first or the second tip of each branch according to the value of the array.

        use_nodes_if_missing : bool, optional
            If True, infer the tip position from the nodes if the branch curve is missing, by default True.

            .. warning::
                This option requires the parent graph to be set.

        graph_index : bool, optional
            Whether the branch_id is indexed according to the graph (True) or to the internal representation (False), by default True.

        Returns
        -------
        np.ndarray
            The coordinates of the tips of the branch(es).

            - If first_tip is not None, the data is an array of shape (b, 2) where b is the length of ``branch_ids`` or (2, ) if ``branch_ids`` is a scalar.
            - If first_tip is None, the data is an array of shape (b, 2, 2) where b is the length of ``branch_ids`` or (2, 2) if ``branch_ids`` is a scalar.
        """  # noqa: E501
        if branch_id is None:
            is_single = False
        else:
            branch_id, is_single = as_1d_array(branch_id)

        nan = np.array([np.nan, np.nan], dtype=np.float32)
        branch_curves = self.branch_curve(branch_id)
        tips_coord = np.array(
            [[nan, nan] if curve is None or len(curve) <= 1 else curve[[0, -1]] for curve in branch_curves]
        )
        unknown_tips: npt.NDArray[np.bool_] = np.isnan(tips_coord).any(axis=2)

        if unknown_tips.any() and use_nodes_if_missing:
            assert self.parent_graph is not None, (
                "The parent graph is not set, impossible to infer the branch tip coordinates."
            )
            branch_ids = np.asarray(branch_id) if branch_id is not None else self.branch_ids
            unk_branches, unk_tips = np.where(unknown_tips)
            unk_branches = branch_ids[unk_branches]
            unk_tips_nodes = self.parent_graph.branch_list[unk_branches, unk_tips]
            tips_coord[unknown_tips] = self.node_coord(unk_tips_nodes.flatten())

        if first_tip is not None:
            if type(first_tip) in (bool, np.bool_):
                tips_coord = tips_coord[:, 0] if first_tip else tips_coord[:, 1]
            else:
                first_tip = np.atleast_1d(first_tip).astype(bool).flatten()
                tip_id = np.where(first_tip[:, None, None], 0, 1)
                tips_coord = np.take_along_axis(tips_coord, tip_id, axis=1).squeeze(1)

        return tips_coord[0] if is_single else tips_coord

    def tip_tangent(
        self,
        branch_id: Optional[int | npt.ArrayLike] = None,
        first_tip: Optional[Bool1DArrayLike] = None,
        *,
        infer_from_nodes_if_missing: bool = True,
        attr: VBranchGeoDataKey = VBranchGeoData.Fields.TIPS_TANGENT,
        graph_index=True,
    ) -> npt.NDArray[np.float64]:
        """Return the tangent of the tips of the branches. Tangents are unit vectors oriented towards the inside of the branch.

        Parameters
        ----------
        branch_id : Optional[int | np.ndarray], optional
            The id of the branch(es), by default None

        first_tip : Optional[bool | np.ndarray], optional
            - If None, return the tangent of both tips of the branch(es).
            - If True, return the tangent of the first tip of the branch(es).
            - If False, return the tangent of the second tip of the branch(es).
            - If an array of bool, return the tangent of the first or the second tip of each branch according to the value of the array.

        infer_from_nodes_if_missing : bool, optional
            If True, infer the branch directions from the nodes if the tangent is missing, by default True.

            .. warning::
                This option requires the parent graph to be set.

        attr : str, optional
            The name of the attribute to return, by default "TIPS_TANGENT"

        graph_index : bool, optional
            Whether the branch_id is indexed according to the graph (True) or to the internal representation (False), by default True.

        Returns
        -------
        np.ndarray
            The tangent of the tips of the branch(es).

            - If first_tip is not None, the data is an array of shape (b, 2) where b is the length of ``branch_ids`` or (2, ) if ``branch_ids`` is a scalar.
            - If first_tip is None, the data is an array of shape (b, 2, 2) where b is the length of ``branch_ids`` or (2, 2) if ``branch_ids`` is a scalar.
        """  # noqa: E501
        branch_ids, is_single = as_1d_array(branch_id) if branch_id is not None else (None, False)

        if self.has_branch_data(attr):
            tangents = self.tip_data(attr, branch_ids, first_tip, graph_index=graph_index)
            unknown_tangents: npt.NDArray[np.bool_] = np.isnan(tangents).any(axis=1) | np.all(tangents == 0, axis=1)
        else:
            B = len(branch_ids) if branch_ids is not None else self.branch_count
            tangents = np.empty((B, 2, 2) if first_tip is None else (B, 2))
            unknown_tangents = np.ones(B, dtype=bool)

        if unknown_tangents.any() and infer_from_nodes_if_missing:
            assert self.parent_graph is not None, (
                "The parent graph is not set, impossible to infer the branch directions."
            )
            branch_list = self.parent_graph.branch_list

            return_both_tips = first_tip is None
            if return_both_tips:
                first_tip = np.tile([True, False], len(tangents))
                tangents = tangents.reshape(-1, 2)
            elif isinstance(first_tip, bool):
                first_tip = np.full(len(tangents), first_tip)
            else:
                first_tip = np.atleast_1d(first_tip).astype(bool).flatten()

            unknown_tangents: npt.NDArray[np.bool_] = np.isnan(tangents).any(axis=1) | np.all(tangents == 0, axis=1)
            if unknown_tangents.any():
                unk_branches = np.asarray(branch_ids) if branch_ids is not None else self.branch_ids
                if return_both_tips:
                    unk_branches = np.repeat(unk_branches, 2)
                unk_branches = unk_branches[unknown_tangents]
                nodes = branch_list[unk_branches]
                loop_branches = nodes[:, 0] == nodes[:, 1]
                if loop_branches.any():
                    warnings.warn(
                        f"Branch(es): {np.unique(unk_branches[loop_branches])} are self-loop branches "
                        "with invalid tangents, their tips tangent will be set to (0, 0).",
                        stacklevel=2,
                    )
                    nodes = nodes[~loop_branches]
                    loop_branches = np.argwhere(unknown_tangents).flatten()[loop_branches]
                    tangents[loop_branches] = 0
                    unknown_tangents[loop_branches] = False

                unk_first_tip: npt.NDArray[np.bool_] = first_tip[unknown_tangents]

                if nodes.shape[0] != 0:
                    t = np.diff(self.node_coord(nodes.flatten()).reshape(-1, 2, 2), axis=1).squeeze(1)
                    t_zero = np.all(t == 0, axis=1)
                    if t_zero.any():
                        warnings.warn(
                            f"The nodes of branch(es) {np.unique(unk_branches[t_zero])} are superposed,"
                            "their tips tangent will be set to (0, 0).",
                            stacklevel=2,
                        )
                    if not t_zero.all():
                        t[~t_zero] = t[~t_zero] / np.linalg.norm(t[~t_zero], axis=1)[:, None]
                        t[~unk_first_tip] *= -1
                    tangents[unknown_tangents] = t

            if return_both_tips:
                tangents = tangents.reshape(-1, 2, 2)
        return tangents[0] if is_single else tangents

    def set_branch_data(
        self,
        attr_name: VBranchGeoDataKey,
        attr_data: VBranchGeoDataLike | List[Optional[VBranchGeoDataLike]],
        branch_id: Optional[int | npt.NDArray[np.int32]] = None,
        *,
        graph_index=True,
        no_check=False,
    ) -> None:
        """Set a geometric attribute of one or more branch.

        Parameters
        ----------
        attr_name : str
            The name of the attribute
        attr_data : VBranchGeoAttr | np.ndarray | List[Optional[VBranchGeoAttr | np.ndarray]]
            The attribute data to set. If a list is provided, the length should be the same as the number of branches.
        branch_id : Optional[int  |  npt.NDArray[np.int32]], optional
            The id of the branch(es), by default None
        version : Optional[str], optional
            The version of the attribute, by default ''
        graph_index : bool, optional
            Whether the branch_id is indexed according to the graph (True) or to the internal representation (False), by default True.
        no_check : bool, optional
            If True, do not check the validity of the attribute data, by default False

        """  # noqa: E501
        # === Check Arguments ===
        # Check: branch_id
        if isinstance(branch_id, int):
            branch_id = [branch_id]
        if branch_id is None:
            branch_id = np.arange(self.branch_count)
        else:
            branch_id = self._graph_to_internal_branch_ids(branch_id, graph_index=graph_index)

        # Check: attr_data
        if isinstance(attr_data, VBranchGeoDataBase):
            assert len(branch_id) == 1, "Invalid number of branches for attribute data."
            attr_data = [attr_data]
        else:
            attr_data = copy(attr_data)
            assert len(attr_data) == len(branch_id), "Invalid number of branches for attribute data."

        attr_desc = VBranchGeoData.Descriptor.parse(attr_name)
        attr_name = attr_desc.name
        attr_type = attr_desc.geo_type

        for i, data in enumerate(attr_data):
            if data is None:
                attr_data[i] = attr_desc.empty
                continue

            ctx = self._geodata_edit_ctx(branch_id[i], attr_name)

            if isinstance(data, np.ndarray):
                if attr_type is None:
                    attr_type = VBranchGeoData.Curve
                data = attr_type(data)
            if attr_type is not None:
                assert isinstance(data, attr_type), (
                    f"Inconsistent type for attribute {attr_name} of branch {branch_id[i]}: {type(data)}. "
                    f"Previous branches were of type: {attr_type}."
                )
            elif isinstance(data, VBranchGeoDataBase):
                attr_type = type(data)
            else:
                raise ValueError(f"Invalid type for attribute data: {type(data)}.")
            if not no_check:
                is_invalid = data.is_invalid(ctx=ctx)
                if is_invalid:
                    raise ValueError(f"Invalid attribute {attr_name} of branch {branch_id[i]}.\n{is_invalid}.")
            attr_data[i] = data

        # === Set Attribute ===
        attr = self._fetch_branch_data(attr_name, attr_type, emplace=True)
        for i, data in zip(branch_id, attr_data, strict=True):
            assert data is not None, "Unexpected None attribute data."
            if issubclass(attr_type, VBranchGeoData.TipsData) and data is None:
                data = attr_type.create_empty()
            attr[i] = data

        if INTEGRITY_CHECK:
            self.check_integrity()

    def _fetch_branch_data(
        self, attr_name: VBranchGeoDataKey, attr_type: Optional[VBranchGeoData.Type] = None, *, emplace: bool = False
    ) -> list[Optional[VBranchGeoDataBase]]:
        """Fetch the attribute data of a branch."""
        desc = self._fetch_branch_data_descriptor(attr_name, attr_type, emplace=emplace)
        attr_data = self._branch_data_dict.get(desc.name, None)

        if attr_data is None:
            attr_data = [desc.empty] * self.branch_count
            self._branch_data_dict[desc.name] = attr_data
        return attr_data

    def _fetch_branch_data_descriptor(
        self, attr_name: VBranchGeoDataKey, attr_type: Optional[VBranchGeoData.Type] = None, *, emplace: bool = False
    ) -> VBranchGeoDescriptor:
        """Fetch the attribute descriptor of a branch."""

        attr_desc = VBranchGeoData.Descriptor.parse(attr_name, attr_type)
        attr_type = attr_desc.geo_type

        desc = self._branches_attrs_descriptors.get(attr_desc.name, None)
        if desc is None:
            if emplace:
                if attr_desc.geo_type is None:
                    raise TypeError(f"Creation of attribute {attr_desc.name} failed: impossible to infer its type.")
                self._branches_attrs_descriptors[attr_desc.name] = attr_desc
                return attr_desc
            else:
                raise KeyError(f"Attribute {attr_name} not found.")
        elif attr_type is not None:
            if not issubclass(attr_type, desc.geo_type):
                raise TypeError(f"Invalid type for attribute {attr_name}: {attr_type}. Expected type: {desc.geo_type}.")
        return desc

    def _remove_branch_data(self, attr_name: VBranchGeoDataKey) -> None:
        """Remove an attribute data of a branch."""
        self._branch_data_dict.pop(attr_name, None)
        self._branches_attrs_descriptors.pop(attr_name, None)

    ####################################################################################################################
    #  === BRANCHES GEOMETRIC ATTRIBUTES SPECIALIZATION  ===
    ####################################################################################################################
    def branch_data_by_types(self, attr_type: VBranchGeoData.Type) -> Dict[str, VBranchGeoDataBase]:
        """Return the attributes of a branch.

        Parameters
        ----------
        branch_id : int
            The id of the branch.

        Returns
        -------
        Dict[str, VBranchGeoAttr]
            The attributes of the branch. The keys are the attribute names and the values are the attribute values.
        """
        return {
            str(attr): attr_data
            for attr, attr_data in self._branch_data_dict.items()
            if isinstance(attr_data, attr_type)
        }

    def set_branch_bspline(
        self,
        bspline: BSpline | npt.NDArray[np.float32] | List[BSpline | npt.NDArray[np.float32]],
        branch_id: Optional[int | npt.NDArray[np.int32]] = None,
        name: VBranchGeoDataKey = VBranchGeoData.Fields.BSPLINE,
        *,
        graph_index=True,
        no_check=False,
    ) -> None:
        """Set a BSpline representation of one or more branch.

        Parameters
        ----------
        name : str
            The name of the attribute

        bspline : BSpline | List[BSpline]
            The BSpline representation of the branch. If a list is provided, the length should be the same as the number of branches.

        branch_id : Optional[int  |  npt.NDArray[np.int32]], optional
            The id of the branch(es), by default None

        graph_index : bool, optional
            Whether the branch_id is indexed according to the graph (True) or to the internal representation (False), by default True.

        no_check : bool, optional
            If True, do not check the validity of the attribute data, by default False
        """  # noqa: E501
        if isinstance(bspline, BSpline) or (isinstance(bspline, np.ndarray) and bspline.ndim == 1):
            bspline = [bspline]
        bspline = [VBranchGeoData.BSpline(b) if b is not None else None for b in bspline]
        self.set_branch_data(name, bspline, branch_id, graph_index=graph_index, no_check=no_check)

    @overload
    def branch_bspline(
        self, branch_id: int, attr: VBranchGeoDataKey = VBranchGeoData.Fields.BSPLINE, fill: bool = False
    ) -> BSpline: ...
    @overload
    def branch_bspline(
        self,
        branch_id: Optional[Indices] = None,
        attr: VBranchGeoDataKey = VBranchGeoData.Fields.BSPLINE,
        fill: bool = False,
    ) -> List[BSpline]: ...
    def branch_bspline(
        self,
        branch_id: Optional[int | Indices] = None,
        attr: VBranchGeoDataKey = VBranchGeoData.Fields.BSPLINE,
        fill: bool = False,
    ) -> BSpline | List[BSpline]:
        """Return the BSpline representation of a branch.

        Parameters
        ----------
        branch_id : int | np.ndarray, optional
            The index of the branch(es) to retrieve the BSpline representation.

            If None (by default), return the BSpline representation of all the branches.

        attr : str
            The name of the attribute

        fill : bool, optional
            If True, interpolate missing BSpline data.


        Returns
        -------
        BSpline
            The BSpline representation of the branch.
        """
        # Check that the attribute is a BSpline
        self._fetch_branch_data_descriptor(attr, VBranchGeoData.BSpline)

        # Fetch the data
        if branch_id is None:
            data = self.branch_data(attr)
            is_single = False
            branch_ids = np.arange(len(data))
        else:
            branch_ids, is_single = as_1d_array(branch_id)
            data = self.branch_data(attr, branch_ids)

        empty = BSpline()
        bsplines = [empty if d is None else d.data for d in data]
        if fill:

            def fill_bspline(b_id: int, bspline: BSpline) -> BSpline:
                tips_yx = self.node_coord(self.parent_graph.branch_list[b_id])
                return bspline.interpolate_missing_curves(
                    Point.from_array(tips_yx[0]), Point.from_array(tips_yx[1]), smoothing=0.5
                )

            bsplines = [fill_bspline(b, d) for b, d in zip(branch_ids, bsplines, strict=True)]

        return bsplines[0] if is_single else bsplines

    ####################################################################################################################
    #  === GRAPH MANIPULATION ===
    ####################################################################################################################
    def _sort_internal_node_ids(self) -> None:
        """Sort the nodes by their indices in the graph."""
        if self._nodes_id is None:
            return
        sorted_index = np.argsort(self.node_ids)
        if np.any(np.diff(sorted_index) < 1):
            self._nodes_id = self._nodes_id[sorted_index]
            self._nodes_coord = self._nodes_coord[sorted_index]

    def _reindex_nodes(self, new_node_index: npt.NDArray[np.int_]) -> None:
        """Reindex the nodes in the graph.  # noqa: E501

        Parameters
        ----------
        node_index : np.ndarray
            The new index of the nodes: node_index[old_index] = new_index or -1 if the node should be removed.
        """
        if self._nodes_id is None:
            if len(new_node_index) > len(self._nodes_coord):
                # If the new index is longer than the current number of nodes, use sparse indexing
                self._nodes_id = self.node_ids
                return self._reindex_nodes(new_node_index)
            else:
                self._nodes_coord = reorder_array(self._nodes_coord, new_node_index)
        else:
            self._nodes_id = reorder_array(self._nodes_id, new_node_index)
            self._sort_internal_node_ids()

    def _append_nodes(self, node_coords: FloatPairArray) -> None:
        """Append new nodes to the graph.

        Parameters
        ----------
        node_coords : np.ndarray
            The coordinates of the new nodes to append.
        """
        if self._nodes_id is not None:
            raise NotImplementedError("Appending nodes is not supported for indexed graphs.")
        self._nodes_coord = np.vstack([self._nodes_coord, node_coords])

    def _drop_nodes(self, node_ids: Iterable[int], *, graph_index=True):
        """Remove nodes from the graph.

        Parameters
        ----------
        node_ids : np.ndarray
            The ids of the nodes to remove.
        """
        if self._nodes_id is None:
            self._nodes_coord = np.delete(self._nodes_coord, node_ids, axis=0)
        else:
            internal_ids = self._graph_to_internal_nodes_index(node_ids, graph_index=graph_index)
            self._nodes_id = np.delete(self._nodes_id, internal_ids)
            self._nodes_coord = np.delete(self._nodes_coord, internal_ids, axis=0)

    def _merge_nodes(
        self,
        cluster: npt.NDArray[np.int_] | List[int],
        weight: Optional[npt.NDArray[np.float32] | List[float]] = None,
        *,
        graph_index=True,
    ):
        """Merge the nodes of a cluster into a single node.

        Parameters
        ----------
        cluster : np.ndarray
            The indices of the nodes to merge.
        """
        if weight is None:
            weight = np.ones(len(cluster), dtype=np.float32)
        else:
            weight = np.asarray(weight, dtype=np.float32)

        cluster = np.asarray(cluster)
        cluster = self._graph_to_internal_nodes_index(cluster, graph_index, sort_index=True, check_valid=False)
        weight = weight[cluster >= 0]
        cluster = cluster[cluster >= 0]

        weight_total = np.sum(weight)
        if weight_total == 0:
            weight = np.ones(len(cluster)) / len(cluster)
        else:
            weight /= weight_total

        new_node = np.sum(weight[:, None] * self.node_coord(cluster, graph_index=False), axis=0)
        self._nodes_coord[cluster[0]] = new_node

    def _sort_internal_branch_ids(self) -> None:
        """Sort the branches by their indices in the graph."""
        if self._branches_id is None:
            return
        sorted_index = np.argsort(self.branch_ids)
        if np.any(np.diff(sorted_index) < 1):
            self._branches_id = self.branch_ids[sorted_index]
            self._branch_curve = [self._branch_curve[i] for i in sorted_index]
            self._branch_data_dict = {k: [v[i] for i in sorted_index] for k, v in self._branch_data_dict.items()}

    def _reindex_branches(self, new_branch_index: npt.NDArray[np.int_]) -> None:
        """Reindex the branches in the graph.

        Parameters
        ----------
        branch_index : np.ndarray
            The new index of the branches. branch_index[old_index] = new_index or -1 if the branch should be removed.
        """
        if self._branches_id is None:
            if len(new_branch_index) > len(self._branch_curve):
                # If the new index is longer than the current number of branch, use sparse indexing
                self._branches_id = self.branch_ids
                return self._reindex_branches(new_branch_index)
            else:
                inverted_index = invert_lookup(new_branch_index)
                self._branch_curve = [self._branch_curve[i] for i in inverted_index]
                self._branch_data_dict = {k: [v[i] for i in inverted_index] for k, v in self._branch_data_dict.items()}
        else:
            self._branches_id = new_branch_index[self._branches_id]
            branch_to_delete = self._branches_id < 0
            if np.any(branch_to_delete):
                self._branches_id = self._branches_id[~branch_to_delete]
                self._branch_curve = [curve for i, curve in enumerate(self._branch_curve) if not branch_to_delete[i]]
                self._branch_data_dict = {
                    k: [v for i, v in enumerate(v) if not branch_to_delete[i]]
                    for k, v in self._branch_data_dict.items()
                }
            self._sort_internal_branch_ids()

    def _append_nodes_and_branches(self, other: VGeometricData, apply_domain: bool = True):
        """Append nodes and branches from another graph to the current graph.

        Parameters
        ----------
        other : VBranchGeoData
            The other graph to append branches from.
        branch_id : Optional[np.ndarray], optional
            The ids of the branches to append, by default None (all branches are appended).
        """

        if apply_domain:
            delta_yx = other.domain.top_left - self.domain.top_left
            other = other.transform(Translation(delta_yx.numpy()))

        # Append nodes
        self._nodes_coord = np.concatenate([self._nodes_coord, other._nodes_coord])

        # Append branches
        if self._branches_id is not None:
            raise NotImplementedError("Appending branches is not supported for indexed graphs.")

        self._branch_curve += other._branch_curve
        for attr_name, attr_data in self._branch_data_dict.items():
            if attr_name in other._branch_data_dict:
                attr_data += other._branch_data_dict[attr_name]
            else:
                # Ensure the attribute is registered
                attr_data += [self._branches_attrs_descriptors[attr_name].empty] * len(other._branch_curve)

        if INTEGRITY_CHECK:
            self.check_integrity()

    def _append_empty_branches(self, n: int):
        """Append empty branches to the graph geometry data.

        Parameters
        ----------
        n : int
            The number of branches to add.
        """
        if self._branches_id is None:
            self._branch_curve += [EMPTY_CURVE] * n

            for attr_name, attr in self._branch_data_dict.items():
                empty = self._branches_attrs_descriptors[attr_name].empty
                attr += [empty] * n
            if INTEGRITY_CHECK:
                self.check_integrity()
        else:
            raise NotImplementedError("Appending branches is not supported for indexed graphs.")

    def _append_branch_duplicates(self, branches_id: Int1DArrayLike, new_branches_id: Int1DArrayLike):
        """Append duplicates of the branches to the graph geometry data.

        Parameters
        ----------
        branches_id : npt.ArrayLike[int]
            The ids of the branches to duplicate.
        """
        branches_id = np.atleast_1d(branches_id).flatten().astype(int)
        new_branches_id = np.atleast_1d(new_branches_id).flatten().astype(int)
        assert len(branches_id) == len(new_branches_id), "Invalid number of new branches ids."

        if INTEGRITY_CHECK:
            self.check_integrity(ignore_parent_data=True)

        if self._branches_id is None:
            B = len(self._branch_curve)
            assert (bmin := branches_id.min()) >= 0, f"Invalid branches index: {bmin}"
            assert (bmax := branches_id.max()) < B, f"Invalid branches index: {bmax} (branches count: {B})"
            unique_new_ids = np.unique(new_branches_id)
            if (
                len(unique_new_ids) != len(new_branches_id)
                or unique_new_ids[0] != B
                or unique_new_ids[-1] != B + len(new_branches_id) - 1
            ):
                # TODO: switch to sparse indexing
                raise NotImplementedError("Invalid new branches ids.")
            new_ids = new_branches_id - B
            self._branch_curve += [self._branch_curve[branches_id[i]] for i in new_ids]

            for attr in self._branch_data_dict.values():
                attr += [attr_v.copy() if (attr_v := attr[branches_id[i]]) is not None else None for i in new_ids]

            if INTEGRITY_CHECK:
                self.check_integrity()
        else:
            raise NotImplementedError

    def _merge_branches(self, consecutive_branches: Iterable[int], *, graph_index=True):
        """Merge consecutive branches into a single branch.

        Parameters
        ----------
        consecutive_branches : np.ndarray
            The indices of the branches to merge.

        graph_index : bool, optional
            Whether the branch_id is indexed according to the graph (True) or to the internal representation (False), by default True.
        """  # noqa: E501
        if graph_index:
            consecutive_branches = self._graph_to_internal_branch_ids(consecutive_branches, sort_index=False)

        if INTEGRITY_CHECK:
            self.check_integrity()

        branch0 = consecutive_branches[0]

        branches_curve = self.branch_curve(consecutive_branches)
        not_none_branches_curve = [b for b in branches_curve if b is not EMPTY_CURVE and b is not None]
        if len(not_none_branches_curve):
            self._branch_curve[branch0] = readonly(np.concatenate(not_none_branches_curve))

        ctx = self._geodata_edit_ctx(branch0)
        ctx = ctx.set_info(curves=branches_curve)
        for attr_name, attr_data in self._branch_data_dict.items():
            attr_type = self._branches_attrs_descriptors[attr_name].geo_type
            if any(attr_data[b] is not None for b in consecutive_branches):
                attr_data[branch0] = attr_type.merge(
                    [attr_data[branch_id] for branch_id in consecutive_branches], ctx.set_name(attr_name)
                )

        if INTEGRITY_CHECK:
            self.check_integrity()

    def _split_branch(
        self,
        branch_id: int,
        split_curve_ids: npt.NDArray,
        split_coord: Optional[npt.NDArray],
        new_branch_ids: List[int] | npt.NDArray[np.int_],
        new_node_ids: List[int] | npt.NDArray[np.int_],
    ):
        """Split a branch into two branches at a given position.

        Parameters
        ----------
        branch_id : int
            The graph index of the branch to split.

        split_curve_ids : npt.NDArray
            The the curve index where the branch must be splitted as an array of shape (n,)

        split_coord : npt.NDArray
            The position where the split nodes should be added as an array of shape (n, 2).
            If None, the split position is inferred from the curve.

        new_branch_ids : List[int]
            The ids of the new branches. The length of the list should be equal to the length of splits_positions.

        new_node_ids : List[int]
            The ids of the new nodes. The length of the list should be equal to the length of splits_positions.

        """  # noqa: E501
        if len(new_branch_ids) == 0:
            return
        assert bool(len(split_curve_ids) == len(new_branch_ids)), (
            f"Invalid number of new branch ids. (Got {len(new_branch_ids)} but expected {len(split_curve_ids)})"
        )
        assert bool(len(split_curve_ids) == len(new_node_ids)), (
            f"Invalid number of new node ids. (Got {len(new_node_ids)} but expected {len(split_curve_ids)})"
        )

        internal_id = int(self._graph_to_internal_branch_ids(branch_id))
        curve = self.branch_curve(internal_id)

        assert split_curve_ids.ndim == 1, "Invalid split positions."
        n_splits = len(split_curve_ids)

        if split_coord is not None:
            assert split_coord.shape == (n_splits, 2), "Invalid split coordinates."

        # === Split the curves ===
        if len(curve) == 0:
            split_bin = [0 for _ in range(n_splits + 2)]
            new_curves = [np.empty((0, 2), dtype=int) for _ in range(n_splits + 1)]

            nodes = self.parent_graph.branch_list[branch_id]
            n1, n2 = self.node_coord(nodes)
            if split_coord is None:
                a = np.linspace(0, 1, n_splits + 2, endpoint=True)[1:-1, None]
                explicit_split_coord = a * n1[None, :] + (1 - a) * n2[None, :]
            else:
                explicit_split_coord = split_coord
        else:
            #: The indices in the curve where it should be splitted (including the start and last index of the curve)
            split_bin = [0]
            for split_pos in split_curve_ids:
                if 0 < split_pos < 1:
                    split_pos = int(split_pos * len(curve))
                split_bin.append(int(split_pos))
            split_bin.append(len(curve))
            new_curves = [curve[start_id:end_id] for start_id, end_id in itertools.pairwise(split_bin)]
            explicit_split_coord = curve[split_bin[1:-1]] if split_coord is None else split_coord

        # === Split the curve and add new curves ===
        self._branch_curve[internal_id] = new_curves[0]
        self._branch_curve.extend(new_curves[1:])
        if self._branches_id is not None:
            self._branches_id = np.concatenate([self._branches_id, new_branch_ids])

        # === Add the new nodes ===
        self._nodes_coord = np.concatenate([self._nodes_coord, np.array(explicit_split_coord, dtype=np.float32)])
        if self._nodes_id is not None:
            self._nodes_id = np.concatenate([self._nodes_id, new_node_ids])

        # === Split the branch geo data ===
        ctx = self._geodata_edit_ctx(internal_id)
        ctx.info["new_curves"] = new_curves
        for attr_name, attr_data in self._branch_data_dict.items():
            attr = attr_data[internal_id]
            if attr is not None:
                branch_attrs = attr.split(split_bin, ctx.set_name(attr_name))
                assert len(branch_attrs) == len(new_branch_ids) + 1, "Invalid number of attributes after split."
                self._branch_data_dict[attr_name][internal_id] = branch_attrs[0]
                self._branch_data_dict[attr_name].extend(branch_attrs[1:])
            else:
                self._branch_data_dict[attr_name].extend([None] * (len(new_branch_ids)))

        self._sort_internal_branch_ids()
        self._sort_internal_node_ids()

        if INTEGRITY_CHECK:
            self.check_integrity(ignore_parent_data=True)

    def _flip_branch_direction(self, branch_id: int | Iterable[int]):
        """Swap the direction of a branch.

        Parameters
        ----------
        branch_id : int | np.ndarray
            The id of the branch to swap.
        """
        ids, _ = as_1d_array(branch_id)
        internal_ids = self._graph_to_internal_branch_ids(ids)

        if INTEGRITY_CHECK:
            self.check_integrity()

        for branch_id in internal_ids:
            ctx = self._geodata_edit_ctx(branch_id)
            branch_curve = self._branch_curve[branch_id]
            if branch_curve is not None:
                flipped_curve = readonly(np.flip(branch_curve, axis=0))
                self._branch_curve[branch_id] = flipped_curve
            for attr_name, attr_data in self._branch_data_dict.items():
                attr = attr_data[branch_id]
                if attr is not None:
                    attr_data[branch_id] = attr.flip(ctx.set_name(attr_name))

        if INTEGRITY_CHECK:
            self.check_integrity()

    def check_integrity(
        self, ignore_parent_data=False, on_error: Literal["raise", "warn", "skip", "report"] | None = None
    ) -> bool:
        """Check the integrity of the geometric fields."""

        check_parent = not ignore_parent_data and self._parent_graph is not None
        if on_error is None:
            on_error = INTEGRITY_CHECK or "skip"

        # === NODES CHECK ===
        if check_parent and self._nodes_coord.shape[0] != self.parent_graph.node_count:
            msg = (
                f"Geometric data integrity check failed:\n"
                f" - Invalid number of nodes: {self._nodes_coord.shape[0]} (expected: {self.parent_graph.node_count})\n"
            )
            if on_error == "raise":
                raise RuntimeError(msg)
            else:
                warnings.warn(msg, stacklevel=2)
            return False

        # === BRANCHES CURVES CHECK ===
        invalid = []
        for i, c in enumerate(self._branch_curve):
            if not c.ndim == 2 or c.shape[1] != 2:
                invalid.append(f" - Invalid curve shape for branch {i}: {c.shape} (expected: (n, 2))")

        if len(invalid):
            msg = "Geometric data integrity check failed:\n" + "\n".join(invalid)
            if on_error == "raise":
                raise RuntimeError(msg)
            else:
                warnings.warn(msg, stacklevel=2)
            return False

        if check_parent and len(self._branch_curve) != self.parent_graph.branch_count:
            msg = (
                "Geometric data integrity check failed:\n"
                f" - Invalid number of branches curves: {len(self._branch_curve)} "
                f"(expected: {self.parent_graph.branch_count})\n"
            )
            if on_error == "raise":
                raise RuntimeError(msg)
            else:
                warnings.warn(msg, stacklevel=2)
            return False

        # === BRANCH GEOMETRIC DATA CHECK ===
        invalid = {}
        for attr_name, attr in self._branches_attrs_descriptors.items():
            attr_data = self._branch_data_dict[attr_name]
            if check_parent and len(attr_data) != self.parent_graph.branch_count:
                msg = (
                    f"Geometric data integrity check failed:\n"
                    f" - Invalid number of branches for attribute '{attr_name}': {len(attr_data)} "
                    f"(expected: {self.parent_graph.branch_count})\n"
                )
                if on_error == "raise":
                    raise RuntimeError(msg)
                else:
                    warnings.warn(msg, stacklevel=2)
                return False

            for i, data in enumerate(attr_data):
                if data is None:
                    invalid.setdefault(attr_name, {})[i] = "Missing data"
                    continue
                ctx = self._geodata_edit_ctx(i, attr_name)
                is_invalid = data.is_invalid(ctx=ctx)
                if is_invalid:
                    if on_error == "raise":
                        raise ValueError(f"Invalid attribute '{attr_name}' of branch {i}.\n{is_invalid}.")
                    invalid.setdefault(attr_name, {})[i] = is_invalid
        if len(invalid):
            msg = "Geometric data integrity check failed:\n"
            for attr_name, attr_invalid in invalid.items():
                msg += f" --- Attribute '{attr_name}' --- \n"
                for branch_id, reason in attr_invalid.items():
                    msg += f"    - Branch {branch_id}: {reason}\n"
            if on_error == "raise":
                raise RuntimeError(msg)
            else:
                warnings.warn(msg, stacklevel=2)
            return False

        return True

    def clear_branch_gdata(self, branch_id: Int1DArrayLike) -> None:
        """Clear the geometric data of a branch.

        Parameters
        ----------
        branch_id : Int1DArrayLike
            The index of the branch(es) to clear.

        """
        ids, _ = as_1d_array(branch_id)
        for i in ids:
            self._branch_curve[i] = EMPTY_CURVE
        for attr_name, attr in self._branches_attrs_descriptors.items():
            attr_data = self._branch_data_dict[attr_name]
            for i in ids:
                attr_data[i] = attr.empty

    def clear_attribute(
        self, *attr: VBranchGeoDataKey, all_except: VBranchGeoDataKey | Iterable[VBranchGeoDataKey] | None = None
    ) -> None:
        """Clear a geometric attribute from all branches.

        Parameters
        ----------
        attr_name : VBranchGeoDataKey | Iterable[VBranchGeoDataKey]
            The name of the attribute(s) to clear.
        """
        attr_ = set(attr)
        if len(attr) == 0 and all_except is not None:
            attr_ = set(self._branch_data_dict.keys())
        if all_except is not None:
            attr_ -= {all_except} if VBranchGeoData.isinstance_key(all_except) else set(all_except)
        attr_ &= set(self._branch_data_dict.keys())

        for name in attr_:
            self._remove_branch_data(name)

    def transform(
        self,
        projection: Transform,
        *,
        warped_domain: Rect | Literal["full", "same"] = "full",
        inplace: bool = False,
    ) -> VGeometricData:
        """Apply a transformation to the geometric data.

        All branch geometric attributes that can't be transformed will be removed.

        Parameters
        ----------
        transform : FundusProjection
            The transformation to apply.

        warped_domain : Rect or 'full' or 'same', optional
            The domain to use for the warped coordinates. If 'full' (by default), the full domain of the projection is used. If 'same', the same domain as the original coordinates is used. If a Rect is given, it is used as the domain for the warped coordinates.

        inplace : bool, optional
            If True, apply the transformation in place, by default False.

        Returns
        -------
        VGeometricData
            The transformed geometric data.
        """  # noqa: E501
        if not inplace:
            self = self.copy()

        if warped_domain == "full":
            new_domain = projection.transform_domain(self._domain)
        elif warped_domain == "same":
            new_domain = self._domain
        else:  # warped_domain is a Rect
            new_domain = warped_domain

        projection = Translation(-new_domain.top_left) @ projection @ Translation(self._domain.top_left)
        self._domain = new_domain

        all_curves = np.concatenate(
            [curve for curve in self._branch_curve if curve is not None and len(curve) > 0], axis=0
        )
        all_curves_transformed = projection.transform(all_curves).astype(np.float64)

        self._nodes_coord = projection.transform(self._nodes_coord).astype(np.float64)
        start_idx = 0
        for branch_id, curve in enumerate(self._branch_curve):
            if curve is None or len(curve) == 0:
                continue

            prev_curve_delta_d = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            curve = all_curves_transformed[start_idx : start_idx + len(curve)]
            new_curve_delta_d = np.linalg.norm(np.diff(curve, axis=0), axis=1)

            local_scale = new_curve_delta_d / (prev_curve_delta_d + 1e-8)
            local_scale = np.concatenate([local_scale[:1], local_scale, local_scale[-1:]])
            local_scale = (local_scale[1:] + local_scale[:-1]) / 2
            curve = np.round(curve).astype(np.int_)

            if not isinstance(projection, Translation):
                cleaned_curve, new_id = remove_consecutive_duplicates(curve, return_index=True)
                if np.all(new_id == np.arange(len(new_id))):
                    cleaned_curve, new_id = curve, None
            else:
                cleaned_curve, new_id = curve, None

            ctx = self._geodata_edit_ctx(branch_id)
            ctx = ctx.set_info(local_scale=local_scale, transformed_curve=curve)

            for attr_name, attr in ctx.geodata_attrs.items():
                try:
                    ctx_attr = ctx._replace(attr_name=attr_name)
                    attr = attr.transform(projection, ctx_attr)
                    if new_id is not None:
                        attr = attr.resample(new_id.astype(np.int_), ctx_attr)
                    self._branch_data_dict[attr_name][branch_id] = attr
                except NotImplementedError:
                    self._remove_branch_data(attr_name)
                    break

            self._branch_curve[branch_id] = readonly(cleaned_curve)
            start_idx += len(curve)

        return self

    def resample_branch_curve(self, branch_id: int, idx: npt.NDArray[np.int_]) -> None:
        """Resample a branch curve to only keep the points at the given indices.

        Parameters
        ----------
        branch_id : int
            The id of the branch to resample.

        idx : npt.NDArray[np.int_]
            The indices of the points to keep.
        """
        curve = self._branch_curve[branch_id][idx, :]
        ctx = self._geodata_edit_ctx(branch_id)
        for attr_name, attr in ctx.geodata_attrs.items():
            ctx_attr = ctx._replace(attr_name=attr_name)
            self._branch_data_dict[attr_name][branch_id] = attr.resample(idx, ctx_attr)
        self._branch_curve[branch_id] = curve

    def sample_branch_curves(self, n_points: int) -> None:
        """Resample all branch curves to have a fixed number of points.

        Parameters
        ----------
        n_points : int
            The number of points to sample each branch curve to.
        """
        for branch_id, curve in enumerate(self._branch_curve):
            if (N := len(curve)) <= n_points:
                continue
            ids = np.linspace(0, N - 1, n_points, endpoint=True, dtype=np.int_)
            self.resample_branch_curve(branch_id, ids)

    ####################################################################################################################
    #  === VISUALISATION TOOLS ===
    ####################################################################################################################
    def jppype_bspline_tangent(
        self, bspline_name: VBranchGeoDataKey = VBranchGeoData.Fields.BSPLINE, scaling=1, normalize=False
    ):
        from jppype.layers import LayerQuiver

        points = []
        tangents = []
        bsplines = self.branch_bspline(attr=bspline_name)
        for bspline in bsplines:
            if bspline is None or not len(bspline):
                continue
            p, t = bspline.intermediate_points(return_tangent=True)
            nan = np.isnan(p).any(axis=1) | np.isnan(t).any(axis=1)
            p = p[~nan]
            t = t[~nan]
            points.extend(p)
            if normalize:
                t_norm = np.linalg.norm(t, axis=1)
                t[t_norm != 0] /= t_norm[t_norm != 0][:, None]
                t *= 10
            tangents.extend(t)

        return LayerQuiver(
            np.stack(points), np.stack(tangents) * scaling, self.domain, "view_sqrt" if normalize else "scene"
        )

    def jppype_branch_tip_tangent(
        self,
        attr: VBranchGeoDataKey = VBranchGeoData.Fields.TIPS_TANGENT,
        scaling=1,
        normalize=False,
        invert_direction: Optional[bool | npt.NDArray[np.bool_] | Literal["tree"]] = None,
        show_only: Optional[npt.NDArray[np.bool_] | Literal["endpoints", "junctions"]] = None,
    ):
        """Draw the tangents of the branch tips in a jppype LayerQuiver.

        Parameters
        ----------
        attr : str, optional
            The name of the attribute containing the tangent data, by default VBranchGeoData.Fields.TIPS_TANGENT

        scaling : float, optional
            The factor by which the tangent vectors are to be scaled, by default 1.

        normalize : bool, optional
            If True, the tangent vectors are normalized to unit length, by default False

        invert_direction : bool | np.ndarray | str, optional
            Reverse the direction of all or some of the tangents, by default None

            - If None or False, the direction of the tangents will not be inverted.
            - If True, invert the direction of all the tangents.
            - If an array is provided, it should have the same length as the number of branches.
            - If "tree", the direction of the tangents will be inverted according to the tree structure of the graph.

        show_only : np.ndarray | str, optional
            An array of boolean values indicating which branch tips should be displayed, by default None

            - If None, all the branch tips will be displayed.
            - If "endpoints", only the branch tips that are endpoints will be displayed.
            - If "junctions", only the branch tips that are junctions will be displayed.
            - If an array is provided, it should have the same length as the number of branches.

        Returns
        -------
        LayerQuiver
            The jppype LayerQuiver object containing the branch tips tangents.
        """

        from jppype.layers import LayerQuiver

        if isinstance(show_only, str):
            if show_only == "endpoints":
                show_only = np.isin(self.parent_graph.branch_list, self.parent_graph.endpoint_nodes())
            elif show_only == "junctions":
                show_only = np.isin(self.parent_graph.branch_list, self.parent_graph.junction_nodes())
            else:
                raise ValueError(f"Invalid value for show_only: {show_only}")
        if show_only is not None and not show_only.any():
            return LayerQuiver(np.empty((0, 2)), np.empty((0, 2)), self.domain, "view_sqrt" if normalize else "scene")

        if isinstance(invert_direction, str):
            if invert_direction == "tree":
                from .vtree import VTree

                assert isinstance(self.parent_graph, VTree), "The parent graph is not a tree."
                invert_direction = np.tile([False, True], self.branch_count).reshape(-1, 2)
                invert_direction[~self.parent_graph.branch_dirs()] = [True, False]
            else:
                raise ValueError(f"Invalid value for invert_direction: {invert_direction}")

        branch_list = self.parent_graph.branch_list

        arrows_p = []
        arrows_v = []
        curves = self.branch_curve()
        tangents = self.tip_tangent(attr=attr)
        for i, curve, tangent in zip(self.branch_ids, curves, tangents, strict=True):
            if show_only is not None and not show_only[i].any():
                continue
            if curve is None or len(curve) <= 1 or (curve[0] == curve[-1]).all():
                tail_p, head_p = self.node_coord(branch_list[i])
            else:
                tail_p = curve[0]
                head_p = curve[-1]

            # tail_to_head = (head_p - tail_p).astype(float)
            # tail_to_head /= np.linalg.norm(tail_to_head)
            # if tangent is None:
            #     tail_t = tail_to_head
            #     head_t = -tail_t
            # else:
            #     tangent = tangent.data
            #     tail_t, head_t = tangent[0], tangent[-1]
            #     if np.isnan(tail_t).any() or np.sum(tail_t) == 0:
            #         tail_t = tail_to_head
            #     if np.isnan(head_t).any() or np.sum(head_t) == 0:
            #         head_t = -tail_to_head
            tail_t, head_t = tangent
            if invert_direction is not None and invert_direction is not False:
                if invert_direction is True:
                    tail_t = -tail_t
                    head_t = -head_t
                else:
                    if invert_direction[i, 0]:
                        tail_t = -tail_t
                    if invert_direction[i, 1]:
                        head_t = -head_t

            if show_only is not None:
                if show_only[i, 0]:
                    arrows_p.append(tail_p)
                    arrows_v.append(tail_t)
                if show_only[i, 1]:
                    arrows_p.append(head_p)
                    arrows_v.append(head_t)
            else:
                arrows_p.extend([tail_p, head_p])
                arrows_v.extend([tail_t, head_t])

        return LayerQuiver(
            np.stack(arrows_p), np.stack(arrows_v) * scaling, self.domain, "view_sqrt" if normalize else "scene"
        )

    def jppype_draw_branch_tip_calibre(self, calibre: VBranchGeoDataKey = VBranchGeoData.Fields.TIPS_CALIBRE):
        from jppype.layers import LayerScatter

        calibres = self.branch_data(calibre)
        points = []
        for branch_id, calibre in enumerate(calibres):
            if calibre is not None:
                points.append(self.branch_curve(branch_id)[0])
        return LayerScatter(np.stack(points), self.domain, "scene")
