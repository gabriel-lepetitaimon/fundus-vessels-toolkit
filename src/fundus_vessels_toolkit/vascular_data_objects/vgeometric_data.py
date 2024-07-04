from __future__ import annotations

from copy import copy, deepcopy
from typing import Dict, Generator, Iterable, List, Optional, Tuple, TypeAlias

from fundus_vessels_toolkit.utils.lookup_array import invert_lookup
import numpy as np
import numpy.typing as npt

from fundus_vessels_toolkit.utils.math import np_find_sorted

from ..utils.geometric import Point, Rect
from .vbranch_geo_attr import VBranchCurveAttr, VBranchGeoAttr

BranchesAttributes: TypeAlias = Dict[str, List[Optional[np.ndarray | VBranchGeoAttr]]]


class VGeometricData:
    """``VGeometricData`` is a class that stores the geometric data of a vascular graph."""

    def __init__(
        self,
        nodes_coord: npt.NDArray[np.uint32],
        branches_curve: List[npt.NDArray[np.uint32]],
        domain: Rect,
        branches_attr: Optional[BranchesAttributes] = None,
        nodes_id: Optional[npt.NDArray[np.uint32]] = None,
        branches_id: Optional[npt.NDArray[np.uint32]] = None,
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
        self._domain: Rect = Rect.from_tuple(domain)

        # Check and define nodes coordinates and index
        nodes_coord = np.asarray(nodes_coord, dtype=np.float32)
        assert (
            nodes_coord.ndim == 2 and nodes_coord.shape[1] == 2
        ), "The node coordinates should be a 2D array with shape (n_nodes, 2)."
        self._nodes_coord = nodes_coord

        if nodes_id is not None:
            nodes_id = np.asarray(nodes_id, dtype=np.uint32)
            assert nodes_id.ndim == 1 and nodes_id.shape[0] == nodes_coord.shape[0], (
                "The node ids should be a 1D array with the same length as the number of nodes. "
                f"Got shape {nodes_id.shape} instead of {nodes_coord.shape}."
            )
        self._nodes_id = nodes_id

        # Check and define branches curves and index
        assert all(
            isinstance(curve, np.ndarray) and curve.ndim == 2 and curve.shape[1] == 2 for curve in branches_curve
        ), "The branch curves should be a list of 2D arrays with shape (n_points, 2)."
        self._branches_curve = branches_curve

        if branches_id is not None:
            branches_id = np.asarray(branches_id, dtype=np.uint32)
            assert branches_id.ndim == 1 and branches_id.shape[0] == len(branches_curve), (
                "The branch ids should be a 1D array with the same length as the number of branches. "
                f"Got shape {branches_id.shape} instead of {len(branches_curve)}."
            )
        self._branches_id = branches_id

        # Check and define branches curves attributes
        if branches_attr is None or not len(branches_attr):
            self._branches_attrs = {}
        else:
            for attr_name, attr in branches_attr.items():
                for branch_id, attr_data in enumerate(attr):
                    if isinstance(attr_data, np.ndarray):
                        attr_data = VBranchCurveAttr(attr_data)
                    elif not isinstance(attr_data, VBranchGeoAttr):
                        raise ValueError(
                            f"Invalid type for attribute {attr_name} of branch {branch_id}. "
                            f"Expected np.ndarray or VBranchGeoAttr, got {type(attr_data)}."
                        )
                    is_invalid = attr_data.is_invalid(self._branch_ctx(branch_id))
                    if is_invalid:
                        raise ValueError(f"Invalid attribute {attr_name} of branch {branch_id}.\n{is_invalid}.")

            self._branches_attrs: Dict[str, List[VBranchGeoAttr]] = branches_attr

        # Ensure that the nodes and branches are sorted by their index in the graph
        self._sort_internal_nodes_index()
        self._sort_internal_branches_index()

    @classmethod
    def from_dict(
        cls,
        nodes_coord: Dict[int, Point],
        branches_curve: Dict[int, npt.NDArray[np.uint32]],
        domain: Rect,
        branches_attr: Optional[Dict[str, Dict[int, np.ndarray | VBranchGeoAttr]]] = None,
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
        nodes_coord = np.array(list(nodes_coord.values()), dtype=np.float32)
        branches_id = np.array(list(branches_curve.keys()), dtype=np.uint32)
        branches_curve = list(branches_curve.values())

        if branches_attr is not None:
            branches_attr = {
                attr_name: [attr.get(i, None) for i in branches_id] for attr_name, attr in branches_attr.items()
            }

        return cls(nodes_coord, branches_curve, domain, branches_attr, nodes_id, branches_id)

    def copy(self) -> VGeometricData:
        """Return a copy of the object."""
        other = copy(self)
        other._nodes_coord = self._nodes_coord.copy()
        other._branches_curve = [v.copy() for v in self._branches_curve]
        other._branches_attrs = deepcopy(self._branches_attrs)
        return other

    ####################################################################################################################
    #  === BASE PROPERTIES ===
    ####################################################################################################################
    @property
    def domain(self) -> Rect:
        """Return the domain of the geometric data."""
        return self._domain

    @property
    def nodes_count(self) -> int:
        """Return the number of nodes whose geometric data are stored in this object."""
        return self._nodes_coord.shape[0]

    @property
    def nodes_id(self) -> np.ndarray:
        """Return the indexes of the nodes as indexed in the graph.
        In other word, a mapping of nodes index from they index in this VGeometricData object to their index in the original graph."""  # noqa: E501
        return np.arange(self.nodes_count) if self._nodes_id is None else self._nodes_id

    def _graph_to_internal_nodes_index(
        self, index: npt.ArrayLike, graph_indexing=True, *, index_sorted=False, sort_index=True, check_valid=False
    ) -> npt.NDArray[np.int_]:
        """Convert the index of the nodes in the graph to their index in the internal representation."""
        index = np.asarray(index)
        if not index_sorted and sort_index:
            index.sort()
            index_sorted = True
        if graph_indexing and self._nodes_id is not None:
            internal_index = np_find_sorted(index, self._nodes_id, assume_keys_sorted=index_sorted)
            assert check_valid or np.all(internal_index >= 0), f"Nodes {index[internal_index < 0]} not found."
            return internal_index
        else:
            assert check_valid or np.all([0 <= index, index < self.nodes_count]), f"Invalid node index {index}."
            return index

    def nodes_coord(
        self, id: Optional[int | npt.NDArray[np.int32]] = None, *, graph_indexing=True
    ) -> Point | np.ndarray:
        """Return the coordinates of the nodes in the graph."""

        if id is None:
            return self._nodes_coord

        is_single = np.isscalar(id)
        if is_single:
            id = np.asarray([id])

        if isinstance(id, np.ndarray):
            internal_id = self._graph_to_internal_nodes_index(id, graph_indexing=graph_indexing)
            return self._nodes_coord[internal_id] if not is_single else self._nodes_coord[internal_id[0]]
        else:
            raise TypeError("Invalid type for node id.")

    @property
    def branches_count(self) -> int:
        """Return the number of branches whose geometric data are stored in this object."""
        return len(self._branches_curve)

    @property
    def branches_id(self) -> np.ndarray:
        """Return the indexes of the branches as indexed in the graph.
        In other word, a mapping of branches index from they index in this VGeometricData object to their index in the original graph."""  # noqa: E501
        return np.arange(self.branches_count) if self._branches_id is None else self._branches_id

    def _graph_to_internal_branches_index(
        self, index: npt.ArrayLike, *, graph_index=True, index_sorted=False, sort_index=False, check_valid=True
    ) -> npt.NDArray[np.int_]:
        """Convert the index of the branches in the graph to their index in the internal representation."""
        index = np.asarray(index)
        if not index_sorted and sort_index:
            index.sort()
            index_sorted = True
        if graph_index and self._branches_id is not None:
            internal_index = np_find_sorted(index, self._branches_id, assume_keys_sorted=index_sorted)
            assert not check_valid or np.all(internal_index >= 0), f"Branch {index[internal_index < 0]} not found."
            return internal_index
        else:
            assert not check_valid or np.all(
                [0 <= index, index < self.branches_count]
            ), f"Invalid branch index {index}."
            return index

    def branches_curve(
        self, id: Optional[int | npt.NDArray[np.int32]] = None, *, graph_indexing=True
    ) -> npt.NDArray[np.int_] | List[npt.NDArray[np.int_]] | Generator[npt.NDArray[np.int_]]:
        """Return the coordinates of the pixels that compose the branches of the graph."""
        if id is None:
            return self._branches_curve

        is_single = np.isscalar(id)
        if is_single:
            id = np.asarray([id])

        if isinstance(id, np.ndarray):
            internal_id = self._graph_to_internal_branches_index(id, graph_index=graph_indexing, check_valid=False)
            if is_single:
                return self._branches_curve[internal_id[0]]
            return [self._branches_curve[i] if i > 0 else None for i in internal_id]

        else:
            raise TypeError("Invalid type for branches index.")

    def _branch_ctx(self, id: int) -> Dict[str, any]:
        """Return the context of a branch."""
        return {"curve": self.branches_curve(id)}

    ####################################################################################################################
    #  === COMPUTABLE GEOMETRIC PROPERTIES ===
    ####################################################################################################################
    def branches_label_map(self) -> npt.NDArray[np.int_]:
        """Return a label map of the branches.

        Returns
        -------
        np.ndarray
            An array of the same shape as the image where each pixel is labeled with the id of the branch it belongs to.
        """
        branches_label_map = np.zeros(self.domain.shape, dtype=np.int_)
        for branch_id, branch_curve in zip(self.branches_id, self._branches_curve, strict=True):
            branches_label_map[branch_curve[:, 0] - self.domain.top, branch_curve[:, 1] - self.domain.left] = (
                branch_id + 1
            )
        return branches_label_map

    def branches_arc_length(
        self, graph_ids: Optional[int | Iterable[int]] = None, fast_approximation=True
    ) -> float | npt.NDArray[np.float32]:
        """Return the arc length of the branches.

        Returns
        -------
        Dict[int, float]
            The arc length of the branches. The keys are the branch ids and the values are the arc length of the    branches.
        """  # noqa: E501
        is_single = np.isscalar(graph_ids)
        arc = []

        for curve in self.branches_curve(graph_ids):
            if curve is None:
                arc.append(0)
            elif fast_approximation:
                arc.append(curve.shape[0])
            else:
                arc.append(np.linalg.norm(curve[-1] - curve[0]))
        return arc[0] if is_single else np.asarray(arc, dtype=np.float32)

    def branches_chord_length(self, graph_ids: Optional[int | Iterable[int]] = None) -> float | npt.NDArray[np.float32]:
        """Return the chord length of the branches."""
        is_single = np.isscalar(graph_ids)
        chord = []

        for curve in self.branches_curve(graph_ids):
            if curve is None:
                chord.append(0)
            else:
                chord.append(np.linalg.norm(curve[-1] - curve[0]))
        return chord[0] if is_single else np.asarray(chord, dtype=np.float32)

    def attr_of_branch(self, branch_id: int) -> Dict[str, VBranchGeoAttr]:
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
            attr_name: attr_data[branch_id]
            for attr_name, attr_data in self._branches_attrs.items()
            if attr_data[branch_id] is not None
        }

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
            for branchID in self.branches_id:
                distance, closest_point = self.distance_to_branch(point, branchID, True)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = closest_point
            return min_distance, closest_point if return_closest_point else min_distance

        curve = self.branches_curve(branch_id)
        point = np.asarray(point)[None, :]
        distances = np.linalg.norm(point - curve, axis=1)
        min_i = np.argmin(distances)
        return distances[min_i], curve[min_i] if return_closest_point else distances[min_i]

    ####################################################################################################################
    #  === GRAPH MANIPULATION ===
    ####################################################################################################################
    def _sort_internal_nodes_index(self) -> None:
        """Sort the nodes by their indexes in the graph."""
        if self._nodes_id is None:
            return
        sorted_index = np.argsort(self.nodes_id)
        if np.any(np.diff(sorted_index) < 1):
            self._nodes_id = self._nodes_id[sorted_index]
            self._nodes_coord = self._nodes_coord[sorted_index]

    def _reindex_nodes(self, new_node_index: npt.NDArray[np.int_]) -> None:
        """Reindex the nodes in the graph.

        Parameters
        ----------
        node_index : np.ndarray
            The new index of the nodes: node_index[old_index] = new_index or -1 if the node should be removed.
        """
        if self._nodes_id is None:
            if len(new_node_index) > len(self._nodes_coord):
                # If the new index is longer than the current number of nodes, use sparse indexing
                self._nodes_id = self.nodes_id
                return self._reindex_nodes(new_node_index)
            else:
                self._nodes_coord = self._nodes_coord[invert_lookup(new_node_index)]
        else:
            self._nodes_id = self._nodes_id[invert_lookup(new_node_index)]
            self._sort_internal_nodes_index()

    def _sort_internal_branches_index(self) -> None:
        """Sort the branches by their indexes in the graph."""
        if self._branches_id is None:
            return
        sorted_index = np.argsort(self.branches_id)
        if np.any(np.diff(sorted_index) < 1):
            self._branches_id = self.branches_id[sorted_index]
            self._branches_curve = [self._branches_curve[i] for i in sorted_index]
            self._branches_attrs = {k: [v[i] for i in sorted_index] for k, v in self._branches_attrs.items()}

    def _reindex_branches(self, new_branch_index: npt.NDArray[np.int_]) -> None:
        """Reindex the branches in the graph.

        Parameters
        ----------
        branch_index : np.ndarray
            The new index of the branches. branch_index[old_index] = new_index or -1 if the branch should be removed.
        """
        if self._branches_id is None:
            if len(new_branch_index) > len(self._branches_curve):
                # If the new index is longer than the current number of branch, use sparse indexing
                self._branches_id = self.branches_id
                return self._reindex_branches(new_branch_index)
            else:
                inverted_index = invert_lookup(new_branch_index)
                self._branches_curve = [self._branches_curve[i] for i in inverted_index]
                self._branches_attrs = {k: [v[i] for i in inverted_index] for k, v in self._branches_attrs.items()}
        else:
            self._branches_id = new_branch_index[self._branches_id]
            branch_to_delete = self._branches_id < 0
            if np.any(branch_to_delete):
                self._branches_id = self._branches_id[~branch_to_delete]
                self._branches_curve = [
                    curve for i, curve in enumerate(self._branches_curve) if not branch_to_delete[i]
                ]
                self._branches_attrs = {
                    k: [v for i, v in enumerate(v) if not branch_to_delete[i]] for k, v in self._branches_attrs.items()
                }
            self._sort_internal_branches_index()

    def _drop_nodes(self, node_ids: Iterable[int], *, graph_indexing=True):
        """Remove nodes from the graph.

        Parameters
        ----------
        node_ids : np.ndarray
            The ids of the nodes to remove.
        """
        if self._nodes_id is None:
            self._nodes_coord = np.delete(self._nodes_coord, node_ids, axis=0)
        else:
            internal_ids = self._graph_to_internal_nodes_index(node_ids, graph_indexing=graph_indexing)
            self._nodes_id = np.delete(self._nodes_id, internal_ids)
            self._nodes_coord = np.delete(self._nodes_coord, internal_ids, axis=0)

    def _merge_nodes(self, cluster: Iterable[int], weight: Iterable[float] = None, *, graph_indexing=True):
        """Merge the nodes of a cluster into a single node.

        Parameters
        ----------
        cluster : np.ndarray
            The indices of the nodes to merge.
        """
        if weight is None:
            weight = np.ones(len(cluster))
        else:
            weight = np.asarray(weight, dtype=float)

        cluster = np.asarray(cluster)
        cluster = self._graph_to_internal_nodes_index(cluster, graph_indexing, sort_index=True, check_valid=False)
        weight = weight[cluster >= 0]
        cluster = cluster[cluster >= 0]

        weight_total = np.sum(weight)
        if weight_total == 0:
            weight = np.ones(len(cluster)) / len(cluster)
        else:
            weight /= weight_total

        new_node = np.sum(weight[:, None] * self.nodes_coord(cluster, graph_indexing=False), axis=0)
        self._nodes_coord[cluster[0]] = new_node

    def _merge_branches(self, consecutive_branches: Iterable[int]):
        """Merge consecutive branches into a single branch.

        Parameters
        ----------
        consecutive_branches : np.ndarray
            The indices of the branches to merge.
        """
        consecutive_branches = self._graph_to_internal_branches_index(consecutive_branches, sort_index=True)
        branch0 = consecutive_branches[0]
        next_branches = consecutive_branches[1:]

        new_branch_curve = np.concatenate(self.branches_curve(consecutive_branches, graph_indexing=False))
        self._branches_curve[branch0] = new_branch_curve

        for attr_name, attr_data in self._branches_attrs.items():
            new_attr = attr_data[branch0].merge(
                [attr_data[branch_id] for branch_id in next_branches],
                self._branch_ctx(branch0),
            )
            self._branches_attrs[attr_name][branch0] = new_attr

    def _split_branch(self, id: int, new_branch_id: int, splitPosition: int | Point):
        """Split a branch into two branches at a given position.

        Parameters
        ----------
        id : int
            The id of the branch to split.

        new_id : int
            The id of the new branch.

        splitPosition : Point
            The position at which to split the branch.

        """
        internal_id = int(self._graph_to_internal_branches_index(id)[0])
        curve = self.branches_curve(internal_id, graph_indexing=False)
        if curve is None:
            return

        if isinstance(splitPosition, Point):
            splitIndex = np.argmin(np.linalg.norm(curve - splitPosition, axis=1))
        else:
            splitIndex = splitPosition
            splitPosition = Point(*curve[splitIndex])

        new_internal_id = len(self.branches_curve)
        self._branches_curve[internal_id] = curve[splitIndex:]
        self._branches_curve[new_internal_id] = curve[:splitIndex]
        self._branches_id[new_internal_id] = new_branch_id

        for attr_name, branches_attr in self._branches_attrs.items():
            attr = branches_attr[internal_id]
            if attr is not None:
                b0, b1 = attr.split(splitPosition, splitIndex, self._branch_ctx(internal_id))
                self._branches_attrs[attr_name][internal_id] = b0
                self._branches_attrs[attr_name][new_internal_id] = b1

        self._sort_internal_branches_index()

    def _flip_branches_direction(self, id: int | Iterable[int]):
        """Swap the direction of a branch.

        Parameters
        ----------
        id : int | np.ndarray
            The id of the branch to swap.
        """
        internal_ids = self._graph_to_internal_branches_index(id)
        if np.isscalar(internal_ids):
            internal_ids = [internal_ids]

        for branch_id in internal_ids:
            self._branches_curve[branch_id] = np.flip(self.branches_curve(branch_id), axis=0)
            for attr_data in self._branches_attrs.values():
                branch_attr = attr_data[branch_id]
                if branch_attr is not None:
                    attr_data[branch_id] = branch_attr.flip()
