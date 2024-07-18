from __future__ import annotations

__all__ = ["VGraph"]

from typing import Iterable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from fundus_vessels_toolkit.utils.lookup_array import complete_lookup, create_removal_lookup, invert_complete_lookup

from ..utils.geometric import Point, Rect
from ..utils.graph.branch_by_nodes import (
    branch_list_to_branches_by_nodes,
    branches_by_nodes_to_branch_list,
)
from ..utils.graph.cluster import reduce_chains, reduce_clusters
from .vgeometric_data import VGeometricData


class VGraph:
    """
    A class to represent a graph of a vascular network.

    This class provides tools to store, retrieve and visualise the data associated with a vascular network.
    """

    def __init__(
        self,
        branches_list: np.ndarray,
        geometric_data: VGeometricData | Iterable[VGeometricData],
        nodes_attr: Optional[pd.DataFrame] = None,
        branches_attr: Optional[pd.DataFrame] = None,
        nodes_count: Optional[int] = None,
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
        assert (
            branches_list.ndim == 2 and branches_list.shape[1] == 2
        ), "branch_list must be a 2D array of shape (B, 2) where B is the number of branches"
        B = branches_list.shape[0]

        if isinstance(geometric_data, VGeometricData):
            geometric_data = [geometric_data]

        self._branch_list = branches_list
        self._geometric_data: List[VGeometricData] = list(geometric_data)

        if nodes_count is None:
            nodes_indexes = np.unique(branches_list)
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

        if nodes_count is None:
            self.check_integrity()

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
            [gdata.copy() for gdata in self._geometric_data],
            self._nodes_attr.copy(),
            self._branches_attr.copy(),
            self._nodes_count,
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

    def incident_branches(
        self, node_idx: int | Iterable[int], return_branch_direction: bool = False
    ) -> npt.NDArray[np.int_] | Tuple[npt.NDArray[np.int_], npt.NDArray[np.bool_]]:
        """Compute the indexes of the branches incident to the given node.

        Parameters
        ----------
        node_idx : int or Iterable[int]
            The indexes of the nodes to get the incident branches from.

        sign_branch_direction : bool, optional
            If True, also return weither the branch is outgoing from the node. Default is False.

        Returns
        -------
        np.ndarray
            The indexes of the branches incident to the given nodes.

        np.ndarray
            The direction of the branches incident to the given nodes. Only returned if ``return_branch_direction`` is True. True indicates that the branch is outgoing from the node.

        """  # noqa: E501
        if isinstance(node_idx, int):
            node_idx = [node_idx]
        node_idx = np.asarray(node_idx, dtype=int)
        branch_ids = np.argwhere(np.any(np.isin(self._branch_list, node_idx), axis=1)).flatten()
        if return_branch_direction:
            return branch_ids, np.isin(self._branch_list[branch_ids][:, 0], node_idx)
        else:
            return branch_ids

    def nodes_degree(self) -> npt.NDArray[np.int_]:
        """Compute the degree of each node in the graph.

        The degree of a node is the number of branches incident to it.

        Returns
        -------
        np.ndarray
            An array of shape (N,) containing the degree of each node.
        """
        _, nodes_count = np.unique(self._branch_list, return_counts=True)
        return nodes_count

    def endpoints_nodes(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indexes of the terminal (or endpoints) nodes in the graph.

        The terminal nodes are the nodes connected to zero or one branch.

        Returns
        -------
        np.ndarray
            The indexes of the termination nodes.
        """
        if as_mask:
            return np.bincount(self._branch_list.flatten()) <= 1
        else:
            nodes_id, nodes_count = np.unique(self._branch_list, return_counts=True)
            return nodes_id[nodes_count <= 1]

    def non_endpoints_nodes(self, as_mask=False) -> npt.NDArray[np.int_]:
        """Compute the indexes of the non-terminal nodes in the graph.

        The non-terminal nodes are the nodes connected to at least two branches.

        Returns
        -------
        np.ndarray
            The indexes of the termination nodes.
        """
        if as_mask:
            return np.bincount(self._branch_list.flatten()) > 1
        else:
            nodes_id, nodes_count = np.unique(self._branch_list, return_counts=True)
            return nodes_id[nodes_count > 1]

    def terminal_branches(self) -> npt.NDArray[np.int_]:
        """Compute the indexes of the terminal branches in the graph.

        The terminal branches are the branches connected to a single node.

        Returns
        -------
        np.ndarray
            The indexes of the terminal branches.
        """
        _, branch_index, nodes_count = np.unique(self._branch_list, return_counts=True, return_index=True)
        return np.unique(branch_index[nodes_count == 1] // 2)

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
        label_map = np.zeros(full_domain.shape, dtype=int)
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
            _description_
        """
        indexes = complete_lookup(indexes, max_index=self.nodes_count - 1)
        if inverse_lookup:
            indexes = invert_complete_lookup(indexes)

        self._branch_list = indexes[self._branch_list]
        self._nodes_attr = self._nodes_attr.reindex(indexes)
        for gdata in self._geometric_data:
            gdata._reindex_nodes(indexes)
        return self

    def sort_nodes_by_degree(self, descending=True) -> VGraph:
        degree = self.nodes_degree()
        new_order = np.argsort(degree)
        if descending:
            new_order = new_order[::-1]
        return self.reindex_nodes(new_order)

    def reindex_branches(self, new_branch_indexes) -> VGraph:
        new_branch_indexes = complete_lookup(new_branch_indexes, max_index=self.branches_count - 1)

        self._branch_list = new_branch_indexes[self._branch_list]
        self._branches_attr = self._branches_attr.reindex(new_branch_indexes)
        for gdata in self._geometric_data:
            gdata._reindex_branches(new_branch_indexes)
        return self

    def flip_branches_direction(self, id: int | Iterable[int]) -> VGraph:
        if isinstance(id, int):
            id = [id]
        self._branch_list[id] = self._branch_list[id, ::-1]
        for gdata in self._geometric_data:
            gdata._flip_branches_direction(id)
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
    def _delete_branches(self, branch_indexes: npt.NDArray[np.int_]):
        if len(branch_indexes) == 0:
            return
        branches_reindex = create_removal_lookup(branch_indexes, replace_value=-1, length=self.branches_count)
        for gdata in self._geometric_data:
            gdata._reindex_branches(branches_reindex)

        self._branch_list = np.delete(self._branch_list, branch_indexes, axis=0)
        self._branches_attr = self._branches_attr.drop(branch_indexes).reset_index(drop=True)

    def _delete_nodes(self, node_indexes: npt.NDArray[np.int_]):
        if len(node_indexes) == 0:
            return
        self._nodes_attr = self._nodes_attr.drop(node_indexes).reset_index(drop=True)

        nodes_reindex = create_removal_lookup(node_indexes, replace_value=-1, length=self.nodes_count)
        self._branch_list = nodes_reindex[self._branch_list]
        for gdata in self._geometric_data:
            gdata._reindex_nodes(nodes_reindex)

        self._nodes_count -= len(node_indexes)

    # --- Branches edition ---
    def delete_branches(
        self, branch_indexes: npt.NDArray[np.int_], delete_orphan_nodes=True, *, inplace=True
    ) -> VGraph:
        """Remove the branches with the given indexes from the graph.

        Parameters
        ----------
        branch_indexes : npt.NDArray

        delete_orphan_nodes : bool, optional
            If True (by default), the nodes that are not connected to any branch after the deletion are removed from the graph.
        inplace : bool, optional
            If True (by default), the graph is modified in place. Otherwise, a new graph is returned.

        Returns
        -------
        VGraph
            The modified graph.
        """  # noqa: E501
        if not inplace:
            return self.copy().delete_branches(branch_indexes, delete_orphan_nodes=delete_orphan_nodes, inplace=True)

        branch_indexes = np.asarray(branch_indexes, dtype=int)
        assert branch_indexes.ndim == 1, "branch_indexes must be a 1D array."

        # Find orphan nodes
        orphan_nodes = np.array([])
        if delete_orphan_nodes:
            connected_nodes = np.unique(self._branch_list[branch_indexes])
            terminations_nodes = self.endpoints_nodes()
            orphan_nodes = np.intersect1d(connected_nodes, terminations_nodes)

        # Remove branches and orphan nodes
        self._delete_branches(branch_indexes)
        if len(orphan_nodes):
            self._delete_nodes(orphan_nodes)

        return self

    def merge_consecutive_branches(
        self,
        consecutive_branches: npt.NDArray[np.int_] | list[npt.NDArray[np.int_]],
        *,
        assume_fusable=False,
        inplace=True,
    ) -> VGraph:
        """Merge pairs of consecutive branches by linking their nodes together.

        The nodes are removed from the graph and their corresponding branches are merged.

        Parameters
        ----------
        consecutive_branches :  npt.NDArray | list[npt.NDArray]
            Single dimension array containing a list of consecutive branches to merge. Or a list of such arrays.

        Returns
        -------
        VGraph
            The modified graph.
        """
        if not inplace:
            return self.copy().merge_consecutive_branches(consecutive_branches, inplace=True)

        if not isinstance(consecutive_branches, list):
            consecutive_branches = [consecutive_branches]
        consecutive_branches = reduce_chains(consecutive_branches)

        fused_nodes = []
        deleted_branches = []
        for branches in consecutive_branches:
            assert len(branches) >= 2, "At least two branches must be provided to merge them."

            # Find the node shared by the first two branches
            nodes, nodes_count = np.unique(self._branch_list[branches[:2]], return_counts=True)
            argmax = np.argmax(nodes_count == 2)
            assert nodes_count[argmax] == 2, f"Branches {branches[0]} is not consecutive to branch {branches[1]}."
            nNext = nodes[argmax]
            flip_branches = np.zeros(len(branches), dtype=bool)

            # Store the other node as n0
            n0, n1 = self._branch_list[branches[0]]
            # n1_id = 1
            if n0 == nNext:
                n0 = n1
                # n1_id = 0
                flip_branches[0] = True

            # For each branch, check that its consecutive to the previous one and store the next node
            for i, b in enumerate(branches[1:]):
                n1, n2 = self._branch_list[b]
                if n1 == nNext:
                    fused_nodes.append(n1)
                    nNext = n2
                elif n2 == nNext:
                    fused_nodes.append(n2)
                    nNext = n1
                    flip_branches[i + 1] = True
                else:
                    raise ValueError(f"Branches {b} is not consecutive to branch {branches[i]}.")
            # Update the first branch
            self._branch_list[branches[0]] = [n0, nNext]

            deleted_branches.extend(branches[1:])
            for gdata in self._geometric_data:
                if flip_branches.any():
                    gdata._flip_branches_direction(np.asarray(branches)[flip_branches])
                gdata._merge_branches(branches)

        if not assume_fusable:
            assert np.all(
                np.isin(np.argwhere(np.any(np.isin(self.branch_list, fused_nodes), axis=1)), deleted_branches)
            ), (
                "Some nodes are still referenced in the branch list and shouldn't be fused. "
                "This is may be because nodes connecting consecutive branches are also connected to other branches."
            )
        self._delete_branches(deleted_branches)
        self._delete_nodes(fused_nodes)

        return self

    def split_branch(self, branchID: int, split_coord: Point, *, inplace=False) -> VGraph:
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
        if not inplace:
            return self.copy().split_branch(branchID, split_coord, inplace=True)

        n1, n2 = self._branch_list[branchID]
        new_nodeID = self.nodes_count
        new_branchID = self.branches_count

        self._branch_list[branchID, 1] = new_nodeID
        self._branch_list = np.concatenate((self._branch_list, [[new_nodeID, n2]]), axis=0)

        for gdata in self._geometric_data:
            gdata._split_branch(branchID, new_branchID, split_coord)

        return self

    def bridge_nodes(self, node_pairs: npt.NDArray, *, fuse_nodes=False, no_check=False, inplace=False) -> VGraph:
        """Fuse pairs of termination nodes by linking them together.

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
        if not inplace:
            return self.copy().bridge_nodes(node_pairs, inplace=True)

        assert node_pairs.ndim == 2 and node_pairs.shape[1] == 2, "node_pairs must be a 2D array of shape (P, 2)."

        if not no_check:
            node_pairs = np.unique(node_pairs, axis=0)
            node_pairs = node_pairs[np.isin(node_pairs, self._branch_list, assume_unique=True, invert=True)]

        self._branch_list = np.concatenate((self._branch_list, node_pairs), axis=0)
        if fuse_nodes:
            self.fuse_nodes(np.unique(node_pairs))
        return self

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

    def fuse_nodes(self, nodes: npt.NDArray[np.int_], *, quiet_invalid_node=False, inplace=False) -> VGraph:
        """Fuse nodes connected to exactly two branches.

        The nodes are removed from the graph and their corresponding branches are merged.

        Parameters
        ----------
        nodes : npt.NDArray
            Array of indexes of the nodes to fuse.

        Returns
        -------
        VGraph
            The modified graph.
        """
        consecutive_branches = []
        for nodes_id in nodes:
            b = np.argwhere(np.any(self._branch_list == nodes_id, axis=1)).flatten()
            if len(b) != 2:
                if not quiet_invalid_node:
                    raise ValueError(f"Node {nodes_id} is not connected to exactly two branches.")
                else:
                    continue
            b0, b1 = b

            for branches in consecutive_branches:
                if branches[-1] == b0:
                    branches.append(b1)
                    break
                elif branches[0] == b1:
                    branches.insert(0, b0)
                    break
            else:
                consecutive_branches.append([b0, b1])

        return self.merge_consecutive_branches(consecutive_branches, inplace=inplace)

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
        """  # noqa: E501 _description_
        if not inplace:
            return self.copy().merge_nodes(
                clusters, inplace=True, assume_reduced=assume_reduced, nodes_weight=nodes_weight
            )

        branches_to_remove = list()
        nodes_to_remove = list()

        if not assume_reduced:
            clusters = reduce_clusters(clusters)

        for cluster in clusters:
            nodes = np.asarray(cluster, dtype=int)
            nodes.sort()
            n0 = nodes[0]

            # 1. Handle branches connections
            branches_connection_count = np.sum(np.isin(self._branch_list, nodes), axis=1)
            #   - Mark every branches inside the cluster (with 2/2 connections in cluster) to be removed
            branches_to_remove.extend(np.argwhere(branches_connection_count == 2).flatten())
            #   - Connect every branches incident to the cluster to the kept node
            branches_to_connect = np.argwhere(branches_connection_count == 1).flatten()
            if len(branches_to_connect) == 0:
                nodes_to_remove.append(nodes)
                continue
            else:
                for b in branches_to_connect:
                    n1, n2 = self._branch_list[b]
                    self._branch_list[b] = [n0, n2] if n1 in nodes else [n1, n0]

            # 2. Handle Geometric data
            weight = None
            if nodes_weight is not None:
                weight = nodes_weight[nodes]
            for gdata in self._geometric_data:
                gdata._merge_nodes(nodes, weight=weight)

            # 3. Mark nodes to be removed
            nodes_to_remove.append(nodes[1:])

        if len(branches_to_remove) > 0:
            branches_to_remove = np.unique(np.asarray(branches_to_remove, dtype=int))
            self._delete_branches(branches_to_remove)

        if len(nodes_to_remove) > 0:
            nodes_to_remove = np.unique(np.concatenate(nodes_to_remove, dtype=int))
            self._delete_nodes(nodes_to_remove)

        return self

    ####################################################################################################################
    #  === VISUALISATION UTILITIES ===
    ####################################################################################################################
    def jppype_layer(
        self,
        edge_labels=False,
        node_labels=False,
        edge_map=True,
        bspline_attr=None,
        max_colored_node_id: Optional[int] = None,
    ):
        from jppype.layers import LayerGraph
        from jppype.utilities.color import colormap_by_name

        from .vgeometric_data import VBranchGeoData

        layer = LayerGraph(self.branch_list, self.nodes_coord(), self.geometric_data().branches_label_map())
        layer.set_options(
            {
                "edge_labels_visible": edge_labels,
                "node_labels_visible": node_labels,
                "edge_map_visible": edge_map,
            }
        )
        if bspline_attr is not None:
            branches_geodata = self.branches_geo_data(bspline_attr)
            layer.edges_path = [
                d.bspline.to_path() if isinstance(d, VBranchGeoData.BSpline) else "" for d in branches_geodata
            ]
        if max_colored_node_id:
            node_cmap = {None: colormap_by_name()} | {
                _: "#444" for _ in range(max_colored_node_id + 1, self.nodes_count + 1)
            }
            layer.nodes_cmap = node_cmap
            incidents_branches = np.setdiff1d(
                np.arange(self.branches_count), self.incident_branches(np.arange(max_colored_node_id + 1))
            )
            branch_cmap = {None: colormap_by_name()} | {int(_): "#444" for _ in incidents_branches}
            layer.edges_cmap = branch_cmap
        return layer

    def branch_normals_map(self, segmentation, only_terminations=False):
        import cv2

        from ..utils.graph.measures import branch_boundaries

        def eval_point(curve):
            return [0, len(curve) - 1] if only_terminations else np.arange(len(curve), step=4)

        out_map = np.zeros_like(segmentation, dtype=np.uint8)

        yx_curves = {k: curve for k, curve in enumerate(self.geometric_data().branch_curve()) if len(curve) > 1}
        boundaries = {id: branch_boundaries(curve, segmentation, eval_point(curve)) for id, curve in yx_curves.items()}

        for i, curve in yx_curves.items():
            bounds = boundaries[i]
            curve = curve[eval_point(curve)]
            for (y, x), ((byL, bxL), (byR, bxR)) in zip(curve, bounds, strict=True):
                dL = np.linalg.norm([x - bxL, y - byL])
                dR = np.linalg.norm([x - bxR, y - byR])
                if not only_terminations and abs(dL - dR) > 1.5:
                    continue
                if byL != y or bxL != x:
                    cv2.line(out_map, (x, y), (bxL, byL), int(i + 1), 1)
                if byR != y or bxR != x:
                    cv2.line(out_map, (x, y), (bxR, byR), int(i + 1), 1)
                out_map[y, x] = 0

        return out_map
