from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.geometric import Point, Rect
from ..utils.graph.branch_by_nodes import (
    branch_list_to_branches_by_nodes,
    branches_by_nodes_to_branch_list,
)


class VGraph:
    """
    A class to represent a graph of a vascular network.

    This class provides tools to store, retrieve and visualise the data associated with a vascular network.
    """

    def __init__(
        self,
        branches_list: np.ndarray,
        branches_labels_map: np.ndarray,
        nodes_yx_coord: np.ndarray,
        nodes_attr: Optional[pd.DataFrame] = None,
        branches_attr: Optional[pd.DataFrame] = None,
    ):
        """Create a Graph object from the given data.

        Parameters
        ----------
        branches_list :
            A 2D array of shape (B, 2) where B is the number of branches. Each row contains the indexes of the nodes
            connected by each branch.

        branches_labels_map : np.ndarray
            A 2D array of shape (H, W) where H and W are the image height and width. Each non-zero pixels is part of the skeleton. The value of the pixel is the index of the branch it belongs to plus one.

        nodes_yx_coord :
            A 2D array of shape (N, 2) where N is the number of nodes. Each row contains the (y, x) coordinates of the node.

        nodes_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each node. The index of the dataframe must be the nodes indexes.

        branches_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each branch. The index of the dataframe must be the branches indexes.


        Raises
        ------
        ValueError
            If the input data does not match the expected shapes.
        """  # noqa: E501
        assert (
            branches_list.ndim == 2 and branches_list.shape[1] == 2
        ), "branch_list must be a 2D array of shape (B, 2) where B is the number of branches"
        B = branches_list.shape[0]
        assert branches_labels_map.ndim == 2, (
            "branch_labels_map must be a 2D array of shape (H, W) where H and W are the image height and width."
            f"Got {branches_labels_map.shape}"
        )
        assert branches_labels_map.max() <= B, (
            "branch_labels_map must have a maximum value equal to the number of branches."
            f"Got {branches_labels_map.max()} instead of {B}"
        )

        if isinstance(nodes_yx_coord, tuple):
            nodes_yx_coord = np.asarray(nodes_yx_coord)
            if nodes_yx_coord.shape[0] == 2 and nodes_yx_coord.shape[1] != 2:
                nodes_yx_coord = nodes_yx_coord.T

        assert nodes_yx_coord.ndim == 2 and nodes_yx_coord.shape[1] == 2, (
            f"nodes_yx_coord must be a 2D array of shape (N, 2)." f" Got shape={nodes_yx_coord.shape}."
        )
        N = nodes_yx_coord.shape[0]

        assert branches_list.max() < N, (
            "The maximum value in branch_list must be lower than the number of nodes."
            f" Got {branches_list.max()} instead of {N}"
        )

        if nodes_attr is not None:
            if isinstance(nodes_attr, pd.DataFrame):
                assert (
                    nodes_attr.index.inferred_type == "integer"
                ), "The index of nodes_attr dataframe must be nodes Index."
                assert nodes_attr.index.max() < N and nodes_attr.index.min() >= 0, (
                    "The maximum value in nodes_attr index must be lower than the number of nodes."
                    f" Got {nodes_attr.index.max()} instead of {N}"
                )
                if len(nodes_attr.index) != N:
                    nodes_attr.reindex(np.arange(N))
            else:
                raise ValueError("nodes_attr must be a pandas.DataFrame")
        else:
            nodes_attr = pd.DataFrame(index=np.arange(N))

        if branches_attr is not None:
            if isinstance(branches_attr, pd.DataFrame):
                assert (
                    branches_attr.index.inferred_type == "integer"
                ), "The index of branches_attr dataframe must be branches Index."
                assert branches_attr.index.max() < B and branches_attr.index.min() >= 0, (
                    "The maximum value in branches_attr index must be lower than the number of branches."
                    f" Got {branches_attr.index.max()} instead of {B}"
                )
                if len(branches_attr.index) != B:
                    branches_attr.reindex(np.arange(B))
            else:
                raise ValueError("branches_attr must be a pandas.DataFrame")
        else:
            branches_attr = pd.DataFrame(index=np.arange(B))

        self._branch_list = branches_list
        self._nodes_yx_coord = nodes_yx_coord
        self._branch_labels_map = branches_labels_map.clip(0)
        self._nodes_attr = nodes_attr
        self._branches_attr = branches_attr

    @classmethod
    def from_branch_by_nodes(
        cls,
        branch_by_nodes: npt.NDArray[np.bool_],
        branch_labels: npt.NDArray[np.int_],
        nodes_yx_coord: npt.NDArray[np.float_],
        nodes_attr: Optional[pd.DataFrame] = None,
        branches_attr: Optional[pd.DataFrame] = None,
    ) -> VGraph:
        """Create a Graph object from a branch-by-nodes connectivity matrix instead of a branch list.

        Parameters
        ----------
        branch_by_nodes : np.ndarray
            A 2D array of shape (B, N) where N is the number of nodes and B is the number of branches. Each True value in the array indicates that the branch is connected to the corresponding node. Such matrix should therefore contains exactly two True values per row.

        branch_labels : np.ndarray
            A 2D array of shape (H, W) where H and W are the image height and width. Each non-zero pixels is part of the skeleton. The value of the pixel is the index of the branch it belongs to.

        nodes_yx_coord : np.ndarray
            A 2D array of shape (N, 2) where N is the number of nodes. Each row contains the (y, x) coordinates of the node.

        nodes_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each node. The index of the dataframe must be the nodes indexes.

        branches_attr : Optional[pd.DataFrame], optional
            A pandas.DataFrame containing the attributes of each branch. The index of the dataframe must be the branches indexes.

        Returns
        -------
            The Graph object created from the given data.
        """  # noqa: E501
        branch_list = branches_by_nodes_to_branch_list(branch_by_nodes)
        return cls(branch_list, branch_labels, nodes_yx_coord, nodes_attr, branches_attr)

    def copy(self) -> VGraph:
        """Create a copy of the current Graph object."""
        return VGraph(
            self._branch_list.copy(),
            self._branch_labels_map.copy(),
            self._nodes_yx_coord.copy(),
            self._nodes_attr.copy(),
            self._branches_attr.copy(),
        )

    ####################################################################################################################
    #  === PROPERTIES ===
    ####################################################################################################################
    @property
    def nodes_count(self) -> int:
        """The number of nodes in the graph."""
        return self._nodes_yx_coord.shape[0]

    @property
    def nodes_yx_coord(self) -> npt.NDArray[np.float_]:
        """The (y, x) coordinates of the nodes in the graph as a 2D array of shape (N, 2) where N is the number of nodes."""  # noqa: E501
        return self._nodes_yx_coord

    @property
    def nodes_attr(self) -> pd.DataFrame:
        """The attributes of the nodes in the graph as a pandas.DataFrame."""
        return self._nodes_attr

    @property
    def branches_count(self) -> int:
        """The number of branches in the graph."""
        return self._branch_list.shape[0]

    @property
    def branch_list(self) -> npt.NDArray[np.int_]:
        """The list of branches in the graph as a 2D array of shape (B, 2) where B is the number of branches. Each row contains the indexes of the nodes connected by each branch."""
        return self._branch_list

    @property
    def branch_labels_map(self) -> npt.NDArray[np.int_]:
        """The labels map of the branches in the graph as a 2D array of shape (H, W) where H and W are the image height and width. Each non-zero pixels is part of the skeleton. The value of the pixel indicate the index of the branch it belongs to.

        ..warning::
            The value of the pixels is the index of the branch plus one because the value 0 is reserved for the background.

        """  # noqa: E501
        return self._branch_labels_map

    @property
    def skeleton(self) -> npt.NDArray[np.bool_]:
        """The binary skeleton of the graph as a 2D array of shape (H, W) where H and W are the image height and width. Each non-zero pixels is part of the skeleton."""  # noqa: E501
        return self._branch_labels_map > 0

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

    def incident_branches(self, node_idx: int | Iterable[int]) -> npt.NDArray[np.int_]:
        """Compute the indexes of the branches incident to the given node.

        Parameters
        ----------
        node_idx : int or Iterable[int]
            The indexes of the nodes to get the incident branches from.

        Returns
        -------
        np.ndarray
            The indexes of the branches incident to the given nodes.
        """
        if isinstance(node_idx, int):
            node_idx = [node_idx]
        return np.where(np.any(np.isin(self._branch_list, node_idx), axis=1))[0]

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

    def terminations_nodes(self) -> npt.NDArray[np.int_]:
        """Compute the indexes of the termination nodes in the graph.

        The termination nodes are the nodes connected to zero or one branch.

        Returns
        -------
        np.ndarray
            The indexes of the termination nodes.
        """
        nodes_id, nodes_count = np.unique(self._branch_list, return_counts=True)
        return nodes_id[nodes_count <= 1]

    ####################################################################################################################
    #  === GRAPH MANIPULATION ===
    ####################################################################################################################
    def shuffle_nodes(self, new_node_indexes):
        new_node_indexes = np.asarray(new_node_indexes, dtype=int)
        assert (
            new_node_indexes.ndim == 1 and new_node_indexes.shape[0] <= self.nodes_count
        ), f"node_indexes must be a 1D array of maximum size {self.nodes_count}"

        if new_node_indexes.shape[0] < self.nodes_count:
            new_node_indexes = np.concatenate(
                (new_node_indexes, np.setdiff1d(np.arange(self.nodes_count), new_node_indexes))
            )

        self._branch_list = new_node_indexes[self._branch_list]
        self._nodes_yx_coord = self._nodes_yx_coord[new_node_indexes]
        self._nodes_attr = self._nodes_attr.reindex(new_node_indexes)
        return self

    def shuffle_branches(self, new_branch_indexes):
        new_branch_indexes = np.asarray(new_branch_indexes, dtype=int)
        assert (
            new_branch_indexes.ndim == 1 and new_branch_indexes.shape[0] <= self.branches_count
        ), f"branch_indexes must be a 1D array of maximum size {self.branches_count}"

        if new_branch_indexes.shape[0] < self.branches_count:
            new_branch_indexes = np.concatenate(
                (new_branch_indexes, np.setdiff1d(np.arange(self.branches_count), new_branch_indexes))
            )

        self._branch_list = new_branch_indexes[self._branch_list]
        self._branch_labels_map = new_branch_indexes[self._branch_labels_map]
        self._branches_attr = self._branches_attr.reindex(new_branch_indexes)
        return self

    def bridge_nodes(self, node_pairs: npt.NDArray, draw_link=True, inplace=False) -> VGraph:
        """Fuse pairs of termination nodes by linking them together.

        The nodes are removed from the graph and their corresponding branches are merged.

        Parameters
        ----------
        node_pairs : npt.NDArray
            Array of shape (P, 2) containing P pairs of indexes of the nodes to link.
        draw_link : bool, optional
            If True, the link between the nodes is drawn as a straight line, by default True

        Returns
        -------
        VascularGraph
            self
        """
        from ..utils.graph.branch_by_nodes import fuse_node_pairs, index_to_mask

        branches_by_nodes, branch_lookup, branch_id = fuse_node_pairs(self.branches_by_nodes(), node_pairs)

        # Apply branch merging to the branch labels map
        branch_labels_map = branch_lookup[self._branch_labels_map]
        if draw_link:
            from skimage.draw import line

            for (n1, n2), b in zip(node_pairs, branch_id, strict=True):
                n1 = Point(*self._nodes_yx_coord[n1]).to_int()
                n2 = Point(*self._nodes_yx_coord[n2]).to_int()
                rr, cc = line(*n1, *n2)
                branch_labels_map[rr, cc] = b + 1

        # Remove the fused nodes from the nodes coordinates
        nodes_yx_coord = self._nodes_yx_coord[index_to_mask(node_pairs.flatten(), self.nodes_count, invert=True)]
        branch_list = branches_by_nodes_to_branch_list(branches_by_nodes)

        if inplace:
            self._branch_list = branch_list
            self._branch_labels_map = branch_labels_map
            self._nodes_yx_coord = nodes_yx_coord
            return self
        else:
            return VGraph(branch_list, branch_labels_map, nodes_yx_coord)

    ####################################################################################################################
    #  === VISUALISATION UTILITIES ===
    ####################################################################################################################
    def jppype_layer(self, edge_labels=False, node_labels=False, edge_map=True):
        from jppype.layers import LayerGraph

        layer = LayerGraph(self.branch_list, self._nodes_yx_coord, self._branch_labels_map)
        layer.set_options(
            {
                "edge_labels_visible": edge_labels,
                "node_labels_visible": node_labels,
                "edge_map_visible": edge_map,
            }
        )
        return layer

    def jppype_quiver_termination_tangents(self, std=7, offset=7, arrow_length=20):
        from jppype.layers import LayerQuiver

        terminations = self.terminations_nodes()
        yx = self.nodes_yx_coord[terminations]
        uv = self.nodes_tangent(terminations, gaussian_std=std, gaussian_offset=offset) * arrow_length
        return LayerQuiver(yx, uv, Rect.from_size(self._branch_labels_map.shape))

    def branch_normals_map(self, segmentation, only_terminations=False):
        import cv2

        from ..utils.graph.measures import branch_boundaries, track_branches

        def eval_point(curve):
            return [0, len(curve) - 1] if only_terminations else np.arange(len(curve), step=4)

        out_map = np.zeros_like(segmentation, dtype=np.uint8)

        yx_curves = track_branches(self.branch_labels_map, self.nodes_yx_coord, self.branch_list)
        boundaries = [branch_boundaries(curve, segmentation, eval_point(curve)) for curve in yx_curves]

        for i, (curve, boundaries) in enumerate(zip(yx_curves, boundaries, strict=True)):
            curve = curve[eval_point(curve)]
            for (y, x), ((byL, bxL), (byR, bxR)) in zip(curve, boundaries, strict=True):
                dL = np.linalg.norm([x - bxL, y - byL])
                dR = np.linalg.norm([x - bxR, y - byR])
                if not only_terminations and abs(dL - dR) > 1.5:
                    continue
                if byL != y or bxL != x:
                    cv2.line(out_map, (x, y), (bxL, byL), i + 1, 1)
                if byR != y or bxR != x:
                    cv2.line(out_map, (x, y), (bxR, byR), i + 1, 1)
                out_map[y, x] = 0

        return out_map
