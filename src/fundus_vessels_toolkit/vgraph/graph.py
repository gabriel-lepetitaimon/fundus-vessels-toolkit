from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt

from ..utils.geometric import Point, Rect
from ..utils.graph.branch_by_nodes import (
    branch_list_to_branches_by_nodes,
    branches_by_nodes_to_branch_list,
)
from ..utils.graph.measures import nodes_tangent, perimeter_from_vertices


class Graph:
    def __init__(
        self,
        branches_list: np.ndarray,
        branches_labels_map: np.ndarray,
        nodes_yx_coord: np.ndarray,
    ):
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

        self._branch_list = branches_list
        self._nodes_yx_coord = nodes_yx_coord
        self._branch_labels_map = branches_labels_map.clip(0)

    @classmethod
    def from_branch_by_nodes(cls, branch_by_nodes: np.ndarray, branch_labels: np.ndarray, nodes_yx_coord: np.ndarray):
        branch_list = branches_by_nodes_to_branch_list(branch_by_nodes)
        return cls(branch_list, branch_labels, nodes_yx_coord)

    def copy(self):
        return Graph(self._branch_list.copy(), self._branch_labels_map.copy(), self._nodes_yx_coord.copy())

    ####################################################################################################################
    #  === PROPERTIES ===
    ####################################################################################################################
    @property
    def branch_list(self):
        return self._branch_list

    @property
    def nodes_yx_coord(self):
        return self._nodes_yx_coord

    @property
    def branch_labels_map(self):
        return self._branch_labels_map

    @property
    def nodes_count(self):
        return self._nodes_yx_coord.shape[0]

    @property
    def branches_count(self):
        return self._branch_list.shape[0]

    @property
    def skeleton(self):
        return self._branch_labels_map > 0

    #  --- Secondary Properties ---
    def branch_adjacency_matrix(self):
        return self._branch_list.dot(self._branch_list.T)

    def node_adjacency_matrix(self):
        return self._branch_list.T.dot(self._branch_list)

    def branches_by_nodes(self):
        return branch_list_to_branches_by_nodes(
            self._branch_list, n_branches=self.branches_count, n_nodes=self.nodes_count
        )

    def terminations_nodes(self) -> npt.NDArray[np.int_]:
        nodes_id, nodes_count = np.unique(self._branch_list, return_counts=True)
        return nodes_id[nodes_count == 1]

    ####################################################################################################################
    #  === COMPUTABLE GRAPH ATTRIBUTES ===
    ####################################################################################################################
    def nodes_distance(self, *nodes_idx, close_loop=False):
        nodes = self._nodes_yx_coord[np.asarray(nodes_idx)]
        return perimeter_from_vertices(nodes, close_loop=close_loop)

    def incident_branches(self, node_idx):
        if isinstance(node_idx, int):
            node_idx = [node_idx]
        return np.where(np.any(np.isin(self._branch_list, node_idx), axis=1))[0]

    def nodes_tangent(
        self,
        nodes: Iterable[int],
        branches_id: Optional[Iterable[int | None]] = None,
        gaussian_offset: float = 7,
        gaussian_std: float = 7,
    ) -> npt.NDArray[np.float]:
        """Compute the tangent of the skeleton at the given nodes.

        Parameters
        ----------
        nodes : Iterable[int], length: N
            List the nodes indexes where the tangents should be computed.
        branch_id : Optional[Iterable[int|None]], optional, length: N
            For each nodes, the index of the branch used to compute the tangent. If None the node is expected to be a
            termination and its only branch will be automatically selected.
            By default: None.
        gaussian_offset : float, optional
            The offset of the Gaussian kernel weighting the nodes surrounding to compute the tangent.
            By default 7.
        gaussian_std : float, optional
            The standard deviation of the Gaussian kernel weighting the nodes surrounding to compute the tangent.
            By default 7.

        Returns
        -------
        npt.NDArray[np.float], shape (N, 2)
            The tangent vectors at the given nodes. The vectors are normalized to unit length.
        """
        assert all(0 <= n < self.nodes_count for n in nodes), f"Invalid node index: must be in [0, {self.nodes_count})."

        # Automatically infer branches_id if not provided
        if branches_id is None:
            branches_id = [None] * len(nodes)
        assert len(branches_id) == len(nodes), "nodes and branches_id must have the same length."

        if any(b is None for b in branches_id):
            for i, (n, b) in enumerate(zip(nodes, branches_id, strict=True)):
                if b is None:
                    incident_branches = self.incident_branches(n)
                    assert len(incident_branches) == 1, (
                        f"Impossible to infer the branch id for node {n}. This node has {len(incident_branches)} "
                        "incident branches instead of 1. Please provide the branch id manually."
                    )
                    branches_id[i] = incident_branches[0]
        assert all(
            0 <= b < self.branches_count for b in branches_id
        ), f"Invalid branch index: must be in [0, {self.branches_count})."

        branches_id = np.asarray(branches_id, dtype=int)

        # Compute the nodes tangents
        node_yx = self.nodes_yx_coord[nodes]
        return nodes_tangent(
            nodes_coord=node_yx,
            branches_label_map=self.branch_labels_map,
            branches_id=branches_id + 1,
            gaussian_offset=gaussian_offset,
            gaussian_std=gaussian_std,
        )

    ####################################################################################################################
    #  === GRAPH MANIPULATION ===
    ####################################################################################################################
    def shuffle_nodes(self, node_indexes):
        node_indexes = np.asarray(node_indexes, dtype=int)
        assert (
            node_indexes.ndim == 1 and node_indexes.shape[0] <= self.nodes_count
        ), f"node_indexes must be a 1D array of maximum size {self.nodes_count}"

        if node_indexes.shape[0] < self.nodes_count:
            node_indexes = np.concatenate((node_indexes, np.setdiff1d(np.arange(self.nodes_count), node_indexes)))

        self._branch_list = node_indexes[self._branch_list]
        self._nodes_yx_coord = self._nodes_yx_coord[node_indexes]
        return self

    def bridge_nodes(self, node_pairs: npt.NDArray, draw_link=True, inplace=False) -> Graph:
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
            return Graph(branch_list, branch_labels_map, nodes_yx_coord)

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
