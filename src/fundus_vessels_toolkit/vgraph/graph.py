from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt

from ..utils.geometric import Point, Rect
from ..utils.graph.branch_by_nodes import branch_by_nodes_to_adjacency_list


class Graph:
    def __init__(
        self,
        branch_by_node: np.ndarray,
        branch_labels_map: np.ndarray,
        nodes_yx_coord: np.ndarray,
    ):
        assert (
            branch_by_node.ndim == 2
        ), "branch_by_node must be a 2D array of shape (B, N) where B is the number of branches and N the number of nodes"
        B, N = branch_by_node.shape
        assert branch_labels_map.ndim == 2, (
            "branch_labels_map must be a 2D array of shape (H, W) where H and W are the image height and width."
            f"Got {branch_labels_map.shape}"
        )
        assert branch_labels_map.max() == B, (
            "branch_labels_map must have a maximum value equal to the number of branches."
            f"Got {branch_labels_map.max()} instead of {B}"
        )

        if isinstance(nodes_yx_coord, tuple):
            nodes_yx_coord = np.asarray(nodes_yx_coord)
            if nodes_yx_coord.shape[0] == 2 and nodes_yx_coord.shape[1] != 2:
                nodes_yx_coord = nodes_yx_coord.T
        assert nodes_yx_coord.shape == (N, 2), (
            f"nodes_yx_coord must be a 2D array of shape (N, 2) where N is the number of node: {N} ."
            f" Got shape={nodes_yx_coord.shape}."
        )

        self._branch_by_node = branch_by_node
        self._nodes_yx_coord = nodes_yx_coord
        self._branch_labels_map = branch_labels_map

    def copy(self):
        return Graph(self._branch_by_node.copy(), self._branch_labels_map.copy(), self._nodes_yx_coord.copy())

    @property
    def skeleton(self):
        return self._branch_labels_map > 0

    def branch_adjacency_matrix(self):
        return self._branch_by_node.dot(self._branch_by_node.T)

    def node_adjacency_matrix(self):
        return self._branch_by_node.T.dot(self._branch_by_node)

    def node_adjacency_list(self):
        return branch_by_nodes_to_adjacency_list(self._branch_by_node)

    def jppype_layer(self, edge_labels=False, node_labels=False, edge_map=True):
        from jppype.layers import LayerGraph

        layer = LayerGraph(self.node_adjacency_list(), self._nodes_yx_coord, self._branch_labels_map)
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

    @property
    def branch_by_node(self):
        return self._branch_by_node

    @property
    def nodes_yx_coord(self):
        return self._nodes_yx_coord

    @property
    def branch_labels_map(self):
        return self._branch_labels_map

    @property
    def nodes_count(self):
        return self._branch_by_node.shape[1]

    @property
    def branches_count(self):
        return self._branch_by_node.shape[0]

    def shuffle_nodes(self, node_indexes):
        node_indexes = np.asarray(node_indexes, dtype=int)
        assert (
            node_indexes.ndim == 1 and node_indexes.shape[0] <= self.nodes_count
        ), f"node_indexes must be a 1D array of maximum size {self.nodes_count}"

        if node_indexes.shape[0] < self._branch_by_node.shape[1]:
            node_indexes = np.concatenate((node_indexes, np.setdiff1d(np.arange(self.nodes_count), node_indexes)))

        self._branch_by_node = self._branch_by_node[:, node_indexes]
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
        from ..seg_to_graph.graph_utilities import fuse_node_pairs, index_to_mask

        branch_by_node, branch_lookup, branch_id = fuse_node_pairs(self._branch_by_node, node_pairs)

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

        if inplace:
            self._branch_by_node = branch_by_node
            self._branch_labels_map = branch_labels_map
            self._nodes_yx_coord = nodes_yx_coord
            return self
        else:
            return Graph(branch_by_node, branch_labels_map, nodes_yx_coord)

    def nodes_distance(self, *nodes_idx, close_loop=False):
        from ..seg_to_graph.graph_utilities import perimeter_from_vertices

        nodes = self._nodes_yx_coord[list(nodes_idx)]
        return perimeter_from_vertices(nodes, close_loop=close_loop)

    def terminations_nodes(self) -> npt.NDArray[int]:
        from ..seg_to_graph.graph_utilities import compute_is_endpoints

        return np.where(compute_is_endpoints(self._branch_by_node))[0]

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
        from ..seg_to_graph.graph_utilities import nodes_tangent

        assert all(0 <= n < self.nodes_count for n in nodes), f"Invalid node index: must be in [0, {self.nodes_count})."

        # Automatically infer branches_id if not provided
        if branches_id is None:
            branches_id = [None] * len(nodes)
        assert len(branches_id) == len(nodes), "nodes and branches_id must have the same length."

        if any(b is None for b in branches_id):
            for i, (n, b) in enumerate(zip(nodes, branches_id, strict=True)):
                if b is None:
                    incident_branches = np.argwhere(self._branch_by_node[:, n]).flatten()
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
