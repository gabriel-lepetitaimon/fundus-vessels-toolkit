import numpy as np


class VascularGraph:
    def __init__(
        self,
        branch_by_node: np.ndarray,
        branch_labels_map: np.ndarray,
        nodes_yx_coord: tuple[np.ndarray, np.ndarray],
    ):
        self._branch_by_node = branch_by_node
        if isinstance(nodes_yx_coord, tuple):
            nodes_yx_coord = np.asarray(nodes_yx_coord)
            if nodes_yx_coord.shape[0] == 2 and nodes_yx_coord.shape[1] != 2:
                nodes_yx_coord = nodes_yx_coord.T
        self._nodes_yx_coord = nodes_yx_coord
        self._branch_labels_map = branch_labels_map

    @property
    def skeleton(self):
        return self._branch_labels_map > 0

    def branch_adjacency_matrix(self):
        return self._branch_by_node.dot(self._branch_by_node.T)

    def node_adjacency_matrix(self):
        return self._branch_by_node.T.dot(self._branch_by_node)

    def node_adjacency_list(self):
        from ..seg2graph.graph_utilities import branch_by_nodes_to_adjacency_list

        return branch_by_nodes_to_adjacency_list(self._branch_by_node)

    def jppype_layer(self, edge_labels=False, node_labels=False, edge_map=True):
        from jppype.layers_2d import LayerGraph

        l = LayerGraph(
            self.node_adjacency_list(), self._nodes_yx_coord, self._branch_labels_map
        )
        l.set_options(
            {
                "edge_labels_visible": edge_labels,
                "node_labels_visible": node_labels,
                "edge_map_visible": edge_map,
            }
        )
        return l

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
        node_indexes = np.asarray(node_indexes, dtype=np.int)
        assert (
            node_indexes.ndim == 1 and node_indexes.shape[0] <= self.nodes_count
        ), f"node_indexes must be a 1D array of maximum size {self.nodes_count}"

        if node_indexes.shape[0] < self._branch_by_node.shape[1]:
            node_indexes = np.concatenate(
                (node_indexes, np.setdiff1d(np.arange(self.nodes_count), node_indexes))
            )

        self._branch_by_node = self._branch_by_node[:, node_indexes]
        self._nodes_yx_coord = self._nodes_yx_coord[node_indexes]
        return self

    def nodes_distance(self, *nodes_idx, close_loop=False):
        from ..seg2graph.graph_utilities import perimeter_from_vertices

        nodes = self._nodes_yx_coord[list(nodes_idx)]
        return perimeter_from_vertices(nodes, close_loop=close_loop)
