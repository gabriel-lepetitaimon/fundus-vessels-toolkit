import numpy as np


class VascularGraph:
    def __init__(self, branch_by_node: np.ndarray,
                 branch_labels_map: np.ndarray, nodes_yx_coord: tuple[np.ndarray, np.ndarray]):
        self._branch_by_node = branch_by_node
        self._nodes_yx_coord = nodes_yx_coord
        self._branch_labels_map = branch_labels_map

    def branch_connectivity_matrix(self):
        return self._branch_by_node.dot(self._branch_by_node.T)

    def node_connectivity_matrix(self):
        return self._branch_by_node.T.dot(self._branch_by_node)

    @property
    def branch_by_node(self):
        return self._branch_by_node

    @property
    def nodes_yx_coord(self):
        return self._nodes_yx_coord

    @property
    def branch_labels_map(self):
        return self._branch_labels_map

