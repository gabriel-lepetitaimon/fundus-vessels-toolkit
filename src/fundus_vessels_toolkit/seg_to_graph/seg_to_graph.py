########################################################################################################################
#   *** VESSELS SKELETONIZATION AND GRAPH CREATION FROM SEGMENTATION ***
#   This module provides a helper class to extract the vascular graph from a binary image of retinal vessels.
#
########################################################################################################################
import numpy as np

from ..utils.graph.branch_by_nodes import branches_by_nodes_to_node_graph
from .graph_extraction import (
    NodeMergeDistanceParam,
    SimplifyTopology,
    seg_to_branches_list,
)
from .skeletonization import SkeletonizeMethod, skeletonize


class SegToGraph:
    """
    Utility class to extract the graph of the skeleton from a binary image.
    """

    def __init__(
        self,
        skeletonize_method: SkeletonizeMethod = "lee",
        fix_hollow=True,
        max_spurs_length: int = 1,
        max_spurs_distance: float = 0,
        nodes_merge_distance: NodeMergeDistanceParam = True,
        merge_small_cycles: float = 0,
        simplify_topology: SimplifyTopology = "node",
    ):
        """
        Args:
            skeletonize_method: Method to use for skeletonization. One of: 'medial_axis', 'zhang', 'lee' (default).
            fix_hollow: If True, fill hollow cross pattern.
            max_spurs_length: Remove terminal branches whose length (number of pixel of the branch) is inferior to this.
                                Default value is 1 because 1px branches should be removed as they will be ignored in the
                                graph construction.
            max_spurs_distance: Remove terminal branches whose distance between nodes is inferior to this. Default: 0.
                                This is different from max_spurs_length as it is computed on the shortest distance
                                between the branches nodes, and not on the actual branch length which is usually longer.
                                It's also faster as it uses the graph representation and not the map of branches labels.
            nodes_merge_distance: If True, merge nodes that are closer than 2 pixels. If False, do not merge nodes.
                                    If an integer, merge nodes that are closer than this distance.  Default: True.
            merge_small_cycles: Remove cycles whose size is inferior to this. Default: 0.
            simplify_topology: If True, remove nodes that are not junctions or endpoints. Default: True.
        """
        self.fix_hollow = fix_hollow
        self.max_spurs_length = max_spurs_length
        self.skeletonize_method = skeletonize_method

        self.max_spurs_distance = max_spurs_distance
        self.nodes_merge_distance = nodes_merge_distance
        self.merge_small_cycles = merge_small_cycles
        self.simplify_topology = simplify_topology
        self.node_simplification_criteria = None

    def __call__(self, vessel_map):
        self.seg2node_graph(vessel_map)

    def seg2node_graph(self, vessel_map, return_label=False):
        skel = self.skeletonize(vessel_map)
        adj_mtx, branch_labels, (node_y, node_x) = self.skel2adjacency(skel, return_label=True)
        if return_label:
            return (
                branches_by_nodes_to_node_graph(adj_mtx, (node_y, node_x)),
                branch_labels,
            )
        else:
            return branches_by_nodes_to_node_graph(adj_mtx, (node_y, node_x))

    # --- Intermediate steps ---
    def skeletonize(self, vessel_map):
        return skeletonize(
            vessel_map,
            fix_hollow=self.fix_hollow,
            max_spurs_length=self.max_spurs_length,
            skeletonize_method=self.skeletonize_method,
        )

    def skel2adjacency(self, skel, return_label=False):
        return seg_to_branches_list(
            skel,
            return_label=return_label,
            max_spurs_distance=self.max_spurs_distance,
            nodes_merge_distance=self.nodes_merge_distance,
            merge_small_cycles=self.merge_small_cycles,
            simplify_topology=self.simplify_topology,
            node_simplification_criteria=self.node_simplification_criteria,
        )

    def seg2adjacency(self, vessel_map, return_label=False):
        skel = self.skeletonize(vessel_map)
        return self.skel2adjacency(skel, return_label=return_label)


class RetinalVesselSegToGraph(SegToGraph):
    """
    Specialization of Seg2Graph for retinal vessels.
    """

    def __init__(self, max_vessel_diameter=5, prevent_node_simplification_on_borders=35):
        super(RetinalVesselSegToGraph, self).__init__(
            fix_hollow=True,
            skeletonize_method="lee",
            max_spurs_length=1,
            simplify_topology="node",
        )
        self.max_vessel_diameter = max_vessel_diameter
        self.prevent_node_simplification_on_borders = prevent_node_simplification_on_borders

    def __call__(self, vessel_map):
        from ..vgraph import VascularGraph

        adj_mtx, branch_labels, node_yx_coord = self.seg2adjacency(vessel_map, return_label=True)
        return VascularGraph(adj_mtx, branch_labels, node_yx_coord)

    def skel2vgraph(self, vessel_skeleton):
        from ..vgraph import VascularGraph

        adj_mtx, branch_labels, node_yx_coord = self.skel2adjacency(skel=vessel_skeleton, return_label=True)
        return VascularGraph(adj_mtx, branch_labels, node_yx_coord)

    @property
    def max_vessel_diameter(self):
        return self._max_vessel_diameter

    @max_vessel_diameter.setter
    def max_vessel_diameter(self, diameter):
        self._max_vessel_diameter = diameter

        max_radius = diameter // 2 + 1
        self.max_spurs_distance = diameter
        self.nodes_merge_distance = dict(junction=max_radius, termination=0, node=max_radius - 1)
        self.merge_small_cycles = diameter

    @property
    def prevent_node_simplification_on_borders(self):
        return self._node_simplification_criteria

    @prevent_node_simplification_on_borders.setter
    def prevent_node_simplification_on_borders(self, prevent: bool | int):
        if prevent:
            if prevent is True:
                prevent = 35

            def criteria(node, node_y, node_x, skeleton, branches_by_nodes_adj):
                h, w = skeleton.shape

                return (
                    ((node_y - h / 2) ** 2 + (node_x - w / 2) ** 2 < (max(h, w) / 2 - prevent) ** 2)
                    & (node_y > prevent)
                    & (node_y < h - prevent)
                    & (node_x > prevent)
                    & (node_x < w - prevent)
                    & node
                )

            self.node_simplification_criteria = criteria
        else:
            self.node_simplification_criteria = None
        self._prevent_node_simplification_on_borders = prevent
