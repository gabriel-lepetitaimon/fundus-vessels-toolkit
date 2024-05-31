__all__ = ["SegToGraph", "FundusVesselSegToGraph"]

import numpy as np
import torch

from ..seg_to_graph.graph_simplification import NodeMergeDistanceParam, SimplifyTopology, simplify_graph
from ..seg_to_graph.skeletonize import SkeletonizeMethod
from ..vgraph import Graph


class SegToGraph:
    """
    Utility class to extract the graph of the skeleton from a binary image.
    """

    def __init__(
        self,
        skeletonize_method: SkeletonizeMethod | str = "lee",
        fix_hollow=True,
        remove_endpoint_branches: bool = True,
        max_spurs_distance: float = 0,
        nodes_merge_distance: NodeMergeDistanceParam = True,
        merge_small_cycles: float = 0,
        simplify_topology: SimplifyTopology = "node",
        legacy_implementation: bool = False,
    ):
        """
        Args:
            skeletonize_method: Method to use for skeletonization. One of: 'medial_axis', 'zhang', 'lee' (default).
            fix_hollow: If True, fill hollow cross pattern.
            remove_endpoint_branches: Remove terminal branches that are made of only an endpoint. Default: True.
            max_spurs_distance: Remove terminal branches whose distance between nodes is inferior to this. Default: 0.
                                This is different from max_spurs_length as it is computed on the shortest distance
                                between the branches nodes, and not on the actual branch length which is usually longer.
                                It's also faster as it uses the graph representation and not the map of branches labels.
            nodes_merge_distance: If True, merge nodes that are closer than 2 pixels. If False, do not merge nodes.
                                    If an integer, merge nodes that are closer than this distance.  Default: True.
            merge_small_cycles: Remove cycles whose size is inferior to this. Default: 0.
            simplify_topology: If True, remove nodes that are not junctions or endpoints. Default: True.
        """

        self.skeletonize_method: SkeletonizeMethod | str = skeletonize_method

        self.fix_hollow = fix_hollow
        self.remove_endpoint_branches = remove_endpoint_branches

        self.max_spurs_distance = max_spurs_distance
        self.nodes_merge_distance = nodes_merge_distance
        self.merge_small_cycles = merge_small_cycles
        self.simplify_topology = simplify_topology
        self.node_simplification_criteria = None

        self.legacy = legacy_implementation

    def __call__(self, vessel_map):
        skel = self.skeletonize(vessel_map)
        graph = self.parse_skeleton_to_graph(skel)
        return self.simplify_graph(graph)

    # --- Intermediate steps ---
    def skeletonize(self, vessel_map):
        from ..seg_to_graph.skeletonize import skeletonize

        return skeletonize(vessel_map, method=self.skeletonize_method)

    def parse_skeleton_to_graph(self, skeleton_mask: np.ndarray) -> np.ndarray:
        if self.legacy:
            from ..seg_to_graph.skeleton_parsing import detect_skeleton_nodes_legacy as detect_skeleton_nodes
            from ..seg_to_graph.skeleton_parsing import parse_skeleton_legacy as parse_skeleton

        else:
            from ..seg_to_graph.skeleton_parsing import detect_skeleton_nodes, parse_skeleton

            if isinstance(skeleton_mask, np.ndarray):
                skeleton_mask = torch.from_numpy(skeleton_mask)

        branch_by_node, branch_labels, nodes_yx = parse_skeleton(
            detect_skeleton_nodes(
                skeleton_mask, fix_hollow=self.fix_hollow, remove_endpoint_branches=self.remove_endpoint_branches
            )
        )
        return Graph(branch_by_node, branch_labels, nodes_yx)

    def simplify_graph(self, graph: Graph):
        return simplify_graph(
            graph,
            max_spurs_distance=self.max_spurs_distance,
            nodes_merge_distance=self.nodes_merge_distance,
            merge_small_cycles=self.merge_small_cycles,
            simplify_topology=self.simplify_topology,
            node_simplification_criteria=self.node_simplification_criteria,
        )


class FundusVesselSegToGraph(SegToGraph):
    """
    Specialization of Seg2Graph for retinal vessels.
    """

    def __init__(self, max_vessel_diameter=5, prevent_node_simplification_on_borders=35):
        super(FundusVesselSegToGraph, self).__init__(
            fix_hollow=True,
            skeletonize_method="lee",
            remove_endpoint_branches=1,
            simplify_topology="node",
        )
        self.max_vessel_diameter = max_vessel_diameter
        self.prevent_node_simplification_on_borders = prevent_node_simplification_on_borders

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
