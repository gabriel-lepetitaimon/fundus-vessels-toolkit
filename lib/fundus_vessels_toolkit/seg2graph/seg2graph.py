########################################################################################################################
#   *** VESSELS SKELETONIZATION AND GRAPH CREATION FROM SEGMENTATION ***
#   This module provides functions to extract the vascular graph from a binary image of retinal vessels.
#   Two implementation are provided: one using numpy array, scipy.ndimage and scikit-image,
#   the other using pytorch and kornia.
#
#
########################################################################################################################
from typing import Mapping, Literal

from .skeletonization import skeletonize, SkeletonizeMethod
from .graph_extraction import compute_adjacency_branches_nodes, branches_by_nodes_to_node_graph, \
                              NodeMergeDistanceParam, SimplifyTopology


class Seg2Graph:
    """
    Utility class to extract the graph of the skeleton from a binary image.
    """
    def __init__(self,
                 skeletonize_method: SkeletonizeMethod = 'lee',
                 fix_hollow=True,
                 max_spurs_length: int = 1,
                 max_spurs_distance: float = 0,
                 nodes_merge_distance: NodeMergeDistanceParam = True,
                 merge_small_cycles: float = 0,
                 simplify_topology: SimplifyTopology = 'node'):
        """
        Args:
            skeletonize_method: Method to use for skeletonization. (One of: 'medial_axis', 'zhang', 'lee'. Default: 'lee'.)
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

    def __call__(self, vessel_map):
        self.seg2node_graph(vessel_map)

    def seg2node_graph(self, vessel_map, return_label=False):
        skel = self.skeletonize(vessel_map)
        adj_mtx, branch_labels, (node_y, node_x) = self.skel2adjacency(skel, return_label=True)
        if return_label:
            return branches_by_nodes_to_node_graph(adj_mtx, (node_y, node_x)), branch_labels
        else:
            return branches_by_nodes_to_node_graph(adj_mtx, (node_y, node_x))

    # --- Intermediate steps ---
    def skeletonize(self, vessel_map):
        return skeletonize(vessel_map, fix_hollow=self.fix_hollow, max_spurs_length=self.max_spurs_length,
                           skeletonize_method=self.skeletonize_method)

    def skel2adjacency(self, skel, return_label=False):
        return compute_adjacency_branches_nodes(skel, return_label=return_label,
                                                max_spurs_distance=self.max_spurs_distance,
                                                nodes_merge_distance=self.nodes_merge_distance,
                                                merge_small_cycles=self.merge_small_cycles,
                                                simplify_topology=self.simplify_topology)

    def seg2adjacency(self, vessel_map, return_label=False):
        skel = self.skeletonize(vessel_map)
        return self.skel2adjacency(skel, return_label=return_label)


class RetinalVesselSeg2Graph(Seg2Graph):
    """
    Specialization of Seg2Graph for retinal vessels.
    """
    def __init__(self, max_vessel_diameter=5):
        max_radius = max_vessel_diameter // 2 + 1
        merge_distance = dict(termination=max_radius, junction=max_vessel_diameter, node=max_radius-1)
        super(RetinalVesselSeg2Graph, self).__init__(fix_hollow=True,
                                                     skeletonize_method='lee',
                                                     max_spurs_length=1,
                                                     max_spurs_distance=max_vessel_diameter-2,
                                                     nodes_merge_distance=merge_distance,
                                                     merge_small_cycles=max_vessel_diameter * 3,
                                                     simplify_topology='node', )

