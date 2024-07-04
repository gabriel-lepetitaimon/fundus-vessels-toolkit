__all__ = ["SegToGraph", "FundusVesselSegToGraph"]

from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from ..segment_to_graph import (
    NodeMergeDistanceParam,
    NodeMergeDistances,
    SimplifyTopology,
    SkeletonizeMethod,
    simplify_graph,
    skeleton_to_vgraph,
)
from ..vascular_data_objects import VGraph


class SegToGraph:
    """
    Utility class to extract the graph of the skeleton from a binary image.
    """

    def __init__(
        self,
        skeletonize_method: SkeletonizeMethod | str = "lee",
        fix_hollow=True,
        clean_branches_extremities=20,
        max_spurs_length=0,
        max_spurs_calibre_factor=1,
        nodes_merge_distance: NodeMergeDistanceParam = True,
        merge_small_cycles: float = 0,
        simplify_topology: SimplifyTopology = "node",
    ):
        """

        Parameters
        ----------
        skeletonize_method:
            Method to use for skeletonization. One of: 'medial_axis', 'zhang', 'lee' (default).

        fix_hollow:
            If True (by default), hollow cross pattern are filled and considered as a 4-branches junction.

        clean_branches_extremities:
            If > 0, clean the skeleton extremities of each branches.
            This step ensure that the branch skeleton actually starts where the branch emerge from the junction/bifurcation and not at the center of the junction/bifurcation. Skeleton inside the junction/bifurcation is often not relevant and may affect the accuracy of tangent and calibre estimation.

            The value of ``clean_branches_extremities`` is the maximum number of pixel that can be removed through this process.

        max_spurs_length:
            If > 0, remove spurs (short terminal branches) that are shorter than this value in pixel.

        max_spurs_calibre_factor:
            If > 0, remove spurs (short terminal branches) that are shorter than ``max_spurs_calibre_factor`` times the calibre of the largest branch adjacent to the spurs.

        nodes_merge_distance:
            - If True (by default), merge nodes that are closer than 2 pixels.
            - If False, do not merge nodes.
            - If an integer, merge nodes that are closer than this distance.

        merge_small_cycles:
            Remove cycles whose size is inferior to this. Default: 0.

        simplify_topology:
            - If True or 'both', remove nodes that are not junctions or endpoints.
            - If False, do not remove any node.
            - If 'node', remove nodes that are not junctions.
            - If 'junction', remove nodes that are not junctions or endpoints.
        """  # noqa: E501
        self.skeletonize_method: SkeletonizeMethod | str = skeletonize_method

        self.fix_hollow = fix_hollow
        self.clean_branches_extremities = clean_branches_extremities
        self.max_spurs_length = max_spurs_length
        self.max_spurs_calibre_factor = max_spurs_calibre_factor

        self.nodes_merge_distance = nodes_merge_distance
        self.merge_small_cycles = merge_small_cycles
        self.simplify_topology = simplify_topology
        self.node_simplification_criteria = None

    def __call__(
        self,
        vessel_mask: npt.NDArray[np.bool_] | torch.Tensor,
    ) -> VGraph:
        skel = self.skeletonize(vessel_mask)
        graph = self.skel_to_vgraph(skel, vessel_mask)
        return self.simplify_vgraph(graph)

    # --- Intermediate steps ---
    def skeletonize(self, vessel_mask: npt.NDArray[np.bool_] | torch.Tensor) -> npt.NDArray[np.bool_] | torch.Tensor:
        from ..segment_to_graph.skeletonize import skeletonize

        return skeletonize(vessel_mask, method=self.skeletonize_method) > 0

    def skel_to_vgraph(
        self,
        skeleton_mask: npt.NDArray[np.bool_] | torch.Tensor,
        segmentation_mask: Optional[npt.NDArray[np.bool_] | torch.Tensor] = None,
    ) -> VGraph:
        return skeleton_to_vgraph(
            skeleton_mask,
            segmentation_map=segmentation_mask,
            fix_hollow=self.fix_hollow,
            clean_branches_extremities=self.clean_branches_extremities,
            max_spurs_length=self.max_spurs_length,
            max_spurs_calibre_factor=self.max_spurs_calibre_factor,
        )

    def simplify_vgraph(self, graph: VGraph):
        return simplify_graph(
            graph,
            nodes_merge_distance=self.nodes_merge_distance,
            max_cycles_length=self.merge_small_cycles,
            simplify_topology=self.simplify_topology,
            # node_simplification_criteria=self.node_simplification_criteria,
        )


class FundusVesselSegToGraph(SegToGraph):
    """
    Specialization of Seg2Graph for retinal vessels.
    """

    def __init__(self, max_vessel_diameter=5, prevent_node_simplification_on_borders=35):
        super(FundusVesselSegToGraph, self).__init__(
            fix_hollow=True,
            skeletonize_method="lee",
            max_spurs_length=0,
            max_spurs_calibre_factor=1,
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

        self.clean_branches_extremities = diameter * 1.5
        self.nodes_merge_distance = NodeMergeDistances(junction=diameter, termination=diameter, node=0)
        self.merge_small_cycles = diameter

    @property
    def prevent_node_simplification_on_borders(self):
        return self._prevent_node_simplification_on_borders

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
