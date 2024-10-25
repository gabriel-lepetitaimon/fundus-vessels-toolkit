__all__ = ["SegToGraph", "FundusVesselSegToGraph"]

from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

from fundus_vessels_toolkit.utils import if_none

from ..segment_to_graph import (
    GraphSimplifyArg,
    ReconnectEndpointsArg,
    SkeletonizeMethod,
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
        clean_branches_tips=20,
        min_terminal_branch_length=4,
        min_terminal_branch_calibre_ratio=1,
        max_spurs_length=30,
        simplify_graph: bool = True,
        simplify_graph_arg: Optional[GraphSimplifyArg] = None,
        parse_geometry=True,
        adaptative_tangents=True,
        bspline_target_error=1.5,
    ):
        """

        Parameters
        ----------
        skeletonize_method:
            Method to use for skeletonization. One of: 'medial_axis', 'zhang', 'lee' (default).

        fix_hollow:
            If True (by default), hollow cross pattern are filled and considered as a 4-branches junction.

        clean_branches_tips:
            If > 0, clean the skeleton extremities of each branches.
            This step ensure that the branch skeleton actually starts where the branch emerge from the junction/bifurcation and not at the center of the junction/bifurcation. Skeleton inside the junction/bifurcation is often not relevant and may affect the accuracy of tangent and calibre estimation.

            The value of ``clean_branches_tips`` is the maximum number of pixel that can be removed through this process.

        min_terminal_branch_length :
        If > 0, remove terminal branches that are shorter than this value in pixel.

        min_terminal_branch_calibre_ratio :
            If > 0, remove terminal branches that are shorter than this value times the calibre of its largest adjacent branch.

        max_spurs_length :
            If > 0, prevent the removal of a terminal branch longer than this value.

        min_orphan_branch_length:
            If > 0, remove branches that are not connected to any junction and are shorter than this value.

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

        parse_geometry:
            If True, populate the geometry of the graph with the tangent and calibre of the branches.

        adaptative_tangents:
            If True, adapt the smoothing of the branches tangents to the local calibre of the vessel.

        bspline_target_error:
            Target error for the bspline interpolation of the branches. Default is 3.
        """  # noqa: E501
        self.skeletonize_method: SkeletonizeMethod | str = skeletonize_method

        self.fix_hollow = fix_hollow
        self.clean_branches_tips = clean_branches_tips
        self.min_terminal_branch_length = min_terminal_branch_length
        self.min_terminal_branch_calibre_ratio = min_terminal_branch_calibre_ratio
        self.max_spurs_length = max_spurs_length

        self.simplify_graph = simplify_graph
        self.simplify_graph_arg = simplify_graph_arg

        self.geometry_parsing_enabled = parse_geometry
        self.adaptative_tangents = adaptative_tangents
        self.bspline_target_error = bspline_target_error

    def __call__(
        self,
        vessel_mask: npt.NDArray[np.bool_] | torch.Tensor | str | Path,
        simplify: Optional[bool] = None,
        parse_geometry: Optional[bool] = None,
    ) -> VGraph:
        """
        Extract the graph of the skeleton from a binary image.

        Parameters
        ----------
        vessel_mask : npt.NDArray[np.bool_] | torch.Tensor | str | Path
            Binary image of the vessels or path to such image.

        Returns
        -------
        VGraph
            The vascular graph
        """
        seg = self.img_to_seg(vessel_mask)
        skel = self.skeletonize(seg)
        graph = self.skel_to_vgraph(skel, vessel_mask)
        if if_none(simplify, self.simplify_graph):
            self.simplify(graph, inplace=True)
        if if_none(parse_geometry, self.geometry_parsing_enabled):
            graph = self.populate_geometry(graph, vessel_mask)
        return graph

    # --- Intermediate steps ---
    def img_to_seg(self, img: npt.ArrayLike | torch.Tensor | str | Path):
        if isinstance(img, (str, Path)):
            from ..utils.data_io import load_image

            img = load_image(img)

        if img.ndim == 3:
            r_value = np.unique(img[..., 2])
            if np.all(r_value == np.array([0, 1])):
                # Red channel is binary: the image is a segmentation mask
                img = img.sum(axis=2) > 0.5
            else:
                # Red channel is not binary: the image is a fundus image
                from ..segment import segment_vessels

                img = segment_vessels(img)
        return img

    def skeletonize(self, vessel_mask: npt.NDArray[np.bool_] | torch.Tensor) -> npt.NDArray[np.bool_] | torch.Tensor:
        from ..segment_to_graph.skeleton_parsing import detect_skeleton_nodes
        from ..segment_to_graph.skeletonize import skeletonize

        binary_skel = skeletonize(vessel_mask, method=self.skeletonize_method) > 0
        remove_endpoint_branches = self.min_terminal_branch_length > 0 or self.min_terminal_branch_calibre_ratio > 0
        return detect_skeleton_nodes(
            binary_skel, fix_hollow=self.fix_hollow, remove_endpoint_branches=remove_endpoint_branches
        )

    def skel_to_vgraph(
        self,
        skeleton_mask: npt.NDArray[np.bool_] | torch.Tensor,
        segmentation_mask: Optional[npt.NDArray[np.bool_] | torch.Tensor] = None,
    ) -> VGraph:
        from ..segment_to_graph.skeleton_parsing import skeleton_to_vgraph

        return skeleton_to_vgraph(
            skeleton_mask,
            vessels=segmentation_mask,
            fix_hollow=self.fix_hollow,
            clean_branches_tips=self.clean_branches_tips,
            min_terminal_branch_length=self.min_terminal_branch_length,
            min_terminal_branch_calibre_ratio=self.min_terminal_branch_calibre_ratio,
            max_spurs_length=self.max_spurs_length,
        )

    def simplify(self, graph: VGraph, inplace=False):
        from ..segment_to_graph.graph_simplification import simplify_graph

        return simplify_graph(
            graph,
            self.simplify_graph_arg,
            inplace=inplace,
        )

    def populate_geometry(
        self, graph: VGraph, vessels_segmentation: npt.NDArray[np.bool_] | torch.Tensor, inplace=False
    ):
        from ..segment_to_graph.geometry_parsing import populate_geometry

        return populate_geometry(
            graph,
            vessels_segmentation,
            adaptative_tangents=self.adaptative_tangents,
            bspline_target_error=self.bspline_target_error,
            inplace=inplace,
        )

    # --- Utility methods ---
    def from_skel(
        self,
        skel: npt.NDArray[np.bool_] | torch.Tensor,
        vessels: npt.NDArray[np.bool_] | torch.Tensor,
        simplify: bool = None,
        populate_geometry: bool = None,
    ) -> VGraph:
        vgraph = self.skel_to_vgraph(skel, vessels)
        if if_none(simplify, self.simplify_graph):
            self.simplify(vgraph, inplace=True)

        if if_none(populate_geometry, self.geometry_parsing_enabled):
            vgraph = self.populate_geometry(vgraph, vessels)

        return vgraph


class FundusVesselSegToGraph(SegToGraph):
    """
    Specialization of Seg2Graph for retinal vessels.
    """

    def __init__(self, max_vessel_diameter=20, prevent_node_simplification_on_borders=35, parse_geometry=True):
        super(FundusVesselSegToGraph, self).__init__(
            fix_hollow=True,
            skeletonize_method="lee",
            min_terminal_branch_length=4,
            min_terminal_branch_calibre_ratio=1,
            parse_geometry=parse_geometry,
        )
        self.max_vessel_diameter = max_vessel_diameter
        self.prevent_node_simplification_on_borders = prevent_node_simplification_on_borders

    @property
    def max_vessel_diameter(self):
        return self._max_vessel_diameter

    @max_vessel_diameter.setter
    def max_vessel_diameter(self, diameter):
        self._max_vessel_diameter = diameter

        self.simplify_graph_arg = GraphSimplifyArg(
            max_spurs_length=0,
            # reconnect_endpoints=ReconnectEndpointsArg(
            #    max_distance=diameter * 2, max_angle=20, intercept_snapping_distance=diameter / 2
            # ),
            max_cycles_length=diameter,
            junctions_merge_distance=diameter * 2 / 3,
            min_orphan_branches_length=diameter * 1.5,
            simplify_topology="node",
            passing_node_min_angle=110,
        )

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
