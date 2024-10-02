__all__ = ["AVSegToTree", "FundusAVSegToTree"]

from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import ior
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from ..segment_to_graph import (
    NodeMergeDistanceParam,
    NodeMergeDistances,
    SimplifyTopology,
    SkeletonizeMethod,
)
from ..utils import if_none
from ..utils.geometric import Point
from ..vascular_data_objects import AVLabel, FundusData, VGraph, VTree
from .seg_to_graph import SegToGraph


class AVSegToTreeBase(metaclass=ABCMeta):
    def __init__(self, mask_optic_disc=True):
        self.mask_optic_disc = mask_optic_disc

    @abstractmethod
    def __call__(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Tuple[VTree, VTree]:
        pass

    def prepare_data(
        self,
        fundus: FundusData,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> FundusData:
        if fundus is None:
            if av is None or od is None:
                raise ValueError("Either `fundus` or `av` and `od` must be provided.")
            fundus = FundusData(vessels=av, od=od)
        else:
            fundus = fundus.update(vessels=av, od=od)

        if self.mask_optic_disc:
            av_seg = fundus.av.copy()
            av_seg[fundus.od] = 0
            fundus = fundus.update(vessels=av_seg)

        return fundus

    @abstractmethod
    def to_vgraph(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
        populate_geometry: Optional[bool] = None,
    ) -> VGraph: ...


class AVSegToTree(AVSegToTreeBase):
    def __init__(
        self,
        skeletonize_method: SkeletonizeMethod | str = "lee",
        fix_hollow=True,
        clean_branches_tips=20,
        min_terminal_branch_length=4,
        min_terminal_branch_calibre_ratio=1,
        max_spurs_length=30,
        min_orphan_branch_length=0,
        nodes_merge_distance: NodeMergeDistanceParam = True,
        merge_small_cycles: float = 0,
        simplify_topology: SimplifyTopology = "node",
        parse_geometry=True,
        adaptative_tangents=False,
        bspline_target_error=3,
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
        super(AVSegToTree, self).__init__()

        self.skeletonize_method: SkeletonizeMethod | str = skeletonize_method

        self.fix_hollow = fix_hollow
        self.clean_branches_tips = clean_branches_tips
        self.min_terminal_branch_length = min_terminal_branch_length
        self.min_terminal_branch_calibre_ratio = min_terminal_branch_calibre_ratio
        self.max_spurs_length = max_spurs_length

        self.assign_ratio_threshold = 4 / 5
        self.assign_split_branch = True
        self.propagate_assigned_labels = True

        self.simplification_enabled = True
        self.min_orphan_branch_length = min_orphan_branch_length
        self.nodes_merge_distance = nodes_merge_distance
        self.iterative_nodes_merge = True
        self.merge_small_cycles = merge_small_cycles
        self.simplify_topology = simplify_topology
        self.node_simplification_criteria = None

        self.geometry_parsing_enabled = parse_geometry
        self.adaptative_tangents = adaptative_tangents
        self.bspline_target_error = bspline_target_error

    def __call__(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Tuple[VGraph, VGraph]:
        fundus = self.prepare_data(fundus, av=av, od=od)

        vessel_skeleton = self.skeletonize(fundus.vessels)
        graph = self.skel_to_vgraph(vessel_skeleton, fundus.vessels)
        self.assign_av_labels(graph, fundus.av, inplace=True)
        graphs = self.split_av_graph(graph)
        trees = []
        for g in graphs:
            trees.append(self.vgraph_to_vtree(g, fundus.od_center))

        if self.geometry_parsing_enabled:
            for tree in trees:
                self.populate_geometry(tree, fundus, inplace=True)

        return trees

    # --- Intermediate steps ---
    def skeletonize(self, vessels: npt.NDArray[np.bool_] | torch.Tensor) -> npt.NDArray[np.bool_] | torch.Tensor:
        from ..segment_to_graph.skeleton_parsing import detect_skeleton_nodes
        from ..segment_to_graph.skeletonize import skeletonize

        binary_skel = skeletonize(vessels > 0, method=self.skeletonize_method) > 0
        remove_endpoint_branches = self.min_terminal_branch_length > 0 or self.min_terminal_branch_calibre_ratio > 0
        return detect_skeleton_nodes(
            binary_skel, fix_hollow=self.fix_hollow, remove_endpoint_branches=remove_endpoint_branches
        )

    def skel_to_vgraph(
        self,
        skeleton: npt.NDArray[np.bool_] | torch.Tensor,
        vessels: Optional[npt.NDArray[np.bool_] | torch.Tensor | FundusData] = None,
    ) -> VGraph:
        from ..segment_to_graph.graph_simplification import simplify_graph
        from ..segment_to_graph.skeleton_parsing import skeleton_to_vgraph

        graph = skeleton_to_vgraph(
            skeleton,
            vessels=vessels,
            fix_hollow=self.fix_hollow,
            clean_branches_tips=self.clean_branches_tips,
            min_terminal_branch_length=self.min_terminal_branch_length,
            min_terminal_branch_calibre_ratio=self.min_terminal_branch_calibre_ratio,
            max_spurs_length=self.max_spurs_length,
        )
        simplify_graph(
            graph,
            nodes_merge_distance=NodeMergeDistances(junction=0, tip=0, node=0),
            max_cycles_length=self.merge_small_cycles,
            simplify_topology=None,
            min_orphan_branches_length=self.min_orphan_branch_length,
            # node_simplification_criteria=self.node_simplification_criteria,
            inplace=True,
        )

        return graph

    def assign_av_labels(
        self, graph: VGraph, av_map: npt.NDArray[np.int_], av_attr: Optional[str] = None, inplace: bool = False
    ) -> VGraph:
        from ..segment_to_graph.av_tree_parsing import assign_av_label

        return assign_av_label(
            graph,
            av_map=av_map,
            ratio_threshold=self.assign_ratio_threshold,
            split_av_branch=self.assign_split_branch,
            av_attr=if_none(av_attr, "av"),
            propagate_labels=self.propagate_assigned_labels,
            inplace=inplace,
        )

    def split_av_graph(self, graph: VGraph, av_attr: Optional[str] = None) -> Tuple[VGraph, VGraph]:
        from ..segment_to_graph.av_tree_parsing import av_split

        return av_split(graph, av_attr=if_none(av_attr, "av"))

    def populate_geometry(self, graph: VGraph, fundus: FundusData, inplace: bool = False) -> VGraph:
        from ..segment_to_graph.geometry_parsing import populate_geometry

        return populate_geometry(
            graph,
            fundus.vessels,
            adaptative_tangents=self.adaptative_tangents,
            bspline_target_error=self.bspline_target_error,
            inplace=inplace,
        )

    def vgraph_to_vtree(self, graph: VGraph, od_pos: Point) -> VTree:
        from ..segment_to_graph.av_tree_parsing import clean_vtree, simplify_av_graph, vgraph_to_vtree

        graph = simplify_av_graph(graph, inplace=False)
        tree = vgraph_to_vtree(graph, od_pos)
        return clean_vtree(tree)

    # --- Utility methods ---
    def to_vgraph(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
        populate_geometry: Optional[bool] = None,
    ) -> VGraph:
        fundus = self.prepare_data(fundus, av=av, od=od)

        vessel_skeleton = self.skeletonize(fundus.vessels)
        graph = self.skel_to_vgraph(vessel_skeleton, fundus)
        self.assign_av_labels(graph, fundus, inplace=True)
        if if_none(populate_geometry, self.geometry_parsing_enabled):
            self.populate_geometry(graph, fundus, inplace=True)
        return graph


FUNDUS_SEG_TO_GRAPH = SegToGraph(
    skeletonize_method="lee",
    fix_hollow=True,
    min_terminal_branch_length=4,
    min_terminal_branch_calibre_ratio=1,
    max_spurs_length=30,
    nodes_merge_distance=False,
    min_orphan_branch_length=30,
    clean_branches_tips=30,
    merge_small_cycles=20,
    simplify_topology=None,
)


class NaiveAVSegToTree(AVSegToTreeBase):
    def __init__(
        self,
        segToGraph: SegToGraph = FUNDUS_SEG_TO_GRAPH,
    ):
        """

        Parameters
        ----------
        segToGraph: SegToGraph
            The SegToGraph instance to use for the segmentation to graph step.
        """  # noqa: E501
        super(NaiveAVSegToTree, self).__init__()
        self.segToGraph = segToGraph

    def __call__(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Tuple[VGraph, VGraph]:
        fundus = self.prepare_data(fundus, av=av, od=od)

        av_skeleton = self.av_skeletonize(fundus.av)

        graphs = [
            self.skel_to_vgraph(av_skeleton, fundus.av, simplify=True, populate_geometry=True, **art_vei)
            for art_vei in [{"artery": True, "vein": False}, {"artery": False, "vein": True}]
        ]

        trees = [self.vgraph_to_vtree(g, fundus.od_center) for g in graphs]
        return trees

    # --- Intermediate steps ---
    def av_skeletonize(
        self, av: npt.NDArray[np.int_] | torch.Tensor | FundusData
    ) -> npt.NDArray[np.bool_] | torch.Tensor:
        av = av.av if isinstance(av, FundusData) else av
        art = reduce(ior, (av == label for label in (AVLabel.ART, AVLabel.BOTH)))
        vei = reduce(ior, (av == label for label in (AVLabel.VEI, AVLabel.BOTH)))
        art_skel = self.segToGraph.skeletonize(art)
        vei_skel = self.segToGraph.skeletonize(vei)
        skel = art_skel.astype(int)
        skel[art_skel > 0] = AVLabel.ART
        skel[vei_skel > 0] = AVLabel.VEI
        skel[(art_skel > 0) & (vei_skel > 0)] = AVLabel.BOTH
        return skel

    def skel_to_vgraph(
        self,
        skeleton: npt.NDArray[np.int_] | torch.Tensor,
        vessels: Optional[npt.NDArray[np.bool_] | torch.Tensor | FundusData] = None,
        artery: Optional[bool] = None,
        vein: Optional[bool] = None,
        unknown: Optional[bool] = None,
        simplify: bool = True,
        populate_geometry: bool = True,
    ) -> VGraph:
        avlabel = AVLabel.select_label(artery=artery, vein=vein, unknown=unknown)

        skel = reduce(ior, (skeleton == label for label in avlabel))
        vessels = vessels.av if isinstance(vessels, FundusData) else vessels
        vessels = reduce(ior, (vessels == label for label in avlabel))

        vgraph = self.segToGraph.skel_to_vgraph(skel, vessels)
        if simplify:
            self.segToGraph.simplify(vgraph, inplace=True)
        if populate_geometry:
            self.segToGraph.populate_geometry(vgraph, vessels, inplace=True)

        return vgraph

    def vgraph_to_vtree(self, graph: VGraph, od_pos: Point) -> VTree:
        from ..segment_to_graph.av_tree_parsing import clean_vtree, vgraph_to_vtree

        tree = vgraph_to_vtree(graph, od_pos)
        tree = clean_vtree(tree)
        return tree

    # --- Utility methods ---
    def to_vgraph(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Tuple[VGraph, VGraph]:
        fundus = self.prepare_data(fundus, av=av, od=od)

        av_skeleton = self.av_skeletonize(fundus.av)

        a_graph = self.skel_to_vgraph(av_skeleton, fundus.av, artery=True, simplify=True, populate_geometry=True)
        v_graph = self.skel_to_vgraph(av_skeleton, fundus.av, vein=True, simplify=True, populate_geometry=True)

        return a_graph, v_graph


class FundusAVSegToTree(AVSegToTree):
    """
    Specialization of AVSegToTree for retinal vessels.
    """

    def __init__(self, max_vessel_diameter=20, parse_geometry=True):
        super(FundusAVSegToTree, self).__init__(
            fix_hollow=True,
            skeletonize_method="lee",
            min_terminal_branch_length=4,
            min_terminal_branch_calibre_ratio=1,
            simplify_topology="node",
            parse_geometry=parse_geometry,
        )
        self.max_vessel_diameter = max_vessel_diameter

    @property
    def max_vessel_diameter(self):
        return self._max_vessel_diameter

    @max_vessel_diameter.setter
    def max_vessel_diameter(self, diameter):
        self._max_vessel_diameter = diameter

        self.min_orphan_branch_length = diameter * 1.5
        self.clean_branches_tips = diameter * 1.5
        self.nodes_merge_distance = NodeMergeDistances(junction=diameter * 2 / 3, tip=diameter, node=0)
        self.merge_small_cycles = diameter
        self.max_spurs_length = diameter * 1.5
