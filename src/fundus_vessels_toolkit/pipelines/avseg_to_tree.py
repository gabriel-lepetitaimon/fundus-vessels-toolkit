__all__ = ["AVSegToTreeBase", "NaiveAVSegToTree"]

from abc import ABCMeta, abstractmethod
from functools import reduce
from operator import ior
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from ..segment_to_graph.graph_simplification import GraphSimplifyArg
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
        fundus: FundusData | None,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> FundusData:
        if fundus is None:
            if av is None or od is None:
                raise ValueError("Either `fundus` or `av` and `od` must be provided.")
            fundus = FundusData(vessels=av, od=od)
        else:
            fundus = fundus.update(vessels=av, od=od)
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


MINIMAL_FUNDUS_SEG_TO_GRAPH = SegToGraph(
    skeletonize_method="lee",
    fix_hollow=True,
    clean_branches_tips=30,
    min_terminal_branch_length=4,
    min_terminal_branch_calibre_ratio=1,
    simplify_graph_arg=GraphSimplifyArg(
        max_spurs_length=0,
        reconnect_endpoints=False,
        junctions_merge_distance=1,
        min_orphan_branches_length=3,
        max_cycles_length=3,
        simplify_topology=False,
    ),
    parse_geometry=True,
    adaptative_tangents=True,
)


class AVSegToTree(AVSegToTreeBase):
    def __init__(
        self,
        segToGraph: Optional[SegToGraph] = None,
    ):
        """

        Parameters
        ----------
        segToGraph: SegToGraph
            The SegToGraph instance to use for the segmentation to graph step.
        """

        super(AVSegToTree, self).__init__()
        self.segToGraph = if_none(segToGraph, MINIMAL_FUNDUS_SEG_TO_GRAPH)
        self.av_attr = "av"

    def __call__(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Tuple[VTree, VTree]:
        fundus = self.prepare_data(fundus, av=av, od=od)
        if fundus.od_center is None:
            raise NotImplementedError("Parsing tree of image without optic disc is not implemented.")
        graph = self.to_vgraph(fundus, simplify=True)
        lines_digraph_info = self.build_line_digraph(graph, fundus, inplace=True)
        tree = self.resolve_digraph_to_vtree(*lines_digraph_info)
        return self.split_av_tree(tree)

    # --- Intermediate steps ---
    def assign_av_labels(
        self,
        graph: VGraph,
        av_map: npt.NDArray[np.int_],
        *,
        propagate_labels=True,
        inplace: bool = False,
    ) -> VGraph:
        from ..segment_to_graph.av_tree_parsing import assign_av_label

        return assign_av_label(
            graph,
            av_map=av_map,
            split_av_branch=True,
            av_attr=self.av_attr,
            propagate_labels=propagate_labels,
            inplace=inplace,
        )

    def simplify_av_graph(self, graph: VGraph, od_center: Optional[Point] = None, inplace: bool = False) -> VGraph:
        from ..segment_to_graph.av_tree_parsing import simplify_av_graph

        return simplify_av_graph(
            graph,
            av_attr=self.av_attr,
            inplace=inplace,
        )

    def build_line_digraph(
        self, graph: VGraph, fundus_data: FundusData, inplace: bool = False
    ) -> Tuple[VGraph, npt.NDArray[np.int_], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
        from ..segment_to_graph.av_tree_parsing import build_line_digraph

        return build_line_digraph(graph, fundus_data, av_attr=self.av_attr, inplace=inplace)

    def resolve_digraph_to_vtree(
        self,
        vgraph: VGraph,
        line_list: npt.NDArray[np.int_],
        line_tips: npt.NDArray[np.int_],
        line_probability: npt.NDArray[np.float64],
        line_through_node: npt.NDArray[np.int_],
        branches_dir_p: npt.NDArray[np.float64],
    ) -> VTree:
        from ..segment_to_graph.av_tree_parsing import resolve_digraph_to_vtree

        vtree = resolve_digraph_to_vtree(
            vgraph, line_list, line_tips, line_probability, line_through_node, branches_dir_p
        )
        return vtree

    def split_av_tree(self, tree: VTree) -> Tuple[VTree, VTree]:
        from ..segment_to_graph.av_tree_parsing import split_av_graph_by_subtree

        return split_av_graph_by_subtree(tree, av_attr=self.av_attr)

    # --- Utility methods ---
    def to_vgraph(self, fundus=None, /, *, av=None, od=None, label_av=True, simplify=True):
        fundus = self.prepare_data(fundus, av=av, od=od)
        mask = None if self.mask_optic_disc is None else ~fundus.od
        skel = self.segToGraph.skeletonize(fundus.vessels, mask=mask)
        vessels = fundus.vessels if self.mask_optic_disc is None else fundus.vessels * ~fundus.od
        graph = self.segToGraph.from_skel(skel=skel, vessels=vessels, parse_geometry=True, simplify=True)
        if label_av:
            self.assign_av_labels(graph, fundus.av, inplace=True)
            if simplify:
                self.simplify_av_graph(graph, od_center=fundus.od_center, inplace=True)
        return graph


FUNDUS_SEG_TO_GRAPH = SegToGraph(
    skeletonize_method="lee",
    fix_hollow=True,
    clean_branches_tips=30,
    min_terminal_branch_length=4,
    min_terminal_branch_calibre_ratio=1,
    simplify_graph_arg=GraphSimplifyArg(
        max_spurs_length=0,
        reconnect_endpoints=True,
        junctions_merge_distance=20,
        min_orphan_branches_length=30,
        max_cycles_length=20,
        simplify_topology="node",
    ),
    parse_geometry=True,
)


class NaiveAVSegToTree(AVSegToTreeBase):
    def __init__(
        self,
        segToGraph: Optional[SegToGraph] = None,
    ):
        """

        Parameters
        ----------
        segToGraph: SegToGraph
            The SegToGraph instance to use for the segmentation to graph step.
        """  # noqa: E501
        super(NaiveAVSegToTree, self).__init__()
        self.segToGraph = if_none(segToGraph, FUNDUS_SEG_TO_GRAPH)

    def __call__(
        self,
        fundus: Optional[FundusData] = None,
        /,
        *,
        av: Optional[npt.NDArray[np.int_] | torch.Tensor | str | Path] = None,
        od: Optional[npt.NDArray[np.bool_] | torch.Tensor | str | Path] = None,
    ) -> Tuple[VGraph, VGraph]:
        fundus = self.prepare_data(fundus, av=av, od=od)
        if fundus.od_center is None:
            raise NotImplementedError("Parsing tree of image without optic disc is not implemented.")
        av = fundus.av.copy()
        if self.mask_optic_disc:
            av[fundus.od] = 0
        av_skeleton = self.av_skeletonize(av)

        graphs = [
            self.skel_to_vgraph(av_skeleton, av, simplify=True, populate_geometry=True, **art_vei)
            for art_vei in [{"artery": True, "vein": False}, {"artery": False, "vein": True}]
        ]

        trees = [self.vgraph_to_vtree(g, fundus.od_center) for g in graphs]
        return trees[0], trees[1]

    # --- Intermediate steps ---
    def av_skeletonize(self, av: npt.NDArray | torch.Tensor | FundusData) -> npt.NDArray[np.bool_] | torch.Tensor:
        av = av.av if isinstance(av, FundusData) else av
        art = reduce(ior, (av == label for label in (AVLabel.ART, AVLabel.BOTH)))
        vei = reduce(ior, (av == label for label in (AVLabel.VEI, AVLabel.BOTH)))
        art_skel = self.segToGraph.skeletonize(art)
        vei_skel = self.segToGraph.skeletonize(vei)
        skel = art_skel.astype(int) if isinstance(art_skel, np.ndarray) else art_skel.int()
        skel[art_skel > 0] = AVLabel.ART
        skel[vei_skel > 0] = AVLabel.VEI
        skel[(art_skel > 0) & (vei_skel > 0)] = AVLabel.BOTH
        return skel

    def skel_to_vgraph(
        self,
        skeleton: npt.NDArray | torch.Tensor,
        vessels: npt.NDArray | torch.Tensor | FundusData,
        artery: Optional[bool] = None,
        vein: Optional[bool] = None,
        unknown: Optional[bool] = None,
        simplify: Optional[bool] = True,
        populate_geometry: Optional[bool] = True,
    ) -> VGraph:
        avlabel = AVLabel.select_label(artery=artery, vein=vein, unknown=unknown)

        skel = reduce(ior, (skeleton == label for label in avlabel))
        av = vessels.av if isinstance(vessels, FundusData) else vessels
        av = reduce(ior, (av == label for label in avlabel))

        vgraph = self.segToGraph.skel_to_vgraph(skel, av)
        if if_none(populate_geometry, self.segToGraph.populate_geometry):
            self.segToGraph.populate_geometry(vgraph, av, inplace=True)
        if if_none(simplify, self.segToGraph.simplify_graph):
            self.segToGraph.simplify(vgraph, inplace=True)

        return vgraph

    def vgraph_to_vtree(self, graph: VGraph, od_pos: Point) -> VTree:
        from ..segment_to_graph.av_tree_parsing import naive_vgraph_to_vtree
        from ..segment_to_graph.tree_simplification import clean_vtree

        tree = naive_vgraph_to_vtree(graph, od_pos)
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
        simplify: Optional[bool] = None,
        populate_geometry: Optional[bool] = None,
    ) -> Tuple[VGraph, VGraph]:
        fundus = self.prepare_data(fundus, av=av, od=od)
        av = fundus.av.copy()
        if self.mask_optic_disc:
            av[fundus.od] = 0

        av_skeleton = self.av_skeletonize(av)

        a_graph = self.skel_to_vgraph(
            av_skeleton, av, artery=True, simplify=simplify, populate_geometry=populate_geometry
        )
        v_graph = self.skel_to_vgraph(
            av_skeleton, av, vein=True, simplify=simplify, populate_geometry=populate_geometry
        )

        return a_graph, v_graph
