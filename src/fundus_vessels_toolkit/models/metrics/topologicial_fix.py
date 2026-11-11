from __future__ import annotations

import weakref
from typing import Literal, Self, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel

from fundus_toolkits.utils.geometric import Point, Rect

from ...segment_to_graph.av_map_fixing import rasterize_tree
from ...segment_to_graph.graph_simplification import simplify_passing_nodes
from ...segment_to_graph.tree_topology import transfer_topology
from ...utils.rasterization import rasterize_line
from ...vascular_data_objects import VGraph, VTree
from ...vascular_data_objects.fundus_data import AVLabel
from ..topology.dataset import BranchDigraphSample


class BaseTopoFix(BaseModel):
    img_id: int
    fix_branch_id: int | None = None
    src_branch_id: int | None = None
    src_branch_portion: float

    def __init__(self, src_graph: VGraph | None = None, fix_tree: VTree | None = None, **data):
        super().__init__(**data)
        self._src_graph = weakref.ref(src_graph) if src_graph is not None else None
        self._fix_tree = weakref.ref(fix_tree) if fix_tree is not None else None

    @classmethod
    def from_base(cls, base: BaseTopoFix) -> Self:
        fix = cls(**base.model_dump())
        if base._src_graph is not None:
            fix._src_graph = base._src_graph
        if base._fix_tree is not None:
            fix._fix_tree = base._fix_tree
        return fix

    def bbox(self) -> Rect:
        geodata = self.fix_tree.geometric_data()
        nodes_coord = geodata.node_coord(self.fix_tree.branch_list[self.fix_branch_id])
        curve = np.concatenate([geodata.branch_curve(self.fix_branch_id), nodes_coord], axis=0)
        return Rect.bounding_box(curve)

    def msg(self, short: bool = False) -> str: ...

    def __str__(self) -> str:
        return f"{self.fix_branch_id} | {self.msg()}"

    def branch_highlight(self) -> tuple[list[int], list[int]]:
        src, fix = [], []
        if self.src_branch_id is not None:
            src.append(self.src_branch_id)
        if self.fix_branch_id is not None:
            fix.append(self.fix_branch_id)
        return src, fix

    @property
    def src_graph(self) -> VGraph:
        if self._src_graph is None:
            raise RuntimeError("Source graph has not been set")
        graph = self._src_graph()
        if graph is None:
            raise RuntimeError("Source graph has been garbage collected")
        return graph

    @property
    def fix_tree(self) -> VTree:
        if self._fix_tree is None:
            raise RuntimeError("Fixed tree has not been set")
        tree = self._fix_tree()
        if tree is None:
            raise RuntimeError("Fixed tree has been garbage collected")
        return tree


class Erased(BaseTopoFix):
    src_branch_id: int  # type: ignore
    fix_branch_id: None = None

    @classmethod
    def from_base(cls, base: BaseTopoFix) -> Erased:
        fix = cls(**base.model_dump(exclude={"fix_branch_id"}))
        if base._src_graph is not None:
            fix._src_graph = base._src_graph
        if base._fix_tree is not None:
            fix._fix_tree = base._fix_tree
        return fix

    def msg(self, short: bool = False) -> str:
        if short:
            return f"Branch erased ({self.src_branch_id:d} {self.src_branch_portion:.1%})"
        return f"Branch erased: {self.src_branch_id} ({self.src_branch_portion:.1%})"

    def bbox(self) -> Rect:
        geodata = self.src_graph.geometric_data()
        nodes_coord = geodata.node_coord(self.src_graph.branch_list[self.src_branch_id])
        curve = np.concatenate([geodata.branch_curve(self.src_branch_id), nodes_coord], axis=0)
        return Rect.bounding_box(curve)


class AVDiff(BaseTopoFix):
    src_branch_id: int  # type: ignore
    fix_branch_id: int  # type: ignore
    src_av: AVLabel
    fix_av: AVLabel

    @classmethod
    def from_base(cls, base: BaseTopoFix, src_av: AVLabel, fix_av: AVLabel) -> AVDiff:
        fix = cls(**base.model_dump(), src_av=src_av, fix_av=fix_av)
        if base._src_graph is not None:
            fix._src_graph = base._src_graph
        if base._fix_tree is not None:
            fix._fix_tree = base._fix_tree
        return fix

    def msg(self, short: bool = False) -> str:
        if short:
            return f"{str(self.src_av)} → {str(self.fix_av)} ({self.fix_branch_id:d} {self.src_branch_portion:.1%})"
        return f"Changed A/V: {str(self.src_av)} → {str(self.fix_av)} ({self.src_branch_portion:.1%})"


class AVSpecified(BaseTopoFix):
    src_branch_id: int  # type: ignore
    fix_branch_id: int  # type: ignore
    fix_av: AVLabel

    @classmethod
    def from_base(cls, base: BaseTopoFix, fix_av: AVLabel) -> AVSpecified:
        fix = cls(**base.model_dump(), fix_av=fix_av)
        if base._src_graph is not None:
            fix._src_graph = base._src_graph
        if base._fix_tree is not None:
            fix._fix_tree = base._fix_tree
        return fix

    def msg(self, short: bool = False) -> str:
        if short:
            return f"UNK → {str(self.fix_av)} ({self.fix_branch_id:d} {self.src_branch_portion:.1%})"
        return f"Specify A/V: UNK → {str(self.fix_av)} ({self.src_branch_portion:.1%})"


class Disconnected(BaseTopoFix):
    src_branch_id: int  # type: ignore
    fix_branch_id: int  # type: ignore

    def msg(self, short: bool = False) -> str:
        if short:
            return f"Branch disconnected ({self.fix_branch_id:d})"
        return f"Branch disconnected: {self.src_branch_id}"

    def bbox(self) -> Rect:
        geodata = self.fix_tree.geometric_data()
        n = Point.from_array(geodata.node_coord(self.fix_tree.tree_branch_list()[self.fix_branch_id, 0]))
        return Rect.from_center(n, (10, 10))


type ReconnectedReason = Literal["crossing", "segmention-fn", "av_change", "out-of-view"]


class Reconnected(BaseTopoFix):
    src_branch_id: int  # type: ignore
    fix_branch_id: int  # type: ignore
    fix_parent: int
    reason: ReconnectedReason | None = None

    @classmethod
    def from_base(cls, base: BaseTopoFix, fix_parent: int, reason: ReconnectedReason | None = None) -> Reconnected:
        fix = cls(**base.model_dump(), fix_parent=fix_parent, reason=reason)
        if base._src_graph is not None:
            fix._src_graph = base._src_graph
        if base._fix_tree is not None:
            fix._fix_tree = base._fix_tree
        return fix

    def msg(self, short: bool = False) -> str:
        if short:
            match self.reason:
                case "crossing":
                    return f"Crossing reconnection ({self.fix_branch_id:d})"
                case "out-of-view":
                    return f"Out of view connection ({self.fix_branch_id:d})"
                case "segmention-fn":
                    return f"Segmentation reconnection ({self.fix_branch_id:d})"
                case "av_change":
                    return f"Reconnection after A/V change ({self.fix_branch_id:d})"
                case _:
                    return f"Branch reconnection ({self.fix_branch_id:d})"
        return f"Branch reconnected: {self.src_branch_id} (reason: {self.reason})"

    def bbox(self) -> Rect:
        geodata = self.fix_tree.geometric_data()
        branch_list = self.fix_tree.tree_branch_list()
        n1 = geodata.node_coord(branch_list[self.fix_branch_id, 0])
        n0 = geodata.node_coord(branch_list[self.fix_parent, 1])
        return Rect.bounding_box([n0, n1])

    def branch_highlight(self) -> tuple[list[int], list[int]]:
        return [], [self.fix_tree.branch_tree[self.fix_branch_id]]


class TopoFixes:
    def __init__(self, diffs: list[BaseTopoFix], src_graph: VGraph, fix_tree: VTree, sample: BranchDigraphSample):
        self.diffs = diffs
        self.src_graph = src_graph
        self.fix_tree = fix_tree
        self.sample = sample

    @classmethod
    def from_sample(cls, sample: BranchDigraphSample) -> TopoFixes:
        graph = sample.graphes["gt"].copy()

        simplify_passing_nodes(graph, min_angle=110, inplace=True)
        tree = transfer_topology(sample.target_topologies, graph, allow_splits=5)
        tree.branch_attr["gt_av"] = tree.branch_attr["ini_branch_id"].map(graph.branch_attr["av"])
        graph_branch_length = pd.Series([b.arc_length(fast=True) for b in graph.branches()])
        tree_branch_length = np.array([b.arc_length(fast=True) for b in tree.branches()])
        tree.branch_attr["b_part"] = tree_branch_length / tree.branch_attr["ini_branch_id"].map(graph_branch_length)

        B = tree.branch_attr.index[tree.branch_attr["ini_branch_id"].notna()].max()

        diffs: list[BaseTopoFix] = []

        graph_terminal_nodes = graph.endpoint_nodes()
        tree_nodes_yx = tree.geometric_data().node_coord()
        tree_branch_list = tree.tree_branch_list()
        for b in tree.branches():
            if b.id > B:
                continue
            src_branch_id = int(b.attr["ini_branch_id"].item())
            if src_branch_id is None:
                raise ValueError(f"Branch {b.id} in tree has no corresponding branch in graph")
            base = BaseTopoFix(
                img_id=0,
                src_graph=graph,
                fix_tree=tree,
                fix_branch_id=b.id,
                src_branch_id=src_branch_id,
                src_branch_portion=b.attr["b_part"].item(),
            )
            src_av: AVLabel = AVLabel(b.attr["gt_av"].item())
            fix_av: AVLabel = AVLabel(b.attr["av"].item())
            if src_av != fix_av and fix_av not in [AVLabel.ART, AVLabel.VEI]:
                diffs.append(Erased.from_base(base))
                continue

            fix_tip0 = 0 if tree.branch_dirs(b.id) else 1
            fix_parent = int(tree.branch_tree[b.id])
            while fix_parent > B:
                fix_parent = int(tree.branch_tree[fix_parent])

            src_tail_node = graph.branch_list[src_branch_id][fix_tip0]
            is_root_src = np.isin(src_tail_node, graph_terminal_nodes).item()

            if fix_parent == -1:
                if not is_root_src:
                    # If at least one of the adjacent branches is not a source branch, count the diff as a disconnection.
                    # Otherwise it's simply a root placement and we don't need to count it as a diff.
                    for adj_src_branch in graph.node(src_tail_node).adjacent_branch_ids:
                        adj_dst_branches = tree.branch_attr.index[tree.branch_attr["ini_branch_id"] == adj_src_branch]
                        if len(adj_dst_branches) == 0 and not np.any(tree.branch_tree[adj_dst_branches] == -1):
                            diffs.append(Disconnected.from_base(base))
                            break
            else:
                fix_tail_node = tree_branch_list[b.id, 0]
                fix_parent_head_node = tree_branch_list[fix_parent, 1]
                n0 = tree_nodes_yx[fix_tail_node]
                n1 = tree_nodes_yx[fix_parent_head_node]
                # If the tail node is not adjacent to the parent head node, count the diff as a reconnection.
                if np.abs(tree_nodes_yx[fix_tail_node] - tree_nodes_yx[fix_parent_head_node]).max() > 1:
                    src_parent_av = AVLabel(tree.branch(fix_parent).attr["gt_av"].item())
                    reason = None
                    fundus_center, fundus_r = sample.fundus.roi_specs.center, sample.fundus.roi_specs.radius
                    av_map = sample._av_maps["gt"] if sample._av_maps is not None else None
                    fundus_r *= 0.95
                    if (
                        np.linalg.norm(n0 - n1) > 20
                        and fundus_center.distance(n0) > fundus_r
                        and fundus_center.distance(n1) > fundus_r
                    ):
                        reason = "out-of-view"
                    elif src_av != src_parent_av and src_av in [AVLabel.ART, AVLabel.VEI]:
                        reason = "av_change"
                    elif av_map is not None and np.mean(av_map[*rasterize_line(tuple(n0), tuple(n1)).T] != 0) > 0.75:
                        reason = "crossing"
                    else:
                        reason = "segmention-fn"
                    diffs.append(Reconnected.from_base(base, fix_parent=fix_parent, reason=reason))

            if src_av != fix_av:
                if src_av in [AVLabel.ART, AVLabel.VEI]:
                    d = AVDiff.from_base(base, src_av=src_av, fix_av=fix_av)
                    diffs.append(d)
                else:
                    diffs.append(AVSpecified.from_base(base, fix_av=fix_av))

        return cls(diffs=diffs, src_graph=graph, fix_tree=tree, sample=sample)

    def cluster_diffs(self) -> list[list[BaseTopoFix]]:
        """
        Cluster diffs based on the fixed tree topology. Diffs that concern the same branch or connected branches are grouped together.
        """  # noqa: E501
        B = self.fix_tree.branch_attr.index[self.fix_tree.branch_attr["ini_branch_id"].notna()].max()
        clusters = {}
        removed_diffs = list()
        for diff in self.diffs:
            if diff.fix_branch_id is not None:
                clusters.setdefault(diff.fix_branch_id, []).append(diff)
            else:
                removed_diffs.extend([diff])

        reverse_depth_order = reversed(list(self.fix_tree.walk_branch_ids(traversal="dfs")))
        for b in reverse_depth_order:
            if b not in clusters:
                continue
            parent = self.fix_tree.branch_tree[b]
            while parent >= B:
                parent = self.fix_tree.branch_tree[parent]
            if parent in clusters:
                clusters[parent].extend(clusters[b])
                del clusters[b]

        return [diff for diff in clusters.values()] + [[_] for _ in removed_diffs]

    def inspect(self):
        from ipywidgets import Button, HBox, Label, Layout, SelectMultiple, VBox
        from jppype import Mosaic

        from ...utils.jppype import AV_COLORS, draw_graph, draw_tree

        m = Mosaic(3, background=self.sample.fundus.image)
        m[0].add_label(self.sample._av_maps["gt"], "av_seg", colormap=AV_COLORS, opacity=0.3)
        graph_layer = draw_graph(self.src_graph, view=m[0], edge_labels=True)
        new_av_seg = (rasterize_tree(self.sample.target_topologies[0].tree) > 0).astype(np.uint8)
        new_av_seg += (rasterize_tree(self.sample.target_topologies[1].tree) > 0) * np.uint8(2)
        m[2].add_label(new_av_seg, "av_seg", colormap=AV_COLORS, opacity=0.3)
        tree_layer = draw_tree(
            self.fix_tree, view=m[2], interactive=True, branch_color="av", edge_labels=True, bspline_dir=True
        )
        tree_layer.set_options(
            {
                "edges_labels": [
                    str(b.id) if not np.isnan(b.attr["ini_branch_id"]) else "" for b in self.fix_tree.branches()
                ]
            }
        )
        label = Label(value="")
        view = VBox(
            children=[label, m.draw_mosaic()],
            layout=Layout(width="100%", row_gap="10px"),
        )

        diffs_cluster = self.cluster_diffs()
        sortedDiffs = [d for cluster in diffs_cluster for d in cluster]
        cluster_to_diffID = {}
        diffID_to_cluster = {}
        i = 0
        for c, cluster in enumerate(diffs_cluster):
            N = len(cluster)
            diffID_to_cluster.update({diffID: c for diffID in range(i, i + N)})
            cluster_to_diffID[c] = list(range(i, i + N))
            i += N

        errors = [f"{i + 1}- {d.msg(short=True)}" for i, d in enumerate(sortedDiffs)]
        btn_layout = Layout(width="35px")
        bPrev = Button(description="←", layout=btn_layout)
        bNext = Button(description="→", layout=btn_layout)
        errorIDLabel = Label(value=f"- / {len(sortedDiffs)}", layout=Layout(flex="1", justify_content="center"))

        navigatorHeader = HBox(
            children=[
                bPrev,
                errorIDLabel,
                bNext,
            ],
            layout=Layout(
                width="100%",
                justify_content="space-around",
            ),
        )

        errors_list = SelectMultiple(
            options=errors,
            value=[],
            disabled=False,
            layout=Layout(width="95%", height="600px", flex="1"),
        )

        navigator = VBox(
            children=[navigatorHeader, errors_list],
            layout=Layout(width="270px", height="650px", justify_content="space-between"),
        )

        def select_errors(error_ids: list[int | str]):
            error_ids_ = [i if isinstance(i, int) else errors.index(i) for i in error_ids]
            selected_errors = [sortedDiffs[i] for i in error_ids_]
            if not selected_errors:
                m[0].goto((0, 0), scale=0)
            else:
                bbox = selected_errors[0].bbox()
                for d in selected_errors[1:]:
                    bbox |= d.bbox()
                m[0].goto(bbox.center.xy, scale=3)
            errors_list.value = [errors[i] for i in error_ids_]
            if len(error_ids) == 0:
                errorIDLabel.value = f"- / {len(sortedDiffs)}"
            elif len(error_ids) == 1:
                errorIDLabel.value = f"{error_ids_[0] + 1} / {len(sortedDiffs)}"
            else:
                errorIDLabel.value = f"{error_ids_[0] + 1}-{error_ids_[-1] + 1} / {len(sortedDiffs)}"

            highlighted_src, highlighted_fix = set(), set()
            for d in selected_errors:
                src, fix = d.branch_highlight()
                highlighted_src.update(src)
                highlighted_fix.update(fix)

            src_branch_cmap = {b: AV_COLORS[AVLabel.BKG] for b in range(self.src_graph.branch_count)}
            for b in self.src_graph.branches():
                av = b.attr["av"]
                if b.id in highlighted_src:
                    match av:
                        case AVLabel.ART:
                            src_branch_cmap[b.id] = "#FF0000"
                        case AVLabel.VEI:
                            src_branch_cmap[b.id] = "#0000FF"
                        case AVLabel.BOTH:
                            src_branch_cmap[b.id] = "#FF00FF"
                        case _:
                            src_branch_cmap[b.id] = "#FFFFFF"
                else:
                    src_branch_cmap[b.id] = AV_COLORS[av]
            graph_layer.edges_cmap = src_branch_cmap

            fix_branch_cmap = {b: AV_COLORS[AVLabel.BKG] for b in range(self.fix_tree.branch_count)}
            for b in self.fix_tree.branches():
                av = b.attr["av"]
                if b.id in highlighted_fix:
                    match av:
                        case AVLabel.ART:
                            fix_branch_cmap[b.id] = "#FF0000"
                        case AVLabel.VEI:
                            fix_branch_cmap[b.id] = "#0000FF"
                        case AVLabel.BOTH:
                            fix_branch_cmap[b.id] = "#FF00FF"
                        case _:
                            fix_branch_cmap[b.id] = "#FFFFFF"
                else:
                    fix_branch_cmap[b.id] = AV_COLORS[av]
            tree_layer.edges_cmap = fix_branch_cmap

        errors_list.observe(lambda change: select_errors(change["new"]), names="value")

        def read_selection() -> list[int]:
            selected_errors = errors_list.value
            error_ids = [errors.index(e) for e in selected_errors]
            return error_ids

        def next_error():
            selected_errors = read_selection()
            if len(selected_errors) == 0:
                select_errors(cluster_to_diffID[diffID_to_cluster[0]])
            else:
                for i in range(selected_errors[0] + 1, len(sortedDiffs)):
                    if i not in selected_errors:
                        select_errors(cluster_to_diffID[diffID_to_cluster[i]])
                        return
                else:
                    select_errors([])

        bNext.on_click(lambda btn: next_error())

        def prev_error():
            selected_errors = read_selection()
            if len(selected_errors) == 0:
                select_errors(cluster_to_diffID[diffID_to_cluster[len(sortedDiffs) - 1]])
            else:
                for i in range(selected_errors[0] - 1, -1, -1):
                    if i not in selected_errors:
                        select_errors(cluster_to_diffID[diffID_to_cluster[i]])
                        return
                else:
                    select_errors([])

        bPrev.on_click(lambda btn: prev_error())

        mainLayout = HBox(children=[navigator, view], layout=Layout(width="100%"))

        return mainLayout


class TopoFixesSummary:
    def __init__(self, diffs: list[BaseTopoFix]):
        self.diffs = diffs

    def __len__(self):
        return len(self.diffs)

    def __getitem__(self, idx: int) -> BaseTopoFix:
        return self.diffs[idx]

    def extend(self, diff: BaseTopoFix | Sequence[BaseTopoFix]):
        if isinstance(diff, Sequence):
            self.diffs.extend(diff)
        else:
            self.diffs.append(diff)

    def __str__(self):
        s = ""
        for i, diffs in self.by_image().items():
            s += f"Image {i} ({len(diffs)} diffs):\n"
            for d in diffs:
                s += f"  {d}\n"
        return s

    def __repr__(self):
        return f"DatasetDiffs({len(self)} diffs)"

    def by_image(self) -> dict[int, list[BaseTopoFix]]:
        by_image = {}
        for d in self.diffs:
            by_image.setdefault(d.img_id, []).append(d)
        return by_image

    def images(self) -> set[int]:
        return set(self.by_image().keys())

    def summary(
        self,
        per_image: bool = False,
        *,
        ERASE="erased",
        AV_CHANGE="av_change",
        AV_SPECIFIED="av_specified",
        DISCONNECTION="disconnected",
        OUT_RECONNECTION="reconnection_out-of-view",
        AV_RECONNECTION="reconnection_av_change",
        CROSSING_RECONNECTION="reconnection_crossing",
        SEGMENTATION_RECONNECTION="reconnection_segmentation-fn",
    ) -> dict[str, int] | dict[str, float]:
        summary = {
            _: 0
            for _ in [
                ERASE,
                AV_CHANGE,
                AV_SPECIFIED,
                DISCONNECTION,
                OUT_RECONNECTION,
                AV_RECONNECTION,
                CROSSING_RECONNECTION,
                SEGMENTATION_RECONNECTION,
            ]
        }
        for d in self.diffs:
            if isinstance(d, Erased):
                summary[ERASE] += 1
            elif isinstance(d, AVDiff):
                summary[AV_CHANGE] += 1
            elif isinstance(d, AVSpecified):
                summary[AV_SPECIFIED] += 1
            elif isinstance(d, Disconnected):
                summary[DISCONNECTION] += 1
            elif isinstance(d, Reconnected):
                match d.reason:
                    case "crossing":
                        summary[CROSSING_RECONNECTION] += 1
                    case "segmention-fn":
                        summary[SEGMENTATION_RECONNECTION] += 1
                    case "av_change":
                        summary[AV_RECONNECTION] += 1
                    case "out-of-view":
                        summary[OUT_RECONNECTION] += 1

        if per_image:
            n_image = len(self.by_image())
            summary = {k: v / n_image for k, v in summary.items()}

        return summary
