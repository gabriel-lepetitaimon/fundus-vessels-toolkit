from functools import partial
from pathlib import Path
from typing import Literal, overload

import numpy as np
from attr import dataclass
from fundus_data_toolkit.functional import open_image
from ipywidgets import Button, GridBox, Label, Layout
from jppype import Mosaic, vscode_theme

from fundus_odmac_toolkit.models.segmentation import segment
from fundus_toolkits import FundusData
from fundus_vessels_toolkit.pipelines.avseg_to_tree import NaiveAVSegToTree
from fundus_vessels_toolkit.segment_to_graph.av_map_fixing import TopologicalLabel, rasterize_tree_topology
from fundus_vessels_toolkit.segment_to_graph.av_tree_parsing import naive_infer_roots
from fundus_vessels_toolkit.segment_to_graph.graph_simplification import simplify_passing_nodes
from fundus_vessels_toolkit.segment_to_graph.tree_simplification import disconnect_crossing
from fundus_vessels_toolkit.utils.jppype import draw_tree
from fundus_vessels_toolkit.vascular_data_objects.vgraph import NodeIndices
from fundus_vessels_toolkit.vascular_data_objects.vtree import VTree, VTreeNode


@dataclass
class BranchSelection:
    artery: None | bool = None
    id: int = 0
    tip: int = 0  # 0 or 1

    def reset(self):
        self.artery = None
        self.id = 0
        self.tip = 0

    def select(self, artery: bool, id: int, tip: int):
        self.artery = artery
        self.id = id
        self.tip = tip

    def is_second_selection(self, artery: bool) -> bool:
        return self.artery is not None and self.artery == artery


@dataclass
class AnnotationContext:
    force_roots: list[VTreeNode] = []

    def clear(self):
        self.force_roots.clear()


class ReviewTool:
    def __init__(
        self,
        raw_path: Path,
        av_path: Path,
        save_path: Path,
        index: int = 0,
        *,
        raw_ext="png",
        av_ext="png",
        height=800,
        av2tree=None,
    ):
        self.raw_path = Path(raw_path)
        self.av_path = Path(av_path)
        self.save_path = Path(save_path)
        self.raw_ext = raw_ext
        self.av_ext = av_ext

        img_names = {_.stem for _ in self.raw_path.glob(f"*.{raw_ext}")} & {
            _.stem for _ in self.av_path.glob(f"*.{av_ext}")
        }
        assert len(img_names) > 0, "No images found in the specified directories."
        self.img_names = sorted(list(img_names))
        self.current_index = 0

        self.av2tree = av2tree or NaiveAVSegToTree()

        self.mosaic = Mosaic((2, 3), rows_titles=["Art", "Vei"], cell_height=height // 2)
        self.mosaic[0, 1].on_click(partial(self.handle_click, artery=True))
        self.mosaic[1, 1].on_click(partial(self.handle_click, artery=False))
        self.label = Label(value="")
        self.undo_btn = Button(description="Undo", disabled=True)
        self.undo_btn.on_click(lambda btn: self.undo())

        # Annotation State
        self.debug_info = {}
        self._fundus: FundusData | None = None
        self.img_name: str = ""

        self._trees: None | tuple[VTree, VTree] = None
        self._previous_trees: None | tuple[VTree, VTree] = None
        self.annotation_ctx: tuple[AnnotationContext, AnnotationContext] = (AnnotationContext(), AnnotationContext())
        self.prev_selected_branch: BranchSelection = BranchSelection()

        # Load first image
        self.load(index)

    def widget(self) -> GridBox:
        btn_layout = Layout(width="80px")
        bPrev = Button(description="Previous", layout=btn_layout)
        bPrev.on_click(lambda btn: self.previous_image())
        bNext = Button(description="Next", layout=btn_layout)
        bNext.on_click(lambda btn: self.next_image())
        bSave = Button(description="Save", layout=btn_layout)
        bSave.on_click(lambda btn: self.save_trees())
        bReset = Button(description="Reset", layout=btn_layout)
        bReset.on_click(lambda btn: self.reset_annotations())
        bCompleteReset = Button(description="Complete Reset", layout=btn_layout)
        bCompleteReset.on_click(lambda btn: self.complete_reset())

        buttons = GridBox(
            children=[bPrev, bNext, self.undo_btn, bSave, bReset, bCompleteReset, self.label],
            layout=Layout(
                width="100%",
                grid_template_columns="repeat(6, 150px) auto",
                grid_template_rows="auto",
                justify_content="space-around",
            ),
        )

        view = GridBox(
            children=[buttons, self.mosaic.draw_mosaic()],
            layout=Layout(
                width="100%",
                grid_template_columns="100%",
                grid_template_rows="auto auto",
                row_gap="10px",
            ),
        )

        return view

    @property
    def trees(self) -> tuple[VTree, VTree]:
        if self._trees is None:
            raise ValueError("No trees loaded.")
        return self._trees

    @property
    def fundus(self) -> FundusData:
        if self._fundus is None:
            raise ValueError("No fundus loaded.")
        return self._fundus

    def load(self, index: int):
        img_name = self.img_names[index]
        self.img_name = img_name
        self.current_index = index
        self.label.value = f"{img_name} ({index + 1}/{len(self.img_names)})"

        self.reset_annotation_ctx()

        fundus = FundusData(
            image=self.raw_path / (img_name + "." + self.raw_ext),
            av=self.av_path / (img_name + "." + self.av_ext),
        )
        # segment_od_mac(fundus)
        od_mac = segment(open_image(self.raw_path / (img_name + "." + self.raw_ext))).numpy(force=True).argmax(axis=0)
        fundus = fundus.update(od=od_mac == 1, macula=od_mac == 2)
        self._fundus = fundus

        # Draw fundus
        self.mosaic[0, 0].add_image(fundus.image, name="fundus")
        fundus.draw(view=self.mosaic[1, 0])
        self.mosaic[0, 1].add_image(fundus.image, name="fundus")
        self.mosaic[1, 1].add_image(fundus.image, name="fundus")

        # Load or compute trees
        self.load_saved_trees(draw=False)
        self.draw_trees()

    def load_saved_trees(self, draw=True) -> tuple[VTree, VTree]:
        art_file = self.save_path / f"{self.img_name}_art.npz"
        vei_file = self.save_path / f"{self.img_name}_vei.npz"
        if art_file.exists() and vei_file.exists():
            self._trees = VTree.load(art_file), VTree.load(vei_file)
        else:
            self._trees = self.load_trees_from_av(draw=False)
        if draw:
            self.draw_trees()
        return self.trees

    def load_trees_from_av(self, draw=True) -> tuple[VTree, VTree]:
        self._trees = self.av2tree(self.fundus)
        if draw:
            self.draw_trees()
        return self.trees

    def save_trees(self):
        art_file = self.save_path / f"{self.img_name}_art"
        vei_file = self.save_path / f"{self.img_name}_vei"
        self.trees[0].save(art_file)
        self.trees[1].save(vei_file)

    def draw_trees(self, which: Literal["artery", "vein", "both"] = "both"):
        if which in ("artery", "both"):
            draw_tree(self.trees[0], view=self.mosaic[0, 1], artery=True, node_labels=True)
        if which in ("vein", "both"):
            draw_tree(self.trees[1], view=self.mosaic[1, 1], artery=False, node_labels=True)

        for i, tree in enumerate(self.trees):
            if (which == "vein" and i == 0) or (which == "artery" and i == 1):
                continue

            label_map, topo_map = rasterize_tree_topology(tree, bridge_gap_smaller_than=50)
            subtree_map = TopologicalLabel.decode_subtree(label_map)
            N_subtree = int(subtree_map.max()) + 1
            color_map = np.zeros(self.fundus.shape + (3,), dtype=np.float32)
            alpha = np.zeros_like(topo_map)

            for s in range(1, N_subtree):
                mask = subtree_map == s
                color_map[mask] = TopologicalLabel.subtree_color(s, format="rgb") / 255.0
                subtree_topo = topo_map[mask]
                alpha[mask] = 1 - 0.8 * (subtree_topo / subtree_topo.max())

            alpha = alpha[:, :, None]

            img = self.fundus.image.transpose(1, 2, 0) * 0.5
            img = (1 - alpha) * img + alpha * color_map
            self.mosaic[i, 2].add_image(img, name="topo_map")

    ##########################################################################
    # === CLICK HANDLERS ===
    ##########################################################################
    def handle_click(self, event, artery: bool):
        modified_tree: Literal["artery", "vein", "both"] = "artery" if artery else "vein"

        art = 0 if artery else 1
        tree = self.trees[art]
        ctx = self.annotation_ctx[art]

        yx = (event["y"], event["x"])

        self.debug_info.clear()
        self.debug_info["event"] = event
        self.debug_info["artery"] = artery
        self.debug_info["yx"] = yx
        self.debug_info["ctx"] = ctx
        self.debug_info["previous_trees"] = self._trees
        self.debug_info["annotation_ctx"] = self.annotation_ctx
        self.debug_info["prev_selected_branch"] = self.prev_selected_branch

        previous_trees = self._trees

        if event["button"] == 0 and event["modifiers"] == ["alt"]:  # Left click + Alt
            tree = self.connect_branches(tree, yx, ctx, artery)
        else:
            self.prev_selected_branch.reset()

        if event["button"] == 0 and event["modifiers"] == ["shift"]:  # Left click + Shift
            tree = self.add_node(tree, yx, ctx)
        elif event["button"] == 2 and event["modifiers"] == []:  # Right click
            tree = self.simplify_nodes(tree, yx)
        elif event["button"] == 2 and event["modifiers"] == ["shift"]:  # Right click + Shift
            tree = self.disconnect_crossing(tree, yx, ctx)
        elif event["button"] == 2 and (
            "alt" in event["modifiers"] or "ctrl" in event["modifiers"]
        ):  # Right click + Alt/Ctrl
            subtree = "ctrl" in event["modifiers"]
            tree = self.delete_branch(tree, yx, ctx, subtree=subtree)
        elif event["button"] == 1 and event["modifiers"] == []:  # Middle click
            tree = self.toggle_force_root(tree, yx, ctx)
        elif event["button"] == 1 and (
            "alt" in event["modifiers"] or "ctrl" in event["modifiers"]
        ):  # Middle click + Alt/Ctrl
            subtree = "ctrl" in event["modifiers"]
            tree = self.swap_av(artery, yx, ctx, subtree=subtree)
            modified_tree = "both"

        if previous_trees is None or tree is not previous_trees[art]:
            self._trees = (tree, self.trees[1]) if artery else (self.trees[0], tree)
            self._previous_trees = previous_trees
            self.undo_btn.disabled = False
            self.draw_trees(which=modified_tree)

    def undo(self):
        if self._previous_trees is not None:
            self._trees, self._previous_trees = self._previous_trees, None
            self.undo_btn.disabled = True
            self.draw_trees(which="both")

    def add_node(self, tree, yx, ctx) -> VTree:
        branch_id, dist, curve_id = self._closest_branch(tree, yx)
        if dist > 20:
            return tree

        tree = tree.split_branch(branch_id, curve_id, split_coord=yx)
        self.infer_roots(tree, ctx, inplace=True)
        return tree

    def connect_branches(self, tree, yx, ctx, artery) -> VTree:
        branch_id, dist, curve_pos = self._closest_branch(tree, yx, relative_pos=True)
        if dist > 20:
            return tree

        tip = 0 if curve_pos < 0.5 else 1

        if not self.prev_selected_branch.is_second_selection(artery):
            # Select first branch
            self.prev_selected_branch.select(artery, branch_id, tip)
            return tree

        # Connect branches
        nodes = [
            tree.branch_list[self.prev_selected_branch.id][self.prev_selected_branch.tip],
            tree.branch_list[branch_id][tip],
        ]
        if nodes[0] == nodes[1]:
            return tree

        tree = tree.add_branch(nodes)

        self.infer_roots(tree, ctx, inplace=True, simplify_nodes=nodes)
        self.prev_selected_branch.reset()
        return tree

    def simplify_nodes(self, tree, yx) -> VTree:
        node, dist = self._closest_node(tree, yx)
        if dist > 20 or node.id in tree.root_nodes_ids():
            return tree

        tree = simplify_passing_nodes(tree, only_fusable=node.id)
        return tree

    def disconnect_crossing(self, tree, yx, ctx) -> VTree:
        node, dist = self._closest_node(tree, yx)
        if dist > 20 or node.id in tree.root_nodes_ids():
            return tree

        tree, node = disconnect_crossing(tree, node.id, return_new_nodes=True)
        self.infer_roots(tree, ctx, inplace=True, simplify_nodes=node)
        return tree

    def delete_branch(self, tree, yx, ctx, *, subtree=False) -> VTree:
        branch_id, dist, _ = self._closest_branch(tree, yx)
        if dist > 20:
            return tree

        if subtree:
            subtrees = tree.branch_ids_by_subtree()
            for subtree in subtrees:
                if branch_id in subtree:
                    branch_id = subtree
                    break

        tree = tree.delete_branch(branch_id)
        self.infer_roots(tree, ctx, inplace=True)
        return tree

    def swap_av(self, artery, yx, ctx, *, subtree=False) -> VTree:
        art = 0 if artery else 1
        tree = self.trees[art]
        other_tree = self.trees[1 - art]

        branch_id, dist, _ = self._closest_branch(tree, yx)
        if dist > 20:
            return tree

        if subtree:
            subtrees = tree.branch_ids_by_subtree()
            for subtree in subtrees:
                if branch_id in subtree:
                    branch_id = subtree
                    break

        subtree = tree.subtree(branch_id)

        self._previous_trees = self._trees

        # Add to other tree
        other_tree = other_tree.append_graph(subtree)
        self.infer_roots(other_tree, self.annotation_ctx[1 - art], inplace=True)

        # Remove from current tree
        tree = tree.delete_branch(branch_id)
        self.infer_roots(tree, ctx, inplace=True)

        self._trees = (tree, other_tree) if artery else (other_tree, tree)

        return tree

    def toggle_force_root(self, tree: VTree, yx, ctx) -> VTree:
        node, dist = self._closest_node(tree, yx)
        if dist > 20:
            return tree
        if node in ctx.force_roots:
            ctx.force_roots.remove(node)
        else:
            ctx.force_roots.append(node)
        tree = self.infer_roots(tree, ctx)
        return tree

    def infer_roots(
        self, tree: VTree, ctx: AnnotationContext, *, inplace=False, simplify_nodes: None | NodeIndices = None
    ) -> VTree:
        old_force_roots = [n for n in ctx.force_roots if n.is_valid()]
        ctx.force_roots = []
        while root := old_force_roots.pop() if old_force_roots else None:
            ctx.force_roots.insert(0, root)
            old_force_roots = [n for n in old_force_roots if n.id != root.id]

        od_center = self.fundus.od_center
        if od_center is None:
            raise ValueError("Optic disc center is not defined in the fundus data.")
        tree = naive_infer_roots(
            tree,
            root_pos=od_center,
            force_roots=[n.id for n in ctx.force_roots if n.is_valid()],
            inplace=inplace,
        )
        if simplify_nodes is not None:
            tree = simplify_passing_nodes(tree, inplace=inplace, only_fusable=simplify_nodes)
        return tree

    def _closest_node(self, tree: VTree, yx) -> tuple[VTreeNode, float]:
        gdata = tree.geometric_data()
        node_id, dist = gdata.closest_nodes(yx, return_distance=True)
        node = tree.node(int(node_id))
        return node, float(dist)

    @overload
    def _closest_branch(self, tree: VTree, yx, *, relative_pos: Literal[False] = False) -> tuple[int, float, float]: ...
    @overload
    def _closest_branch(self, tree: VTree, yx, *, relative_pos: Literal[True]) -> tuple[int, int, float]: ...
    def _closest_branch(
        self, tree: VTree, yx, *, relative_pos=False
    ) -> tuple[int, float, float] | tuple[int, int, float]:
        gdata = tree.geometric_data()
        (branch_id, curve_id), dist = gdata.closest_branches(yx, interpolate=True, return_distance=True)
        if relative_pos:
            curve_id = curve_id / gdata.branch_arc_length(branch_id, fast_approximation=True)
        return branch_id, float(dist), curve_id

    def debug_retrigger_event(self):
        if "event" not in self.debug_info:
            return
        self._trees = self.debug_info.get("previous_trees", self._trees)
        self.annotation_ctx = self.debug_info.get("annotation_ctx", self.annotation_ctx)
        self.prev_selected_branch = self.debug_info.get("prev_selected_branch", self.prev_selected_branch)
        event = self.debug_info["event"]
        artery = self.debug_info["artery"]
        self.handle_click(event, artery)

    ##########################################################################
    # === BUTTON INTERACTIONS ===
    ##########################################################################
    def next_image(self):
        if self.current_index + 1 < len(self.img_names):
            self.load(self.current_index + 1)

    def previous_image(self):
        if self.current_index - 1 >= 0:
            self.load(self.current_index - 1)

    def reset_annotations(self):
        self.load_saved_trees(draw=True)
        for tree, ctx in zip(self.trees, self.annotation_ctx, strict=True):
            ctx.force_roots = list(tree.nodes(tree.root_nodes_ids()))

    def complete_reset(self):
        self.load_trees_from_av(draw=True)
        self.reset_annotation_ctx()

    def reset_annotation_ctx(self):
        for ctx in self.annotation_ctx:
            ctx.clear()
        self._previous_trees = None
        self.undo_btn.disabled = True
        self.prev_selected_branch.reset()
