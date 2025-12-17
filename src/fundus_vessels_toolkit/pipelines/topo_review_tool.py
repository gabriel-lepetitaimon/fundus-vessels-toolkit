from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from attr import dataclass
from cycler import V
from fundus_data_toolkit.functional import open_image
from ipywidgets import Button, GridBox, HBox, Label, Layout
from jppype import Mosaic, vscode_theme

from fundus_odmac_toolkit import segment_od_mac
from fundus_toolkits import FundusData
from fundus_vessels_toolkit.models import segment_av
from fundus_vessels_toolkit.pipelines.avseg_to_tree import GNNAVSegToTree, NaiveAVSegToTree
from fundus_vessels_toolkit.segment_to_graph.av_map_fixing import TopologicalLabel, rasterize_tree_topology
from fundus_vessels_toolkit.segment_to_graph.av_tree_parsing import naive_infer_roots
from fundus_vessels_toolkit.segment_to_graph.graph_simplification import simplify_passing_nodes
from fundus_vessels_toolkit.segment_to_graph.tree_simplification import disconnect_crossing
from fundus_vessels_toolkit.utils.jppype import draw_tree
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
        return self.artery is not None and self.artery != artery


@dataclass
class AnnotationContext:
    force_roots: set[VTreeNode] = set()


class ReviewTool:
    def __init__(self, raw_path: Path, av_path: Path, save_path: Path, raw_ext="png", av_ext="png", av2tree=None):
        self.raw_path = raw_path
        self.av_path = av_path
        self.save_path = save_path
        self.raw_ext = raw_ext
        self.av_ext = av_ext

        img_names = set(raw_path.glob(f"*.{raw_ext}")) & set(av_path.glob(f"*.{av_ext}"))
        self.img_names = sorted([p.stem for p in img_names])
        self.current_index = 0

        self.av2tree = av2tree or NaiveAVSegToTree()

        self.mosaic = Mosaic(
            (2, 3), cols_titles=["Image", "Graph", "Topo"], rows_titles=["Art", "Vei"], cell_height=500
        )
        self.mosaic[0, 1].on_click(partial(self.handle_click, artery=True))
        self.mosaic[1, 1].on_click(partial(self.handle_click, artery=False))
        self.label = Label(value="")

        # Annotation State
        self.debug_info = {}
        self._fundus: FundusData | None = None
        self.img_name: str = ""

        self._trees: None | tuple[VTree, VTree] = None
        self.annotation_ctx: tuple[AnnotationContext, AnnotationContext] = (AnnotationContext(), AnnotationContext())
        self.prev_selected_branch: BranchSelection = BranchSelection()

    def widget(self) -> GridBox:
        btn_layout = Layout(width="80px")
        bPrev = Button(description="Previous", layout=btn_layout)
        bNext = Button(description="Next", layout=btn_layout)
        bSave = Button(description="Save", layout=btn_layout)
        bReset = Button(description="Reset", layout=btn_layout)
        bCompleteReset = Button(description="Complete Reset", layout=btn_layout)

        box = GridBox(
            children=[bPrev, bNext, bSave, bReset, bCompleteReset, self.label],
            layout=Layout(
                width="100%",
                grid_template_columns="repeat(5, 100px)",
                grid_template_rows="auto",
                justify_content="space-around",
            ),
        )
        return box

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

        fundus = FundusData(
            image=self.raw_path / img_name,
            av=self.av_path / img_name,
        )
        segment_od_mac(fundus)
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
        art_file = self.save_path / f"{self.img_name}_art.vtree"
        vei_file = self.save_path / f"{self.img_name}_vei.vtree"
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
        art_file = self.save_path / f"{self.img_name}_art.vtree"
        vei_file = self.save_path / f"{self.img_name}_vei.vtree"
        self.trees[0].save(art_file)
        self.trees[1].save(vei_file)

    def draw_trees(self, which: Literal["artery", "vein", "both"] = "both"):
        if which in ("artery", "both"):
            draw_tree(self.trees[0], view=self.mosaic[0, 1], artery=True)
        if which in ("vein", "both"):
            draw_tree(self.trees[1], view=self.mosaic[1, 1], artery=False)

        for i, tree in enumerate(self.trees):
            if (which == "vein" and i == 0) or (which == "artery" and i == 1):
                continue

            label_map, topo_map = rasterize_tree_topology(tree, bridge_gap_smaller_than=50)
            subtree_map = TopologicalLabel.decode_subtree(label_map)
            N_subtree = subtree_map.max() + 1
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

    def handle_click(self, event, artery: bool):
        modified_tree: Literal["artery", "vein", "both"] = "artery" if artery else "vein"

        art = 0 if artery else 1
        tree = self.trees[art]
        gdata = tree.geometric_data()
        ctx = self.annotation_ctx[art]

        yx = (event["y"], event["x"])

        self.debug_info.clear()
        self.debug_info["event"] = event
        self.debug_info["artery"] = artery
        self.debug_info["yx"] = yx

        if event["button"] != 0 or "alt" not in event["modifiers"]:
            self.prev_selected_branch.reset()

        if event["button"] == 0:  # Left click
            (branch_id, curve_id), dist = gdata.closest_branches(yx, interpolate=True, return_distance=True)
            if dist > 20:
                return
            curve_ratio = curve_id / gdata.branch_arc_length(branch_id, fast_approximation=True)

            if "shift" in event["modifiers"]:
                # Split a branch and add a new node
                tree.split_branch(branch_id, curve_id, split_coord=yx, inplace=True)
            elif "alt" in event["modifiers"]:
                # Connect two branch together
                tip = 0 if curve_ratio < 0.5 else 1
                if not self.prev_selected_branch.is_second_selection(artery):
                    self.prev_selected_branch.select(artery, branch_id, tip)
                    return

                nodes = [
                    tree.branch_list[self.prev_selected_branch.id][self.prev_selected_branch.tip],
                    tree.branch_list[branch_id][tip],
                ]
                tree = self.infer_roots(tree.add_branch(nodes), ctx)
                simplify_passing_nodes(tree, only_fusable=nodes, inplace=True)
                self.prev_selected_branch.reset()
            else:
                return

        elif event["button"] == 1:  # Middle click
            if event["modifiers"] == []:  # No modifiers
                # Toggle force root
                node_id, dist = gdata.closest_nodes(yx, return_distance=True)
                node = tree.node(node_id)
                if dist > 20:
                    return
                if node in ctx.force_roots:
                    ctx.force_roots.remove(node)
                else:
                    ctx.force_roots.add(node)
            elif "alt" in event["modifiers"] or "ctrl" in event["modifiers"]:
                # Swap artery/vein branch
                (branch_id, curve_id), dist = gdata.closest_branches(yx, interpolate=True, return_distance=True)
                if dist > 20:
                    return

                if "ctrl" in event["modifiers"]:
                    # Swap the whole subtree
                    for s in tree.branch_ids_by_subtree():
                        if branch_id in s:
                            branch_id = s
                            break
                    else:
                        return
                subtree = tree.subtree(branch_id)

                other_tree = self.trees[1 - art].append_graph(subtree, inplace=True)
                self.infer_roots(other_tree, self.annotation_ctx[1 - art], inplace=True)
                modified_tree = "both"
                tree.delete_branch(branch_id, inplace=True)

        elif event["button"] == 2 and ("alt" in event["modifiers"] or "ctrl" in event["modifiers"]):
            # Delete branch or subtree
            (branch_id, curve_id), dist = gdata.closest_branches(yx, interpolate=True, return_distance=True)
            if dist > 20:
                return

            if "alt" in event["modifiers"]:
                tree.delete_branch(branch_id, inplace=True)
            else:  # ctrl
                subtrees = tree.branch_ids_by_subtree()
                for subtree in subtrees:
                    if branch_id in subtree:
                        tree.delete_branch(subtree, inplace=True)
                        break

        elif event["button"] == 2:
            # Disconnect crossing
            node_id, dist = gdata.closest_nodes(yx, return_distance=True)
            if dist > 20 or node_id in tree.root_nodes_ids():
                return

            if "shift" in event["modifiers"]:
                _, node_id = disconnect_crossing(tree, node_id, inplace=True, return_new_nodes=True)
                self.infer_roots(tree, ctx, inplace=True)
            simplify_passing_nodes(tree, inplace=True, only_fusable=node_id)
            ctx.force_roots.difference_update({n for n in ctx.force_roots if not n.is_valid()})
        else:
            return

        self.infer_roots(tree, ctx, inplace=True)
        self.draw_trees(which=modified_tree)

    def infer_roots(self, tree: VTree, ctx: AnnotationContext, *, inplace=False) -> VTree:
        return naive_infer_roots(
            tree, root_pos=self.fundus.od_center, force_roots=[n.id for n in ctx.force_roots], inplace=inplace
        )
