from typing import Dict, Literal, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from coloraide import Color
from jppype import Mosaic, View2D, View2dGroup, imshow, vscode_theme
from jppype.layers import Layer, LayerGraph, LayerImage, LayerQuiver

from fundus_toolkits import AVLabel
from fundus_toolkits.utils.color import ColorSpec, parse_color
from fundus_toolkits.utils.geometric import Point

from ..vascular_data_objects import VGraph, VTree

AV_COLORS: Dict[AVLabel, str] = {
    AVLabel.BKG: "grey",
    AVLabel.ART: "red",
    AVLabel.VEI: "blue",
    AVLabel.BOTH: "purple",
    AVLabel.UNK: "green",
}


def subgraph_colormap(x):
    cmap = [
        "red",
        "blue",
        "purple",
        "green",
        "orange",
        "cyan",
        "pink",
        "yellow",
        "teal",
        "lime",
        "magenta",
        "brown",
        "navy",
        "olive",
        "maroon",
        "aqua",
        "fuchsia",
        "silver",
        "gold",
        "coral",
        "indigo",
        "violet",
    ]
    return cmap[x % len(cmap)]


def draw_tree(
    tree: VTree,
    view: View2D | View2dGroup,
    artery: Optional[bool] = None,
    name="tree",
    edge_labels=False,
    node_labels=False,
    edge: Literal["bspline", "line", "skeleton"] = "bspline",
    branch_color: Literal["av", "rank", "subtree"] = "rank",
    bspline_dir: bool = False,
) -> LayerGraph:
    bsplines = []
    layer = tree.jppype_layer(
        edge_map=edge == "skeleton",
        bspline=edge == "bspline",
        edge_labels=edge_labels,
        node_labels=node_labels,
        bsplines_out=bsplines,
    )

    if bspline_dir and edge == "bspline":
        geodata = tree.geometric_data()
        branch_dir = tree.branch_dirs()
        nodes_coord = geodata.node_coord()
        for i, path in enumerate(layer._edges_path):
            n1, n2 = [Point(*nodes_coord[_]) for _ in tree.branch_list[i]]
            bspline = bsplines[i].extend_bspline(start=n1, end=n2, smoothing=0.5)

            t = bspline.relative_pos_to_t(0.5)
            p = Point.from_array(bspline.evaluate(t))
            tan = Point.from_array(bspline.evaluate_tangent(t, normalized=True)) * 10
            if not branch_dir[i]:
                tan = -tan

            left, right = p - tan.rotate(np.pi / 4), p - tan.rotate(-np.pi / 4)

            path = path + f" M {left.x:.2f} {left.y:.2f} L {p.x:.2f} {p.y:.2f} L {right.x:.2f} {right.y:.2f}"
            layer._edges_path[i] = path

    if artery is None:
        if "av" in tree.node_attr:
            layer.nodes_cmap = tree.node_attr["av"].fillna(0).map(AV_COLORS).to_dict()
        else:
            layer.nodes_cmap = AV_COLORS[AVLabel.UNK]
        main_color = AV_COLORS[AVLabel.UNK]
    else:
        if artery is True:
            root_color = "#7a1a1a"
            leaf_color = "#da7676"
            label = AVLabel.ART
        elif artery is False:
            root_color = "#1a1a7a"
            leaf_color = "#7676da"
            label = AVLabel.VEI
        main_color = AV_COLORS[label]
        nodes_color = pd.Series(main_color, index=tree.node_attr.index)
        nodes_color[tree.root_nodes_ids()] = root_color
        nodes_color[tree.leaf_nodes_ids()] = leaf_color
        layer.nodes_cmap = nodes_color.to_dict()

    if branch_color == "rank" and "rank" in tree.node_attr:
        MAX_RANK = 4
        if artery:
            gradient = Color.interpolate(["#ff0000", "#ff7a7a"], space="lab")
        else:
            gradient = Color.interpolate(["#0000ff", "#7a7aff"], space="lab")
        edge_gradient = gradient.steps(MAX_RANK)
        edge_gradient = [edge_gradient[x].convert("srgb").to_string(hex=True) for x in range(MAX_RANK)]
        edges_rank = tree.node_attr["rank"][tree.branch_head()].clip(1, MAX_RANK) - 1
        layer.edges_cmap = [edge_gradient[x] for x in edges_rank]

    elif branch_color == "subtree":
        layer.edges_cmap = pd.Series(tree.subtrees_branch_labels()).map(subgraph_colormap).to_dict()
    else:
        layer.edges_cmap = main_color

    view[name] = layer
    return layer


def draw_trees(
    trees: Tuple[VTree, VTree],
    view: View2D | View2dGroup,
    edge_labels=False,
    node_labels=False,
    edge: Literal["bspline", "line", "skeleton"] = "bspline",
    branch_color: Literal["av", "rank", "subtree"] = "rank",
    bspline_dir: bool = False,
) -> None:
    """
    Draw a vessel tree on a given view.

    Parameters
    ----------
    tree : VTree
        The vessel tree to draw.
    view : View2D | Mosaic
        The view to draw the tree on.
    """
    draw_tree(
        trees[0],
        view,
        artery=True,
        name="artery_tree",
        edge_labels=edge_labels,
        node_labels=node_labels,
        edge=edge,
        branch_color=branch_color,
        bspline_dir=bspline_dir,
    )
    draw_tree(
        trees[1],
        view,
        artery=False,
        name="vein_tree",
        edge_labels=edge_labels,
        node_labels=node_labels,
        edge=edge,
        branch_color=branch_color,
        bspline_dir=bspline_dir,
    )


def draw_graph(
    graph: VGraph,
    view: View2D,
    edge: Literal["bspline", "line", "skeleton", "skeleton-dot"] = "bspline",
    branch_color: Literal["av", "subtree", "branch"] | Dict[int, float | str] | Sequence[float | str] = "av",
    branch_color_scale: Optional[str] = None,
    av_attr: Optional[str] = None,
    edge_labels: bool = False,
    node_labels: bool = False,
) -> None:
    """
    Draw a vessel graph on a given view.

    Parameters
    ----------
    graph : VGraph
        The vessel graph to draw.
    view : View2D | Mosaic
        The view to draw the graph on.
    av_attr : str | None
        The attribute to use for coloring the vessels.
    """
    layer = graph.jppype_layer(
        edge_labels=edge_labels,
        node_labels=node_labels,
        edge_map=edge.startswith("skeleton"),
        interpolate=edge == "skeleton",
        bspline=edge == "bspline",
    )
    if isinstance(branch_color, str) and branch_color == "av":
        if av_attr is None:
            av_attr = "av"
        if av_attr in graph.node_attr:
            layer.nodes_cmap = graph.node_attr[av_attr].fillna(0).map(AV_COLORS).to_dict()
        if av_attr in graph.branch_attr:
            layer.edges_cmap = graph.branch_attr[av_attr].fillna(0).map(AV_COLORS).to_dict()
    if isinstance(branch_color, str) and branch_color == "subtree":
        layer.edges_cmap = pd.Series(graph.subgraph_branch_labels()).map(subgraph_colormap).to_dict()

    if isinstance(branch_color, (Sequence, np.ndarray, pd.Series)) and not isinstance(branch_color, str):
        branch_color = {i: c for i, c in enumerate(branch_color) if c is not None and i < graph.branch_count}
    if isinstance(branch_color, dict):
        if branch_color_scale is None:
            branch_color_scale = "viridis"

        cmap = matplotlib.colormaps.get_cmap(branch_color_scale)

        def parse_color(c):
            if isinstance(c, str):
                return Color(c).convert("srgb").to_string(hex=True)
            else:
                return Color("srgb", cmap(np.clip(c, 0, 1))[:3]).to_string(hex=True)

        branch_color = {k: parse_color(c) for k, c in branch_color.items()}
        branch_color[None] = "grey"
        layer.edges_cmap = branch_color

    view["vessel_graph"] = layer
