from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from coloraide import Color
from jppype import Mosaic, View2D, View2dGroup, imshow, vscode_theme
from jppype.layers import Layer, LayerGraph, LayerImage, LayerQuiver

from fundus_toolkits import AVLabel
from fundus_toolkits.utils.geometric import Point

from ..vascular_data_objects import VGraph, VTree
from ..vascular_data_objects.vbranch_geodata import VBranchGeoData
from .bezier import BSpline, BezierCubic

vscode_theme()


AV_COLORS: Dict[AVLabel, str] = {
    AVLabel.BKG: "grey",
    AVLabel.ART: "red",
    AVLabel.VEI: "blue",
    AVLabel.BOTH: "purple",
    AVLabel.UNK: "green",
}


def draw_tree(
    tree: VTree,
    view: View2D | View2dGroup,
    artery: bool,
    name="tree",
    edge_labels=False,
    node_labels=False,
    edge: Literal["bspline", "line", "skeleton"] = "bspline",
    branch_color: Literal["av", "rank", "subtree"] = "rank",
    bspline_dir: bool = False,
) -> LayerGraph:
    layer = tree.jppype_layer(
        edge_map=edge == "skeleton", bspline=edge == "bspline", edge_labels=edge_labels, node_labels=node_labels
    )

    if bspline_dir and edge == "bspline":
        geodata = tree.geometric_data()
        branches_bspline = geodata.branch_data(VBranchGeoData.Fields.BSPLINE)
        branch_dir = tree.branch_dirs()
        nodes_coord = geodata.node_coord()
        for i, path in enumerate(layer._edges_path):
            n1, n2 = [Point(*nodes_coord[_]) for _ in tree.branch_list[i]]
            bspline = branches_bspline[i]
            if isinstance(bspline, VBranchGeoData.BSpline):
                bspline = bspline.data.extend_bspline(start=n1, end=n2, smoothing=0.5)
            else:
                bspline = BSpline([BezierCubic(n1, n1, n2, n2)])

            t = bspline.relative_pos_to_t(0.5)
            p = Point.from_array(bspline.evaluate(t))
            tan = Point.from_array(bspline.evaluate_tangent(t, normalized=True)) * 10
            if not branch_dir[i]:
                tan = -tan

            left, right = p - tan.rotate(np.pi / 4), p - tan.rotate(-np.pi / 4)

            path = path + f" M {left.x:.2f} {left.y:.2f} L {p.x:.2f} {p.y:.2f} L {right.x:.2f} {right.y:.2f}"
            layer._edges_path[i] = path

    if artery:
        root_color = "#7a1a1a"
        leaf_color = "#da7676"
        label = AVLabel.ART
    else:
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

        def colormap(x):
            cmap = ["red", "blue", "purple", "green", "orange", "cyan", "pink", "yellow", "teal", "lime"]
            return cmap[x % len(cmap)]

        layer.edges_cmap = pd.Series(tree.subtrees_branch_labels()).map(colormap).to_dict()

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
    )


def draw_graph(
    graph: VGraph, view: View2D, av_attr: Optional[str] = None, edge_labels: bool = False, node_labels: bool = False
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
    layer = graph.jppype_layer(edge_labels=edge_labels, node_labels=node_labels, bspline=True)
    if av_attr:
        if av_attr in graph.node_attr:
            layer.nodes_cmap = graph.node_attr[av_attr].map(AV_COLORS).to_dict()
        if av_attr in graph.branch_attr:
            layer.edges_cmap = graph.branch_attr[av_attr].map(AV_COLORS).to_dict()
    view["vessel_graph"] = layer
