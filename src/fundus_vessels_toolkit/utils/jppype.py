from typing import Optional, Tuple

import pandas as pd
from jppype import Mosaic, View2D, imshow, vscode_theme
from jppype.layers import Layer, LayerImage, LayerQuiver


from ..vascular_data_objects.fundus_data import AVLabel
from ..vascular_data_objects import VTree, VGraph

vscode_theme()


GRAPH_AV_COLORS = {
    AVLabel.BKG: "grey",
    AVLabel.ART: "red",
    AVLabel.VEI: "blue",
    AVLabel.BOTH: "purple",
    AVLabel.UNK: "green",
}


def draw_tree(tree: VTree, view: View2D, artery: bool, name="tree", edge_labels=False, node_labels=False) -> None:
    layer = tree.jppype_layer(bspline=True, edge_labels=edge_labels, node_labels=node_labels)

    if artery:
        root_color = "#7a1a1a"
        leaf_color = "#da7676"
        label = AVLabel.ART
    else:
        root_color = "#1a1a7a"
        leaf_color = "#7676da"
        label = AVLabel.VEI
    main_color = GRAPH_AV_COLORS[label]
    nodes_color = pd.Series(main_color, index=tree.node_attr.index)
    nodes_color[tree.root_nodes_ids()] = root_color
    nodes_color[tree.leaf_nodes_ids()] = leaf_color
    layer.nodes_cmap = nodes_color.to_dict()
    layer.edges_cmap = main_color

    view[name] = layer


def draw_trees(trees: Tuple[VTree, VTree], view: View2D, edge_labels=False, node_labels=False) -> None:
    """
    Draw a vessel tree on a given view.

    Parameters
    ----------
    tree : VTree
        The vessel tree to draw.
    view : View2D | Mosaic
        The view to draw the tree on.
    """
    draw_tree(trees[0], view, artery=True, name="artery_tree", edge_labels=edge_labels, node_labels=node_labels)
    draw_tree(trees[1], view, artery=False, name="vein_tree", edge_labels=edge_labels, node_labels=node_labels)


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
            layer.nodes_cmap = graph.node_attr[av_attr].map(GRAPH_AV_COLORS).to_dict()
        if av_attr in graph.branch_attr:
            layer.edges_cmap = graph.branch_attr[av_attr].map(GRAPH_AV_COLORS).to_dict()
    view["vessel_graph"] = layer
