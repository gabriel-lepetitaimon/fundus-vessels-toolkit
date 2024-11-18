from typing import Optional

import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import HTML, GridBox, Layout
from jppype.utils.color import colormap_by_name
from plotly import graph_objects as go

from ..utils.fundus_projections import FundusProjection
from ..utils.jppype import Mosaic
from ..utils.lookup_array import complete_lookup, invert_complete_lookup
from ..vascular_data_objects import VGraph


def inspect_matching(
    nodes_similarity: np.ndarray,
    matching: np.ndarray,
    src_nodes_id: np.ndarray,
    dst_nodes_id: np.ndarray,
    src_graph: VGraph,
    dst_graph: VGraph,
    src_raw: np.ndarray,
    dst_raw: np.ndarray,
    src_features: Optional[pd.DataFrame] = None,
    dst_features: Optional[pd.DataFrame] = None,
    branch_matching: Optional[np.ndarray] = None,
    matching_gt: Optional[np.ndarray] = None,
):
    # === INITIALIZATION ===
    # Copy the graphs to avoid modifying the original ones
    src_graph, dst_graph = src_graph.copy(), dst_graph.copy()

    # Check input consistency
    src_nodes_id = np.asarray(src_nodes_id, dtype=int)
    assert src_nodes_id.ndim == 1, "nodes_id1 must be a 1D numpy array"
    dst_nodes_id = np.asarray(dst_nodes_id, dtype=int)
    assert dst_nodes_id.ndim == 1, "nodes_id2 must be a 1D numpy array"
    N_src, N_dst = len(src_nodes_id), len(dst_nodes_id)

    nodes_similarity = np.asarray(nodes_similarity)
    assert nodes_similarity.shape == (N_src, N_dst), f"Invalid similarity matrix: shape should be {(N_src, N_dst)}"

    matching = np.asarray(matching, dtype=int)
    assert matching.ndim == 2 and matching.shape[0] == 2, "Invalid matched nodes: shape should be (2, N)"
    src_match, dst_match = matching
    N_match = len(src_match)

    assert isinstance(src_graph, VGraph) and isinstance(dst_graph, VGraph), "graph1 and graph2 must be VGraph instances"
    src_raw, dst_raw = np.asarray(src_raw), np.asarray(dst_raw)
    assert src_raw.ndim in (2, 3) and dst_raw.ndim in (2, 3), "raw1 and raw2 must be images"

    if matching_gt is not None:
        # Sort the junctions to match the ground truth
        matching_gt = np.asarray(matching_gt, dtype=int)
        assert (
            matching_gt.ndim == 2 and matching_gt.shape[0] == 2
        ), "Invalid ground truth matching: shape should be (2, N)"
        N_gt_match = len(matching_gt[0])
        src_gt, dst_gt = matching_gt
        src_reorder = complete_lookup(src_gt, N_src - 1)
        dst_reorder = complete_lookup(dst_gt, N_dst - 1)
        src_match = invert_complete_lookup(src_reorder)[src_match]
        dst_match = invert_complete_lookup(dst_reorder)[dst_match]
    else:
        # Sort the junctions according to the matching
        src_reorder = complete_lookup(src_match, N_src - 1)
        dst_reorder = complete_lookup(dst_match, N_dst - 1)
        src_match = dst_match = np.arange(N_match)
        N_gt_match = N_match

    src_graph.reindex_nodes(src_nodes_id[src_reorder], inverse_lookup=True, inplace=True)
    dst_graph.reindex_nodes(dst_nodes_id[dst_reorder], inverse_lookup=True, inplace=True)
    if src_features is not None:
        src_features.set_index(invert_complete_lookup(src_reorder), inplace=True)
    if dst_features is not None:
        dst_features.set_index(invert_complete_lookup(dst_reorder), inplace=True)
    src_nodes_id, dst_nodes_id = np.arange(N_src), np.arange(N_dst)
    nodes_similarity[src_reorder][:, dst_reorder] = nodes_similarity

    if matching_gt is None:
        true_match = np.arange(N_match)
    else:
        true_match = (src_match == dst_match) & (src_match < N_gt_match)
        true_match = np.argwhere(true_match).flatten()

    dst_match_by_src = -np.ones(N_src, dtype=int)
    dst_match_by_src[src_match] = dst_match
    src_match_by_dst = -np.ones(N_dst, dtype=int)
    src_match_by_dst[dst_match] = src_match

    # === VISUALIZATION ===
    # --- Create the mosaic ---
    graph_viewer = Mosaic(2, cell_height=800, sync=False)
    for i, (raw, g, nodes_id, matched_nodes) in enumerate(
        [(src_raw, src_graph, src_nodes_id, src_match), (dst_raw, dst_graph, dst_nodes_id, dst_match)]
    ):
        graph_viewer[i].add_image(raw * 0.8, "background")
        graph_viewer[i]["vgraph"] = g.jppype_layer(edge_map=False, bspline=True, node_labels=True, edge_labels=True)

        # Color matched nodes and branches
        if matching_gt is None:
            nodes_cmap = {None: colormap_by_name()} | {
                int(_): "#444" for _ in np.setdiff1d(np.arange(g.node_count), true_match)
            }
        else:
            nodes_cmap = (
                {None: "#444"}
                | {int(nodes_id[_]): "orange" for _ in np.arange(N_gt_match)}
                | {int(nodes_id[n]): "#4CC417" if i in true_match else "red" for i, n in enumerate(matched_nodes)}
            )
        graph_viewer[i]["vgraph"].nodes_cmap = nodes_cmap
        if branch_matching is None:
            invalid_branches = np.setdiff1d(np.arange(g.branch_count), g.incident_branches(true_match))
        else:
            invalid_branches = np.setdiff1d(np.arange(g.branch_count), branch_matching[i])
        branch_cmap = {None: colormap_by_name()} | {int(_): "#444" for _ in invalid_branches}
        graph_viewer[i]["vgraph"].edges_cmap = branch_cmap

    # --- Create the similarity matrix display ---
    y, x = np.arange(N_src), np.arange(N_dst)
    y_range, x_range = [x[0], x[-1]], [x[0], y[-1]]

    fig = go.FigureWidget()
    hovertext = [
        [
            f"src: <b>{src}</b>, dst: <b>{dst}</b><br />"
            f"dst match: {dst_match_by_src[src]} <br />src match: {src_match_by_dst[dst]}<br />"
            f"similarity: <b>{nodes_similarity[src, dst]:.2f}</b><br />"
            for dst in x
        ]
        for src in y
    ]
    hover_data = dict(text=hovertext, hoverinfo="text")
    z = nodes_similarity
    # z[np.arange(N_gt_match), np.arange(N_gt_match)] *= -1
    COLORSCALE = [[0.0, "#666"], [0.5, "#fff"], [1.0, "#8cc350"]]
    heatmap_style = dict(colorscale=COLORSCALE, zmid=0, showscale=False)
    heatmap = go.Heatmap(z=z, y=y, x=x, **heatmap_style, **hover_data)
    fig.add_trace(heatmap)

    diag_line_style = dict(mode="lines", marker_color="white", line_width=1, hoverinfo="skip")
    diag_line = go.Scatter(x=[x[0], x[N_gt_match - 1]], y=[y[0], y[N_gt_match - 1]], **diag_line_style)
    fig.add_trace(diag_line)

    points_style = dict(type="circle", xref="x", yref="y", fillcolor="yellow", line_width=0)
    r = 0.5
    points = [
        go.layout.Shape(x0=x - r, y0=y - r, x1=x + r, y1=y + r, **points_style)
        for y, x in np.stack((src_match, dst_match)).T
    ]
    fig.update_layout(shapes=points)

    axes_opts = dict(type="category", categoryorder="array")
    fig.update_yaxes(categoryarray=y, autorange="reversed", range=y_range, title="Source", **axes_opts)
    fig.update_xaxes(categoryarray=x, range=x_range, title="Destination", **axes_opts)
    fig.update_layout(width=700, height=700)

    display(
        GridBox(
            [fig] + graph_viewer._views,
            layout=Layout(grid_template_columns="repeat(3, 1fr)", grid_template_rows="700px"),
        )
    )

    if src_features is not None or dst_features is not None:
        displayed_nodes = []

        features_html = HTML("")
        display(features_html)

        def display_node_features(src_node, dst_node):
            try:
                displayed_nodes.remove((dst_node, False))
            except ValueError:
                pass
            displayed_nodes.insert(0, (dst_node, False))
            try:
                displayed_nodes.remove((src_node, True))
            except ValueError:
                pass
            displayed_nodes.insert(0, (src_node, True))

            df = pd.DataFrame(
                {
                    f"SRC[{n}]" if is_src else f"DST[{n}]": (src_features.loc[n] if is_src else dst_features.loc[n])
                    for n, is_src in displayed_nodes
                    if (is_src and src_features is not None and n in src_features.index)
                    or (not is_src and dst_features is not None and n in dst_features.index)
                }
            ).T

            features_html.value = df.to_html()

    # Link the mosaic and the heatmap
    def zoom_on(src_node, dst_node):
        src_coord = src_graph.node_coord()[src_node]
        dst_coord = dst_graph.node_coord()[dst_node]

        graph_viewer[0].goto((src_coord[1], src_coord[0]), 3)
        graph_viewer[1].goto((dst_coord[1], dst_coord[0]), 3)
        if src_features is not None or dst_features is not None:
            display_node_features(src_node, dst_node)

    def heatmap_on_click(trace, points, selector):
        if not points.ys:
            return
        y, x = int(points.ys[0]), int(points.xs[0])
        zoom_on(y, x)

    fig.data[0].on_click(heatmap_on_click)

    # === EVALUATION ===
    if matching_gt is not None:
        TP = len(true_match)
        print(f"Precision: {TP} / {N_gt_match} = {TP/N_gt_match:.0%} \n" f"Recall: {TP} / {N_match} = {TP/N_match:.0%}")


def inspect_registration(
    fix_graph: VGraph,
    moving_graph: VGraph,
    transform: FundusProjection,
    raw1: np.ndarray,
    raw2: np.ndarray,
    mask1: Optional[np.ndarray] = None,
    mask2: Optional[np.ndarray] = None,
):
    from ..utils.stitching import fundus_circular_fuse_mask, stitch_images

    masks = [
        fundus_circular_fuse_mask(img if m is None else m) for img, m in zip([raw1, raw2], [mask1, mask2], strict=True)
    ]
    stitched_img, stitched_rect, warped_rects = stitch_images(
        [raw1, raw2], [None, transform], masks, return_stitch_domain=True, return_warped_domains=True
    )
    moving_graph = moving_graph.transform(transform)

    mosaic = Mosaic(1, cell_height=800)
    v = mosaic[0]
    v.add_image(stitched_img, "stitched_image", domain=stitched_rect)
    v["graph1"] = fix_graph.jppype_layer(edge_map=True, node_labels=True)
    v["graph1"].domain = fix_graph.geometric_data().domain
    v["graph1"].nodes_cmap = {None: "cyan"}
    v["graph1"].edges_cmap = {None: "cyan"}
    v["graph2"] = moving_graph.jppype_layer(edge_map=True, node_labels=True)
    v["graph2"].domain = moving_graph.geometric_data().domain
    v["graph2"].nodes_cmap = {None: "purple"}
    v["graph2"].edges_cmap = {None: "purple"}
    mosaic.show()

    return mosaic, fix_graph, moving_graph
