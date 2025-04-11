import warnings
from typing import Literal, Optional

import numpy as np

from ..utils.exceptions import GeometryParserWarning
from ..utils.geometric import Rect
from ..utils.graph.measures import extract_branch_geometry
from ..utils.math import intercept_segment
from ..vascular_data_objects import FundusData, VBranchGeoData, VGraph


def populate_geometry(
    vgraph: VGraph,
    vessel_segmentation: Optional[np.ndarray | FundusData] = None,
    *,
    adaptative_tangents=True,
    bspline_target_error: float = 2,
    populate_tip_geodata: bool = True,
    split_on_gaps: float = 5,
    inplace: bool = False,
) -> VGraph:
    """Populate the geometric data of the branches of the graph.

    Parameters
    ----------
    vgraph : VGraph
        The graph to populate with geometric data.
    vessel_segmentation : np.ndarray
        The binary segmentation of the vessels.
    adaptative_tangents : bool, optional
        If True, the tangent of the branches are computed adaptatively, by default False.
    bspline_target_error : float, optional
        The target error for the bspline interpolation, by default 3.
    inplace : bool, optional
        If True, the geometric data is added to the input graph, by default False.
    populate_tip_geodata : bool, optional
        If True, use the geometry of the branch to populate the tip geometrical data, by default True.


    Returns
    -------
    VGraph
        The graph with the geometric data populated.
    """
    if not inplace:
        vgraph = vgraph.copy()

    geo_data = vgraph.geometric_data()

    if vessel_segmentation is None:
        try:
            vessel_segmentation = geo_data.fundus_data.vessels
        except AttributeError:
            raise ValueError("No vessel segmentation provided and no vessel segmentation found in the graph.") from None
    elif isinstance(vessel_segmentation, FundusData):
        vessel_segmentation = vessel_segmentation.vessels

    curves = geo_data.branch_curve()
    valid_curves_id = np.argwhere([c is not None and len(c) > 1 for c in curves]).flatten()
    valid_curves = [curves[int(i)] for i in valid_curves_id]
    curves, curve_splits, tangents, calibres, boundaries, curvatures, curv_roots, bsplines = extract_branch_geometry(
        valid_curves,
        vessel_segmentation,
        adaptative_tangents=adaptative_tangents,
        bspline_target_error=bspline_target_error,
        split_on_gaps=split_on_gaps,
        return_boundaries=True,
        return_calibre=True,
        return_curvature=True,
        return_curvature_roots=True,
        extract_bspline=True,
    )

    # Check if the calibres are valid
    # if not all(valid_calibres := [np.all(abs(np.diff(c)) < 4) for c in calibres]):
    #     invalid_calibres = np.where(~np.array(valid_calibres, dtype=bool))[0]
    #     warnings.warn(
    #         f"Calibres are not continuous for branches {invalid_calibres}",
    #         GeometryParserWarning,
    #         stacklevel=2,
    #     )
    # if not all(valid_calibres := [np.any(c) <= 40 and np.all(np.isfinite(c)) for c in calibres]):
    #     invalid_calibres = np.where(~np.array(valid_calibres, dtype=bool))[0]
    #     warnings.warn(
    #         f"Calibres larger than 40px for branches {invalid_calibres}",
    #         GeometryParserWarning,
    #         stacklevel=2,
    #     )

    for i, c in enumerate(curves):
        if c is not None and len(c) >= 3:
            curves[i] = c
        else:
            curves[i] = None
            tangents[i] = None
            calibres[i] = None
            boundaries[i] = None
            curvatures[i] = None
            curv_roots[i] = None
            bsplines[i] = None

    geo_data.set_branch_curve(curves, branch_id=valid_curves_id, graph_index=True)
    geo_data.set_branch_data(VBranchGeoData.Fields.TANGENTS, tangents, branch_id=valid_curves_id, graph_index=True)
    geo_data.set_branch_bspline(bsplines, branch_id=valid_curves_id, graph_index=True)
    geo_data.set_branch_data(VBranchGeoData.Fields.CALIBRES, calibres, branch_id=valid_curves_id, graph_index=True)
    geo_data.set_branch_data(VBranchGeoData.Fields.BOUNDARIES, boundaries, branch_id=valid_curves_id, graph_index=True)
    geo_data.set_branch_data(VBranchGeoData.Fields.CURVATURES, curvatures, branch_id=valid_curves_id, graph_index=True)
    geo_data.set_branch_data(
        VBranchGeoData.Fields.CURVATURE_ROOTS, curv_roots, branch_id=valid_curves_id, graph_index=True
    )

    if split_on_gaps > 0:
        for i, curve, split in zip(valid_curves_id, curves, curve_splits, strict=True):
            if curve is not None and len(split):
                split_coord = curve[np.stack([split - 1, split])].mean(axis=0)
                vgraph.split_branch(i, split_coord=split_coord, split_curve_id=split, inplace=True)

    if populate_tip_geodata:
        derive_tips_geometry_from_curve_geometry(vgraph, inplace=True)

    return vgraph


def derive_tips_geometry_from_curve_geometry(
    vgraph: VGraph,
    *,
    calibre: None | bool | int = None,
    tangent: Literal["bspline"] | bool | int | None = None,
    boundaries: bool | None = None,
    tangent_from_nodes: bool | int = True,
    inplace: bool = False,
) -> VGraph:
    """Derive the geometry of tips from the curve geometry of the branches.

    Parameters
    ----------
    vgraph : VGraph
        The graph to derive the tips geometry from.


    calibre : bool | int, optional
        If neither False nor null, derive ``TERMINATION_CALIBRE`` from ``CALIBRE``.
        If True, average the calibre of the 10 points closest to the tips.
        If a int, average the calibre of the n points closest to the tips.

    tangent : Literal["bspline"] | bool | int | None, optional
        If neither False nor null, derive ``TERMINATION_TANGENT`` from ``TANGENT``.
        If True, average the tangent of the 10 points closest to the tips.
        If a int, average the tangent of the n points closest to the tips.

    boundaries : bool | None, optional
        If neither False nor null, derive ``TERMINATION_BOUNDARIES`` from ``BOUNDARIES``.

    tangent_from_nodes : bool | int, optional
        If not False, when the tangent is not available, derive it from the nodes coordinates if the distance between the nodes is less than the value (30 px if True).


    inplace : bool, optional
        If True, the graph is modified in place, by default False.

    Returns
    -------
    VGraph
        The graph with the tips geometry set.
    """  # noqa: E501
    if not inplace:
        vgraph = vgraph.copy()

    gdata = vgraph.geometric_data()

    # === Calibre ===
    if calibre is None and gdata.has_branch_data(VBranchGeoData.Fields.CALIBRES):
        calibre = True
    if calibre:
        if isinstance(calibre, bool):
            calibre = 10

        tips_calibre = []
        # Fetch the branch calibres
        branches_calibres = gdata.branch_data(VBranchGeoData.Fields.CALIBRES, graph_index=False)
        # Compute the mean of the branch calibre at the tips
        for branch_calibres in branches_calibres:
            if branch_calibres:
                d = branch_calibres.data
                tips_calibre.append(np.array([d[:calibre].mean(), d[-calibre:].mean()]))
            else:
                tips_calibre.append(None)
        # Store the tips calibre back as TERMINATION_CALIBRE
        gdata.set_branch_data(VBranchGeoData.Fields.TIPS_CALIBRE, tips_calibre, graph_index=False, no_check=True)

    if tangent_from_nodes is True:
        tangent_from_nodes = 30
    elif tangent_from_nodes is False:
        tangent_from_nodes = 0

    # === Tangents ===
    if tangent is None:
        if gdata.has_branch_data(VBranchGeoData.Fields.BSPLINE):
            tangent = "bspline"
        elif gdata.has_branch_data(VBranchGeoData.Fields.TANGENTS):
            tangent = True

    if tangent:
        tips_tangents = []
        nodes_coord = gdata.node_coord()
        branch_list = vgraph.branch_list  # TODO: transpose to graph_index=False

        if tangent == "bspline":
            branches_bsplines = gdata.branch_bspline(attr=VBranchGeoData.Fields.BSPLINE)
            for i, branch_bspline in enumerate(branches_bsplines):
                if branch_bspline:
                    tips_tangents.append(np.array(branch_bspline.tips_tangents(normalize=True), dtype=float))
                elif (
                    tangent_from_nodes
                    and (dist := np.linalg.norm(u21 := np.subtract(*nodes_coord[branch_list[i]]))) < tangent_from_nodes
                ):
                    if dist < 1e-6:
                        tips_tangents.append(np.zeros((2, 2), dtype=float))
                    else:
                        u21 = u21 / dist
                        tips_tangents.append(np.array([-u21, u21], dtype=float))
                else:
                    tips_tangents.append(np.zeros((2, 2), dtype=float))
        else:
            if tangent is True:
                tangent = 10
            branches_tangents = gdata.branch_data(VBranchGeoData.Fields.TANGENTS, graph_index=False)
            for i, branch_tangent in enumerate(branches_tangents):
                if branch_tangent:
                    d = branch_tangent.data
                    tips_tangents.append(np.stack([d[:tangent].mean(axis=0), -d[-tangent:].mean(axis=0)]))
                elif (
                    tangent_from_nodes
                    and (dist := np.linalg.norm(u21 := np.subtract(*nodes_coord[branch_list[i]]))) < tangent_from_nodes
                ):
                    u21 = u21 / (dist + 1e-6)
                    tips_tangents.append(np.array([-u21, u21], dtype=float))
                else:
                    tips_tangents.append(None)
        gdata.set_branch_data(VBranchGeoData.Fields.TIPS_TANGENT, tips_tangents, graph_index=False, no_check=True)

    # === Boundaries ===
    if boundaries is None and gdata.has_branch_data(VBranchGeoData.Fields.BOUNDARIES):
        boundaries = True
    if boundaries:
        tips_bounds = []
        branches_bounds = gdata.branch_data(VBranchGeoData.Fields.BOUNDARIES, graph_index=False)
        for branch_bounds in branches_bounds:
            if branch_bounds:
                d = branch_bounds.data
                tips_bounds.append(np.array([d[0], d[-1]]))
            else:
                tips_bounds.append(None)
        gdata.set_branch_data(VBranchGeoData.Fields.TIPS_BOUNDARIES, tips_bounds, graph_index=False, no_check=True)

    return vgraph


def center_junction_nodes(
    graph: VGraph, *, tangent_tips: VBranchGeoData.Key = VBranchGeoData.Fields.TIPS_TANGENT, inplace: bool = False
) -> VGraph:
    """Place the nodes at the center of the branches bifurcations.

    Each node which has three incident branches or more is placed at the barycenter of the intersection of the tangents from the branches tips.

    Parameters
    ----------
    graph : VGraph
        The graph to center the nodes of.

    tangent_tips : VBranchGeoData.Fields.TIPS_TANGENT
        The field containing the tangent of the tips of the branches.

    inplace : bool, optional
        If True, the graph is modified in place, by default False.

    Returns
    -------
    VGraph
        The graph with the nodes centered.
    """  # noqa: E501
    if not inplace:
        graph = graph.copy()

    if graph.node_count == 0:
        return graph

    for gdata in graph._geometric_data:
        tips_data = gdata.tip_data_around_node(attrs=[tangent_tips], graph_index=False)
        all_tips_yx = tips_data["yx"]
        all_tips_t = tips_data[tangent_tips]

        for node_id, (tips_yx, tips_t) in enumerate(zip(all_tips_yx, all_tips_t, strict=True)):
            n_branch = len(tips_yx)
            if n_branch < 3:
                continue

            # Compute the tangent intersections
            a0, a1, b0, b1 = [], [], [], []
            intercepts = []
            for t0 in range(n_branch - 1):
                a0 = tips_yx[t0]
                a1 = tips_yx[t0] - tips_t[t0]
                b0 = tips_yx[t0 + 1 :]
                b1 = tips_yx[t0 + 1 :] - tips_t[t0 + 1 :]
                intercepts.append(intercept_segment(a0, a1, b0, b1, a1_bound=False, b1_bound=False).squeeze(0))
            intercepts = np.concatenate(intercepts)
            # intercepts = intercepts[np.isfinite(intercepts).all(axis=1)]
            intercepts = intercepts[Rect.bounding_box(tips_yx).contains(intercepts)]
            if len(intercepts):
                gdata._nodes_coord[node_id] = intercepts.mean(axis=0)

    return graph
