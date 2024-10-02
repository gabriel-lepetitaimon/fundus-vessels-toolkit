from typing import Literal, Optional

import numpy as np

from ..utils.graph.measures import extract_branch_geometry
from ..vascular_data_objects import FundusData, VBranchGeoData, VGraph


def populate_geometry(
    vgraph: VGraph,
    vessel_segmentation: Optional[np.ndarray | FundusData] = None,
    *,
    adaptative_tangents=False,
    bspline_target_error: float = 2,
    populate_tip_geodata: bool = True,
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
        return populate_geometry(
            vgraph,
            vessel_segmentation,
            adaptative_tangents=adaptative_tangents,
            bspline_target_error=bspline_target_error,
            inplace=True,
        )

    geo_data = vgraph.geometric_data()

    if vessel_segmentation is None:
        try:
            vessel_segmentation = geo_data.fundus_data.vessels
        except AttributeError:
            raise ValueError("No vessel segmentation provided and no vessel segmentation found in the graph.") from None
    elif isinstance(vessel_segmentation, FundusData):
        vessel_segmentation = vessel_segmentation.vessels

    tangents, calibres, boundaries, curvatures, curv_roots, bsplines = extract_branch_geometry(
        geo_data.branch_curve(),
        vessel_segmentation,
        adaptative_tangent=adaptative_tangents,
        bspline_target_error=bspline_target_error,
        bspline_max_gap=7,
        return_boundaries=True,
        return_calibre=True,
        return_curvature=True,
        return_curvature_roots=True,
        extract_bspline=True,
    )
    geo_data.set_branch_data(VBranchGeoData.Fields.TANGENTS, tangents)
    geo_data.set_branch_bspline(bsplines)
    geo_data.set_branch_data(VBranchGeoData.Fields.CALIBRES, calibres)
    geo_data.set_branch_data(VBranchGeoData.Fields.BOUNDARIES, boundaries)
    geo_data.set_branch_data(VBranchGeoData.Fields.CURVATURES, curvatures)
    geo_data.set_branch_data(VBranchGeoData.Fields.CURVATURE_ROOTS, curv_roots)

    if populate_geometry:
        derive_tips_geometry_from_curve_geometry(vgraph, inplace=True)

    return vgraph


def derive_tips_geometry_from_curve_geometry(
    vgraph: VGraph,
    *,
    calibre: None | bool | float = None,
    tangent: Literal["bspline"] | bool | float | None = None,
    boundaries: bool | None = None,
    inplace: bool = False,
) -> VGraph:
    """Derive the geometry of tips from the curve geometry of the branches.

    Parameters
    ----------
    vgraph : VGraph
        The graph to derive the tips geometry from.


    calibre : bool | float, optional
        If neither False nor null, derive ``TERMINATION_CALIBRE`` from ``CALIBRE``.
    inplace : bool, optional
        If True, the graph is modified in place, by default False.

    Returns
    -------
    VGraph
        The graph with the tips geometry set.
    """
    if not inplace:
        vgraph = vgraph.copy()
        return derive_tips_geometry_from_curve_geometry(vgraph, inplace=True)

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

    # === Tangents ===
    if tangent is None:
        if gdata.has_branch_data(VBranchGeoData.Fields.BSPLINE):
            tangent = "bspline"
        elif gdata.has_branch_data(VBranchGeoData.Fields.TANGENTS):
            tangent = True

    if tangent:
        tips_tangents = []
        if tangent == "bspline":
            branches_bsplines = gdata.branch_bspline(name=VBranchGeoData.Fields.BSPLINE)
            for i, branch_bspline in enumerate(branches_bsplines):
                if branch_bspline:
                    tips_tangents.append(np.array(branch_bspline.tips_tangents(normalize=True), dtype=float))
                else:
                    tips_tangents.append(np.zeros((2, 2), dtype=float))
        else:
            if tangent is True:
                tangent = 10
            branches_tangents = gdata.branch_data(VBranchGeoData.Fields.TANGENTS, graph_index=False)
            for branch_tangent in branches_tangents:
                if branch_tangent:
                    d = branch_tangent.data
                    tips_tangents.append(np.stack(d[:tangent].mean(axis=0), -d[-tangent:].mean(axis=0)))
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
