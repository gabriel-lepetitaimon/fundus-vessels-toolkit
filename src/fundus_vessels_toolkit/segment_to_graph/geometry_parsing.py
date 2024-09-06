import numpy as np

from ..utils.graph.measures import extract_branch_geometry
from ..vascular_data_objects import VBranchGeoData, VGraph


def populate_geometry(vgraph: VGraph, vessel_segmentation: np.ndarray, *, bspline_target_error: float = 3) -> VGraph:
    geo_data = vgraph.geometric_data()
    tangents, calibres, boundaries, curvatures, bsplines = extract_branch_geometry(
        geo_data.branch_curve(),
        vessel_segmentation,
        bspline_target_error=bspline_target_error,
        return_boundaries=True,
        return_curvature=True,
        return_calibre=True,
        extract_bspline=True,
    )
    geo_data.set_branch_data(VBranchGeoData.Fields.TANGENTS, tangents)
    geo_data.set_branch_bspline(bsplines)
    geo_data.set_branch_data(VBranchGeoData.Fields.CALIBRES, calibres)
    geo_data.set_branch_data(VBranchGeoData.Fields.BOUNDARIES, boundaries)
    geo_data.set_branch_data(VBranchGeoData.Fields.CURVATURES, curvatures)

    return vgraph
