import numpy as np

from ..utils.graph.measures import extract_branch_geometry
from ..vascular_data_objects import VGraph


def populate_geometry(vgraph: VGraph, vessel_segmentation: np.ndarray, *, bspline_target_error: float = 3) -> VGraph:
    geo_data = vgraph.geometric_data()
    tangents, calibres, bsplines = extract_branch_geometry(
        geo_data.branch_curve(),
        vessel_segmentation,
        bspline_target_error=bspline_target_error,
        return_curvature=False,
        return_calibre=True,
    )
    geo_data.set_branch_data("tangents", tangents)
    geo_data.set_branch_bspline("bspline", bsplines)
    geo_data.set_branch_data("calibre", calibres)

    return vgraph
