import pandas as pd

from ..utils.graph.measures import extract_bifurcations_parameters as extract_bifurcations_parameters
from ..vascular_data_objects import VBranchGeoData, VGraph


def bifurcations_parameters(
    vgraph: VGraph, calibre=VBranchGeoData.Fields.CALIBRES, tangent=VBranchGeoData.Fields.TANGENTS
) -> dict:
    """
    Extract parameters of bifurcations from a VGraph.

    Parameters
    ----------
    vgraph:
        The VGraph to analyze.


    """
    calibres = [_.data for _ in vgraph.branches_geo_data(calibre)]
    tangents = [_.data for _ in vgraph.branches_geo_data(tangent)]
    df = extract_bifurcations_parameters(calibres, tangents, vgraph.branch_list, directed=False)
    df = df.drop(df.index[(df["d1"] == 0) | (df["d2"] == 0)])
    df["θ_branching"] = df["θ1"] + df["θ2"]
    df["θ_assymetry"] = abs(df["θ1"] - df["θ2"])
    df["assymetry_ratio"] = df["d2"] ** 2 / df["d1"] ** 2
    df["branching_coefficient"] = (df["d1"] + df["d2"]) ** 2 / df["d0"] ** 2

    return df
