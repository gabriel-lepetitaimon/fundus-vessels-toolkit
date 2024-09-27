import numpy as np
import pandas as pd

from ..utils.graph.measures import extract_bifurcations_parameters as extract_bifurcations_parameters
from ..utils.math import modulo_pi
from ..vascular_data_objects import VBranchGeoData, VTree


def bifurcations_biomarkers(d0, d1, d2, θ1, θ2) -> dict:
    """
    Compute bifurcation biomarkers from the calibres and angles of its branches.

    Parameters
    ----------
    vgraph:
        The VGraph to analyze.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parameters of the bifurcations.
    """
    d = {}
    d["θ_branching"] = θ1 + θ2
    d["θ_assymetry"] = θ1 - θ2
    d["assymetry_ratio"] = d2**2 / d1**2
    d["branching_coefficient"] = (d1 + d2) ** 2 / d0**2
    return d


def parametrize_bifurcations(
    vtree: VTree, *, calibre_tip=VBranchGeoData.Fields.TIPS_CALIBRE, tangent_tip=VBranchGeoData.Fields.TIPS_TANGENT
) -> pd.DataFrame:
    """Extract parameters and biomarkers from the bifurcations of a VTree.

    The parameters extracted are:
    - node: the node id of the bifurcation.
    - branch0: the id of the parent branch.
    - branch1: the id of the secondary branch.
    - branch2: the id of the tertiary branch.
    - d0: the calibre of the parent branch.
    - d1: the calibre of the secondary branch.
    - d2: the calibre of the tertiary branch.
    - θ1: the angle between the parent and secondary branch.
    - θ2: the angle between the parent and tertiary branch.
    - θ_branching: the sum of the angles
    - θ_assymetry: the difference of the angles
    - assymetry_ratio
    - branching_coefficient

    Parameters
    ----------
    vtree: VTree
        The VTree to analyze.

    calibre: str
        The field of the VBranchGeoData containing the calibres. Use the standard 'TIPS_CALIBRE' field by default.

    tangent: str
        The field of the VBranchGeoData containing the tangents. Use the standard 'TIPS_TANGENT' field by default.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parameters of the bifurcations
    """

    bifurcations = []

    for branch in vtree.branches():
        if branch.n_successors < 2:
            continue

        # === Get the calibres and tangents data for this branch ===
        head_data = branch.head_tip_geodata([calibre_tip, tangent_tip])
        head_tangent = head_data.get(tangent_tip, np.zeros(2, dtype=float))
        head_calibre = head_data.get(calibre_tip, np.nan)

        if np.isnan(head_tangent).any() or np.sum(head_tangent) == 0 or np.isnan(head_calibre):
            continue

        # === Get the calibres and tangents data for this branch ===
        successors_data = branch.successors_tip_geodata([calibre_tip, tangent_tip])
        tertiary_branches = list(branch.successors())
        tertiary_tangents = list(successors_data[tangent_tip])
        tertiary_calibres = list(successors_data[calibre_tip])

        # === Find the largest successor branch ===
        second_branch_i = 0
        secondary_calibre = 0
        i = 0
        while i < len(tertiary_branches):
            calibre = tertiary_calibres[i]
            if np.isnan(tertiary_tangents[i]).any() or np.sum(tertiary_tangents[i]) == 0 or np.isnan(calibre):
                # Ignore branches with invalid tip data
                del tertiary_branches[i]
                del tertiary_tangents[i]
                del tertiary_calibres[i]
            else:
                # Search for the largest calibre
                if secondary_calibre < calibre:
                    secondary_calibre = calibre
                    second_branch_i = i
                i += 1

        # === Check if there is still enough successors ===
        if len(tertiary_branches) < 2:
            continue

        # === Prepare the secondary and tertiary branches data ===
        secondary_branch = tertiary_branches.pop(second_branch_i)
        secondary_tangent = tertiary_tangents.pop(second_branch_i)
        secondary_calibre = tertiary_calibres.pop(second_branch_i)

        # === Compute secondary branch parameters ===
        d0 = head_calibre
        d1 = secondary_calibre
        α0 = np.arctan2(*(-head_tangent))
        α1 = np.arctan2(*secondary_tangent)
        θ1 = np.rad2deg(modulo_pi(α1 - α0))

        # === For each bifurcation at this node compute parameters ===
        for tertiary_branch, tertiary_tangent, tertiary_calibre in zip(
            tertiary_branches, tertiary_tangents, tertiary_calibres, strict=True
        ):
            α2 = np.arctan2(*tertiary_tangent)
            d2 = tertiary_calibre
            θ2 = np.rad2deg(modulo_pi(α2 - α0))
            if abs(θ1) > abs(θ2):
                thetas = (θ1, -θ2) if θ1 > 0 else (-θ1, θ2)
            else:
                thetas = (-θ1, θ2) if θ2 > 0 else (θ1, -θ2)
            bifurcations.append(
                (branch.head_id, branch.id, secondary_branch.id, tertiary_branch.id, d0, d1, d2, *thetas)
            )

    # === Compute the bifurcation parameters ===
    parameters = pd.DataFrame(
        bifurcations, columns=["node", "branch0", "branch1", "branch2", "d0", "d1", "d2", "θ1", "θ2"]
    )
    biomarkers = parameters.apply(
        lambda x: bifurcations_biomarkers(x["d0"], x["d1"], x["d2"], x["θ1"], x["θ2"]), axis=1
    )
    return pd.concat([parameters, pd.DataFrame(biomarkers.tolist(), index=parameters.index)], axis=1)
