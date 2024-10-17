from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from fundus_vessels_toolkit.utils.geometric import Point

from ..utils.graph.measures import extract_bifurcations_parameters as extract_bifurcations_parameters
from ..utils.math import modulo_pi
from ..vascular_data_objects import FundusData, VBranchGeoData, VTree


def bifurcations_biomarkers(d0, d1, d2, θ1, θ2, *, as_dict=True) -> Dict[str, float] | List[float]:
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
    θ_branching = θ1 + θ2
    θ_assymetry = abs(θ1 - θ2)
    assymetry_ratio = d2**2 / d1**2
    branching_coefficient = (d1 + d2) ** 2 / d0**2

    if as_dict:
        return {
            "θ_branching": θ_branching,
            "θ_assymetry": θ_assymetry,
            "assymetry_ratio": assymetry_ratio,
            "branching_coefficient": branching_coefficient,
        }
    return [θ_branching, θ_assymetry, assymetry_ratio, branching_coefficient]


def parametrize_bifurcations(
    vtree: VTree,
    *,
    calibre_tip=VBranchGeoData.Fields.TIPS_CALIBRE,
    tangent_tip=VBranchGeoData.Fields.TIPS_TANGENT,
    strahler_field: Optional[str] = "strahler",
    branch_rank_field: Optional[str] = "rank",
    fundus_data: Optional[FundusData] = None,
) -> pd.DataFrame:
    """Extract parameters and biomarkers from the bifurcations of a VTree.

    The parameters extracted are:
    - node: the node id of the bifurcation.
    - branch0: the id of the parent branch.
    - branch1: the id of the secondary branch.
    - branch2: the id of the tertiary branch.
    - (strahler): the Strahler number of the parent branch.
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

    strahler_field: str
        The branch attribute containing the Strahler numbers (if the attribute is empty, computes it).

        If None, the Strahler numbers are not returned.

    branch_rank_field: str
        The branch attribute containing the rank of the branches (if the attribute is empty, computes it).

        If None, the ranks are not returned.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parameters of the bifurcations
    """  # noqa: E501

    gdata = vtree.geometric_data()
    nodes_yx = gdata.nodes_coord()
    bifurcations = []
    bifurcations_yx = []

    if strahler_field is not None and strahler_field not in vtree.branches_attr:
        assign_strahler_number(vtree, field=strahler_field)

    if branch_rank_field is not None and branch_rank_field not in vtree.branches_attr:
        vtree.branches_attr[branch_rank_field] = 1
        vtree.nodes_attr[branch_rank_field] = 0

    for branch in vtree.walk_branches():
        if branch_rank_field is not None:
            branch_rank = branch.attr[branch_rank_field]
            branch.head_node().attr[branch_rank_field] = branch_rank

        n_successors = branch.n_successors
        if branch.n_successors < 2:
            if branch_rank_field is not None and n_successors == 1:
                branch.successor(0).attr[branch_rank_field] = branch_rank
            continue

        # === Get the calibres and tangents data for this branch ===
        head_data = branch.head_tip_geodata([calibre_tip, tangent_tip], geodata=gdata)
        head_tangent = -head_data.get(tangent_tip, np.zeros(2, dtype=float))
        head_calibre = head_data.get(calibre_tip, np.nan)

        if np.isnan(head_tangent).any() or np.sum(head_tangent) == 0:
            head_tangent = np.diff(gdata.nodes_coord(list(branch.directed_nodes_id)), axis=0)[0]
            head_tangent /= np.linalg.norm(head_tangent)

        # === Get the calibres and tangents data for its successors ===
        successors_data = branch.successors_tip_geodata([calibre_tip, tangent_tip], geodata=gdata)
        tertiary_branches = list(branch.successors())
        tertiary_calibres = list(successors_data[calibre_tip])
        tertiary_tangents = list(successors_data[tangent_tip])

        for i, ter_branch in enumerate(tertiary_branches):
            if np.isnan(tertiary_tangents[i]).any() or np.sum(tertiary_tangents[i]) == 0:
                # If the tangent is not available, fallback to the difference of nodes coordinates
                tertiary_tangents[i] = np.diff(gdata.nodes_coord(list(ter_branch.directed_nodes_id)), axis=0)[0]
                tertiary_tangents[i] /= np.linalg.norm(tertiary_tangents[i])

        # === Select the secondary branch (main successor) ===
        second_branch_i = 0
        if not np.any(np.isnan(tertiary_calibres)):
            # If the calibres are available, select the one with the highest calibre
            second_branch_i = np.argmax(tertiary_calibres)
            # Check that it is at least 1.5px larger than any other tertiary branches
            if not np.all(tertiary_calibres[second_branch_i] - 1.5 > np.delete(tertiary_calibres, second_branch_i)):
                # If not, select the one with the closest angle to the parent branch
                second_branch_i = np.argmax([np.dot(head_tangent, t) for t in tertiary_tangents])

        else:
            # If the calibres are not available, select the one with the closest angle to the parent branch
            second_branch_i = np.argmax([np.dot(head_tangent, t) for t in tertiary_tangents])

        # === Prepare the secondary and tertiary branches data ===
        secondary_branch = tertiary_branches.pop(second_branch_i)
        secondary_tangent = tertiary_tangents.pop(second_branch_i)
        secondary_calibre = tertiary_calibres.pop(second_branch_i)

        # === Assign branch ranks to successors ===
        secondary_branch.attr[branch_rank_field] = (
            branch_rank
            if np.isnan([secondary_calibre, head_calibre]).any() or secondary_calibre > head_calibre * 0.75
            else branch_rank + 1
        )
        for ter_branch in tertiary_branches:
            ter_branch.attr[branch_rank_field] = branch_rank + 1

        # === Compute secondary branch parameters ===
        d0 = head_calibre
        d1 = secondary_calibre
        α0 = np.arctan2(*head_tangent)
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
            infos = [branch.head_id, branch.id, secondary_branch.id, tertiary_branch.id]
            if strahler_field is not None:
                infos.append(branch.attr[strahler_field])
            if branch_rank_field is not None:
                infos.append(branch_rank)
            params = [d0, d1, d2, *thetas]
            bifurcations_yx.append(nodes_yx[branch.head_id])
            biomarkers = bifurcations_biomarkers(*params, as_dict=False)
            bifurcations.append(infos + params + biomarkers)

    # === Compute the bifurcation parameters ===
    columns = ["node", "branch0", "branch1", "branch2"]
    if strahler_field is not None:
        columns.append(strahler_field)
    if branch_rank_field is not None:
        columns.append(branch_rank_field)
    columns += ["d0", "d1", "d2", "θ1", "θ2"]
    columns.extend(bifurcations_biomarkers(*((1,) * 5), as_dict=True).keys())
    df = pd.DataFrame(bifurcations, columns=columns)

    if fundus_data is not None:
        bifurcations_yx = np.stack(bifurcations_yx)
        if fundus_data.has_macula:
            df.insert(4, "dist_macula", fundus_data.macula_center.distance(bifurcations_yx))
        if fundus_data.has_od:
            df.insert(4, "dist_od", fundus_data.od_center.distance(bifurcations_yx))
            macula_center = fundus_data.macula_center if fundus_data.has_macula else fundus_data.infered_macula_center()
            norm_coord, norm_dist_od = node_normalized_coordinates(
                bifurcations_yx, fundus_data.od_center, fundus_data.od_diameter, macula_center
            )
            df.insert(4, "norm_coord_x", norm_coord[:, 1])
            df.insert(4, "norm_coord_y", norm_coord[:, 0])
            df.insert(4, "norm_dist_od", norm_dist_od)

    return df


def assign_strahler_number(vtree: VTree, field: str = "strahler") -> VTree:
    """
    Assign Strahler numbers to the branches of a VTree.

    Parameters
    ----------
    vtree:
        The VTree to analyze.

    field:
        The field of the VBranchGeoData to store the Strahler numbers.

    Returns
    -------
    VTree
        The VTree with the Strahler numbers assigned.
    """
    vtree.nodes_attr[field] = 1
    vtree.branches_attr[field] = 1
    reverse_depth_order = np.array(list(vtree.walk(depth_first=True)), dtype=int)[::-1]
    for branch in vtree.branches(reverse_depth_order):
        if branch.has_successors:
            strahlers = sorted([s.attr[field] for s in branch.successors()], reverse=True)
            if len(strahlers) == 1 or strahlers[0] != strahlers[-1]:
                branch.attr[field] = strahlers[0]
            else:
                branch.attr[field] = strahlers[0] + 1
        branch.tail_node().attr[field] = branch.attr[field]


def node_normalized_coordinates(
    yx: npt.NDArray[np.float64], od_center: Point, od_diameter: float, macula_center: Point
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the normalized coordinates of the nodes of a VTree.

    Parameters
    ----------
    yx: npt.NDArray[np.float64]
        The coordinates of the nodes as a 2D array of shape (n, 2).

    od_center: Point
        The center of the optic disc.

    od_diameter: float
        The diameter of the optic disc.

    macula_center: Point
        The center of the macula.

    Returns
    -------
    normalized_coord: npt.NDArray[np.float64]
        The normalized coordinates of the nodes as a 2D array of shape (n, 2).

    normalized_od_dist: npt.NDArray[np.float64]
        The normalized distance of the nodes to the optic disc as a 1D array of shape (n,).
    """
    normalized_od_dist = od_center.distance(yx) / od_diameter - 0.5
    centered_coord = yx - od_center.numpy()[None, :]
    normalized_coord = centered_coord / od_center.distance(macula_center)
    u = (macula_center - od_center).normalized()
    y, x = normalized_coord.T
    rotated_coord = np.stack([y * u.x + x * u.y, x * u.x - y * u.y], axis=1)
    if od_center.x < macula_center.x:
        rotated_coord[:, 0] = -rotated_coord[:, 0]

    return rotated_coord, normalized_od_dist
