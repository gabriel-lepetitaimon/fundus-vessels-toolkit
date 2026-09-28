import warnings
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from fundus_toolkits import FundusData
from fundus_toolkits.utils.geometric import Point
from fundus_toolkits.utils.typing import FloatPair

from ..utils.graph.measures import extract_bifurcations_parameters as extract_bifurcations_parameters
from ..utils.math import intercept_segment_norm_dist, modulo_pi
from ..vascular_data_objects import VBranchGeoData, VGeometricData, VTree


@overload
def bifurcations_biomarkers(d0, d1, d2, θ1, θ2, *, as_dict: Literal[True] = True) -> dict[str, float]: ...
@overload
def bifurcations_biomarkers(d0, d1, d2, θ1, θ2, *, as_dict: Literal[False] = False) -> List[float]: ...
def bifurcations_biomarkers(d0, d1, d2, θ1, θ2, *, as_dict: bool = True) -> Dict[str, float] | List[float]:
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
    area_ratio = d1**2 + d2**2 / d0**2

    # Optimality Ratio: https://link.springer.com/content/pdf/10.1016/j.artres.2010.06.003.pdf
    optimality_ratio = ((d1**3 + d2**3) / 2 * d0**3) ** (1 / 3)
    optimality_dev = abs(optimality_ratio - 1 / 2 ** (1 / 3))

    junctional_exponent_dev = (abs(d0**3 - d1**3 - d2**3) ** (1 / 3)) / d0

    if as_dict:
        return {
            "θ_branching": θ_branching,
            "θ_assymetry": θ_assymetry,
            "assymetry_ratio": assymetry_ratio,
            "branching_coefficient": branching_coefficient,
            "area_ratio": area_ratio,
            "optimality_ratio": optimality_ratio,
            "optimality_dev": optimality_dev,
            "junctional_exponent_dev": junctional_exponent_dev,
        }
    return [
        θ_branching,
        θ_assymetry,
        assymetry_ratio,
        branching_coefficient,
        area_ratio,
        optimality_ratio,
        optimality_dev,
        junctional_exponent_dev,
    ]


BIFURCATIONS_BIOMARKERS: list[str] = list(bifurcations_biomarkers(*((1,) * 5), as_dict=True).keys())


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

    from ..segment_to_graph.geometry_parsing import derive_tips_geometry_from_curve_geometry

    derive_tips_geometry_from_curve_geometry(vtree, tangent=True, calibre=True, inplace=True)

    gdata = vtree.geometric_data()
    nodes_yx = gdata.node_coord()
    bifurcations = []
    bifurcations_yx = []

    if strahler_field is not None and strahler_field not in vtree.branch_attr:
        assign_strahler_number(vtree, field=strahler_field)

    branch_rank = 1
    if branch_rank_field is not None and branch_rank_field not in vtree.branch_attr:
        vtree.branch_attr[branch_rank_field] = 1
        vtree.node_attr[branch_rank_field] = 0

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
            head_tangent = np.diff(gdata.node_coord(list(branch.directed_node_ids)), axis=0)[0]
            head_tangent /= np.linalg.norm(head_tangent)

        # === Get the calibres and tangents data for its successors ===
        successors_data = branch.successors_tip_geodata([calibre_tip, tangent_tip], geodata=gdata)
        tertiary_branches = list(branch.successors())
        tertiary_calibres = list(successors_data[calibre_tip])
        tertiary_tangents = list(successors_data[tangent_tip])

        for i, ter_branch in enumerate(tertiary_branches):
            if np.isnan(tertiary_tangents[i]).any() or np.sum(tertiary_tangents[i]) == 0:
                # If the tangent is not available, fallback to the difference of nodes coordinates
                tertiary_tangents[i] = np.diff(gdata.node_coord(list(ter_branch.directed_node_ids)), axis=0)[0]
                if (tertiary_tangents[i] != 0).any():
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
        if len(bifurcations_yx) == 0 or all(len(_) == 0 for _ in bifurcations_yx):
            warnings.warn(
                "No bifurcations was found in the provided VTree"
                + (f" (for image: {fundus_data.name})" if fundus_data.has_name else "")
                + "."
            )
            return df

        bifurcations_yx = np.stack(bifurcations_yx)
        macula_center = fundus_data.inferred_macula_center()
        if macula_center is not None:
            df.insert(4, "dist_macula", macula_center.distance(bifurcations_yx))
        if fundus_data.od_center is not None:
            df.insert(4, "dist_od", fundus_data.od_center.distance(bifurcations_yx))
            if macula_center is not None and fundus_data.has_od_diameter:
                norm_coord, norm_dist_od = node_normalized_coordinates(
                    bifurcations_yx, fundus_data.od_center, fundus_data.od_diameter, macula_center
                )
                df.insert(4, "norm_coord_x", norm_coord[:, 1])
                df.insert(4, "norm_coord_y", norm_coord[:, 0])
                df.insert(4, "norm_dist_od", norm_dist_od)
                df.insert(4, "x", bifurcations_yx[:, 1])
                df.insert(4, "y", bifurcations_yx[:, 0])
        df.insert(4, "dist_center", (Point(*fundus_data.shape) / 2).distance(bifurcations_yx))

    return df


def split_complex_bifurcations(
    vtree: VTree,
    topo_field: Optional[str] = None,
    *,
    geodata: Optional[VGeometricData | int] = None,
    calibre_tip=VBranchGeoData.Fields.TIPS_CALIBRE,
    tangent_tip=VBranchGeoData.Fields.TIPS_TANGENT,
    inplace: bool = False,
) -> VTree:
    """Split complex bifurcations (with more than 2 successors) into simple bifurcations and reorder branches index to ensure that the main branch has the lowest index.

    Returns
    -------
    VTree
        The binary tree.
    """  # noqa: E501
    from ..segment_to_graph.geometry_parsing import derive_tips_geometry_from_curve_geometry
    from ..segment_to_graph.tree_simplification import disconnect_crossing
    from ..segment_to_graph.tree_topology import TopologicalLabel

    if not inplace:
        vtree = vtree.copy()
    vtree.flip_branch_to_tree_dir(inplace=True)

    disconnect_crossing(vtree, inplace=True, fuse_passing_nodes=True)

    derive_tips_geometry_from_curve_geometry(vtree, tangent=True, calibre=True, inplace=True)
    gdata = vtree.geometric_data(geodata)
    topoLabels: dict[int, TopologicalLabel] = {}

    def analyze_branch_bifurcation(branch_id: int):
        branch = vtree.branch(branch_id)
        topo: TopologicalLabel = TopologicalLabel(topoLabels[branch_id])

        # === If 0 or 1 successor don't reorder, ... ===
        if branch.n_successors == 1:
            topoLabels[branch.successors_ids[0]] = topo
            analyze_branch_bifurcation(branch.successors_ids[0])
        if branch.n_successors <= 1:
            return

        # === ..., otherwise, get the calibres and tangents data for this branch head ===
        head_tangent = -branch.head_tip_geodata(tangent_tip, geodata=gdata)

        if np.isnan(head_tangent).any() or np.sum(head_tangent) == 0:
            head_tangent = np.diff(gdata.node_coord(branch.directed_node_ids), axis=0)[0]
            head_tangent /= np.linalg.norm(head_tangent)

        # === Get the calibres and tangents data for its successor tails ===
        class Successor(NamedTuple):
            branch_id: int
            calibre: npt.NDArray[np.int_]
            tangent: npt.NDArray[np.float64]
            main_succ_rank: int = 0

            @property
            def branch(self) -> VTree.Branch:
                return vtree.branch(self.branch_id)

        successors_calibres = branch.successors_tip_geodata(calibre_tip, geodata=gdata)
        successors_tangents = branch.successors_tip_geodata(tangent_tip, geodata=gdata)
        successors = [
            Successor(branch_id=b, calibre=c, tangent=t)
            for b, c, t in zip(branch.successors_ids, successors_calibres, successors_tangents, strict=True)
        ]

        for succ_id in successors:
            if np.isnan(succ_id.tangent).any() or np.sum(succ_id.tangent) == 0:
                # If the tangent is not available, fallback to the difference of nodes coordinates
                succ_id.tangent[:] = np.diff(gdata.node_coord(succ_id.branch.directed_node_ids), axis=0)[0]
                if (succ_id.tangent != 0).any():
                    succ_id.tangent[:] /= np.linalg.norm(succ_id.tangent)

        sorted_successors: list[Successor] = []
        while successors:
            if not np.isnan(successors_calibres).any():
                # If the calibres are available, select the one with the highest calibre
                main_succ_id = np.argmax([_.calibre for _ in successors])
                main_succ = successors[main_succ_id]
                # Check that it is at least 1.4px larger than any other tertiary branches
                if all(main_succ.calibre - 1.4 > _.calibre for _ in successors if _ is not main_succ):
                    sorted_successors.append(successors.pop(main_succ_id))
                    continue

            # If calibres are not available or not decisive, select the branch with the closest angle
            main_succ_id = np.argmax([np.dot(head_tangent, _.tangent) for _ in successors])
            sorted_successors.append(successors.pop(main_succ_id))

        # === Sort the successors ===
        def recursive_resolve_bifurcation(
            successors: list[Successor],
            parent_topo: TopologicalLabel,
            parent_id: int,
        ):
            if len(successors) < 2:
                return
            main1, main2 = successors.pop(0), successors.pop(0)
            tail1, tail2 = main1.branch.tail_tip_coord().numpy(), main2.branch.tail_tip_coord().numpy()
            parent_head = vtree.branch(parent_id).head_tip_coord().numpy()
            main2_tan = main2.tangent
            u = np.clip(intercept_segment_norm_dist(parent_head, tail1, tail2, tail2 - main2_tan)[0, 0, 0], 0.1, 0.75)
            bifurcation12 = parent_head + u * (tail1 - parent_head)

            succ_pre: list[tuple[Successor, FloatPair]] = []  # Successors bifurcating before main1 and main2
            succ_main1, succ_main2 = [], []  # Successors bifurcating off from main1 or main2 respectively

            if successors:
                # Compute intercepts of the other successors and segments to decide their position in the tree.
                # If the nearest intercept points is on the segment:
                #   - [parent_head, bifurcation12] -> bifurcating before main1 and main2, off from main1
                #   - [head, tail1] -> bifurcating after main1 and main2, off from main1
                #   - [head, tail2] -> bifurcation after main1 and main2, off from main2
                for succ in successors:
                    tail = succ.branch.tail_tip_coord().numpy()
                    tan = succ.tangent
                    u0 = intercept_segment_norm_dist(parent_head, bifurcation12, tail, tail - tan)[0, 0, 0]
                    if u0 <= 1:
                        succ_pre.append((succ, parent_head + u0 * (bifurcation12 - parent_head)))
                    else:
                        u1, d1 = intercept_segment_norm_dist(bifurcation12, tail1, tail, tail - tan)[0, 0]
                        u2, d2 = intercept_segment_norm_dist(bifurcation12, tail2, tail, tail - tan)[0, 0]
                        valid_u1, valid_u2 = 0 <= u1 <= 1, 0 <= u2 <= 1
                        if valid_u1 != valid_u2:  # If only one of the two is valid, assign to the valid one
                            (succ_main1 if valid_u1 else succ_main2).append(succ)
                        elif valid_u1:  # If both are valid, assign to the closest one
                            (succ_main1 if d1 < d2 else succ_main2).append(succ)
                        else:  # If none are valid, assign to the closest segment
                            head_tail1 = bifurcation12 - tail1 / (np.linalg.norm(bifurcation12 - tail1) + 1e-8)
                            head_tail2 = bifurcation12 - tail2 / (np.linalg.norm(bifurcation12 - tail2) + 1e-8)
                            d1 = np.linalg.norm(np.cross(head_tail1, bifurcation12 - tail))
                            d2 = np.linalg.norm(np.cross(head_tail2, bifurcation12 - tail))
                            (succ_main1 if d1 <= d2 else succ_main2).append(succ)

            # If there are successors bifurcating before main1 and main2, process them first, one after the other
            for succ, coord in succ_pre:
                succ_id = succ.branch_id
                # Create a new node at the bifurcation point and connect the parent and successor branches to it
                new_node = vtree.add_nodes([np.round(coord)])[0]
                vtree._branch_list[[succ_id, parent_id], [0, 1]] = new_node  # Successor tail, Parent head
                # Create a new branch from the bifurcation to main1 (and main2) tail
                new_branch_id = vtree.add_branch([(new_node, main1.branch.tail_id)], auto_connect=False)[0]
                vtree._branch_tree[new_branch_id] = parent_id  # Parent of the new branch is the current parent
                # Update the current parent id and topological label to the new branch for the next successors
                parent_id = new_branch_id
                parent_topo, succ_topo = parent_topo.children
                # Save the topological labels
                topoLabels[new_branch_id] = parent_topo
                topoLabels[succ.branch_id] = succ_topo

            # Move parent_id head node to the bifurcation point
            gdata._node_coord[vtree.branch_list[parent_id, 1]] = np.round(bifurcation12)

            # Ensure that main1 and main2 points to their appropriate parent
            # (which may have changed if any bifurcations were created before them)
            vtree._branch_tree[[main1.branch_id, main2.branch_id]] = parent_id
            vtree.branch_list[[main1.branch_id, main2.branch_id], 0] = vtree.branch_list[parent_id, 1]
            main1_topo, main2_topo = parent_topo.children

            def create_bifurcation(successors: list[Successor], label: TopologicalLabel) -> int:
                # Create a new node and connect the current parent branch to it
                new_node = vtree.add_nodes([bifurcation12])[0]  # It will be move later to the appropriate bifurcation
                vtree._branch_list[[_.branch_id for _ in successors], 0] = new_node
                # Create a new branch from the bifurcation to the new node
                new_branch_id = vtree.add_branch([(vtree._branch_list[parent_id, 1], new_node)], auto_connect=False)[0]
                vtree._branch_tree[new_branch_id] = parent_id  # Parent of the new branch is the current parent
                topoLabels[new_branch_id] = label
                # Set it as the parent of the successors
                vtree._branch_tree[[_.branch_id for _ in successors]] = new_branch_id
                return new_branch_id

            if succ_main1:
                main1_and_succ = [main1] + succ_main1
                main1_parent_id = create_bifurcation(main1_and_succ, main1_topo)
                recursive_resolve_bifurcation(main1_and_succ, main1_topo, main1_parent_id)
            else:
                topoLabels[main1.branch_id] = main1_topo
                analyze_branch_bifurcation(main1.branch_id)
            for succ, _ in succ_pre:
                analyze_branch_bifurcation(succ.branch_id)
            if succ_main2:
                main2_and_succ = [main2] + succ_main2
                main2_parent_id = create_bifurcation(main2_and_succ, main2_topo)
                recursive_resolve_bifurcation(main2_and_succ, main2_topo, main2_parent_id)
            else:
                topoLabels[main2.branch_id] = main2_topo
                analyze_branch_bifurcation(main2.branch_id)

        branch_id = branch.id
        del branch  # Deallocate reference to branch before recursive_resolve_bifurcation change the graph structure
        recursive_resolve_bifurcation(sorted_successors, topo, branch_id)

    for subtree, i in enumerate(vtree.root_branch_ids()):
        topoLabels[i] = TopologicalLabel.encode(subtree, [])
        analyze_branch_bifurcation(i)

    if topo_field is not None:
        _topoLabels = np.zeros(vtree.branch_count, dtype=np.uint64)
        for branch_id, label in topoLabels.items():
            _topoLabels[branch_id] = label
        vtree.branch_attr[topo_field] = _topoLabels

    return vtree


def reorder_branch_by_bifurcations(
    vtree: VTree,
    rank_field: Optional[str] = None,
    *,
    geodata: Optional[VGeometricData | int] = None,
    calibre_tip=VBranchGeoData.Fields.TIPS_CALIBRE,
    tangent_tip=VBranchGeoData.Fields.TIPS_TANGENT,
    apply_branch_reordering: bool = True,
) -> npt.NDArray[np.int_]:
    """Sort the branch IDs by bifurcations.
    This sort ensure that, in a bifurcation, the indexes of the secondary branches is always higher than the main branches.

    Returns
    -------
    VTree
        The binary tree.
    """  # noqa: E501
    from ..segment_to_graph.geometry_parsing import derive_tips_geometry_from_curve_geometry

    derive_tips_geometry_from_curve_geometry(vtree, tangent=True, calibre=True, inplace=True)
    gdata = vtree.geometric_data(geodata)
    branch_lookup = np.arange(vtree.branch_count, dtype=np.int_)

    if rank_field is not None and rank_field not in vtree.branch_attr:
        vtree.branch_attr[rank_field] = 1
        vtree.node_attr[rank_field] = 0

    def recursive_reorder_successors(branch_id: int, rank: int):
        branch = vtree.branch(branch_id)

        # === Assign branch and node rank ===
        if rank_field is not None:
            branch.attr[rank_field] = rank
            branch.head_node().attr[rank_field] = rank

        # === If 0 or 1 successor don't reorder, ... ===
        if branch.n_successors == 1:
            recursive_reorder_successors(branch.successors_ids[0], rank)
        if branch.n_successors <= 1:
            return

        # === ..., otherwise, get the calibres and tangents data for this branch head ===
        head_tangent = -branch.head_tip_geodata(tangent_tip, geodata=gdata)

        if np.isnan(head_tangent).any() or np.sum(head_tangent) == 0:
            head_tangent = np.diff(gdata.node_coord(branch.directed_node_ids), axis=0)[0]
            head_tangent /= np.linalg.norm(head_tangent)

        # === Get the calibres and tangents data for its successor tails ===
        class Successors(NamedTuple):
            branch: VTree.Branch
            calibre: npt.NDArray[np.int_]
            tangent: npt.NDArray[np.int_]

        successors_calibres = branch.successors_tip_geodata(calibre_tip, geodata=gdata)
        successors_tangents = branch.successors_tip_geodata(tangent_tip, geodata=gdata)
        successors = [
            Successors(branch=b, calibre=c, tangent=t)
            for b, c, t in zip(branch.successors(), successors_calibres, successors_tangents, strict=True)
        ]

        for succ_id in successors:
            if np.isnan(succ_id.tangent).any() or np.sum(succ_id.tangent) == 0:
                # If the tangent is not available, fallback to the difference of nodes coordinates
                succ_id.tangent[:] = np.diff(gdata.node_coord(succ_id.branch.directed_node_ids), axis=0)[0]
                if (succ_id.tangent != 0).any():
                    succ_id.tangent[:] /= np.linalg.norm(succ_id.tangent)

        # === Sort the successors ===
        sorted_successors = []
        while successors:
            if not np.isnan(successors_calibres).any():
                # If the calibres are available, select the one with the highest calibre
                main_succ_id = np.argmax([_.calibre for _ in successors])
                main_succ = successors[main_succ_id]
                # Check that it is at least 1.5px larger than any other tertiary branches
                if all(main_succ.calibre - 1.5 > _.calibre for _ in successors if _ is not main_succ):
                    sorted_successors.append(successors.pop(main_succ_id))
                    continue

            # If calibres are not available or not decisive, select the branch with the closest angle
            main_succ_id = np.argmax([np.dot(head_tangent, _.tangent) for _ in successors])
            sorted_successors.append(successors.pop(main_succ_id))

        # === Reindex branches ===
        old_succ_ids = np.array([succ.branch.id for succ in sorted_successors], dtype=np.int_)
        new_succ_ids = np.sort(old_succ_ids)
        branch_lookup[old_succ_ids] = new_succ_ids
        # Deallocate references to branches before reindexing
        del sorted_successors
        del branch

        # === Process successors recursively ===
        recursive_reorder_successors(new_succ_ids[0], rank)
        for succ_id in new_succ_ids[1:]:
            recursive_reorder_successors(succ_id, rank + 1)

    for i in vtree.root_branch_ids():
        recursive_reorder_successors(i, 0)

    if apply_branch_reordering:
        vtree.reindex_branches(branch_lookup, inplace=True)

    return branch_lookup


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
    vtree.node_attr[field] = 1
    vtree.branch_attr[field] = 1
    reverse_depth_order = np.array(list(vtree.walk_branch_ids(traversal="dfs")), dtype=int)[::-1]
    for branch in vtree.branches(reverse_depth_order):
        if branch.has_successors:
            strahlers = sorted([s.attr[field] for s in branch.successors()], reverse=True)
            if len(strahlers) == 1 or strahlers[0] != strahlers[-1]:
                branch.attr[field] = strahlers[0]
            else:
                branch.attr[field] = strahlers[0] + 1
        branch.tail_node().attr[field] = branch.attr[field]

    return vtree


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
