from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils.bezier import BSpline
from ..utils.math import quantified_roots
from ..vascular_data_objects import FundusData, VBranchGeoData, VGraph


def parametrize_branches(vgraph: VGraph, fundus_data: Optional[FundusData] = None) -> pd.DataFrame:
    """Extract parameters and biomarkers from the branches of a VGraph.

    The parameters extracted are:
    - mean_calibre: the mean calibre of the branch.
    - std_calibre: the standard deviation of the calibres of the branch.
    - mean_curvature: the mean curvature of the branch.
    - τHart: the Hart tortuosity of the branch.
    - τGrisan: the Grisan tortuosity of the branch.
    - n_curvatures_roots: the number of roots of the branch curvature.

    Parameters
    ----------
    vgraph : VGraph
        The VGraph to analyze.

    fundus_data : Optional[FundusData], optional
        The fundus data associated with the VGraph, by default None.
        If provided, the distance to the optic disc and macula, and the normalized coordinates of the bifurcations
        are computed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parameters of the branches.
    """
    branches_params = []

    gdata = vgraph.geometric_data()
    all_curves = gdata.branch_curve()
    all_calibres = gdata.branch_data(VBranchGeoData.Fields.CALIBRES)
    all_curvatures = gdata.branch_data(VBranchGeoData.Fields.CURVATURES)

    if gdata.has_branch_data(VBranchGeoData.Fields.CURVATURE_ROOTS):
        all_curv_roots = gdata.branch_data(VBranchGeoData.Fields.CURVATURE_ROOTS)
    else:
        all_curv_roots = None
    if gdata.has_branch_data(VBranchGeoData.Fields.BSPLINE):
        all_bsplines = gdata.branch_data(VBranchGeoData.Fields.BSPLINE)
    else:
        all_bsplines = None

    for branch in vgraph.branches():
        if branch.arc_length() < 5:
            continue

        # === Get the calibres and curvature data for this branch ===
        calibres = all_calibres[branch.id]
        curvature = all_curvatures[branch.id]
        if calibres is None or curvature is None:
            continue

        calibres = calibres.data
        curvature = curvature.data
        curve = all_curves[branch.id]
        curv_roots = c.data if all_curv_roots is not None and (c := all_curv_roots[branch.id]) is not None else None

        # === Compute parameters and biomarkers ===
        mean_calibre = np.mean(calibres)
        std_calibre = np.std(calibres)
        mean_curvature = np.mean(curvature)
        std_curvature = np.std(curvature)
        curve_arc = arc(curve)

        length_diameter_ratio = curve_arc / mean_calibre

        params = [
            branch.id,
            curve_arc,
            chord(curve),
            mean_calibre,
            std_calibre,
            mean_curvature,
            std_curvature,
            length_diameter_ratio,
        ]
        tortuosities = branch_tortuosity(curve, curvature, curv_roots, as_dict=False)

        if all_bsplines is not None:
            if bspline := all_bsplines[branch.id]:
                bspline_tortuosities = branch_bspline_tortuosity(bspline.data, as_dict=False)
            else:
                bspline_tortuosities = [np.nan] * len(BSPLINE_TORTUOSITY_KEYS)

        branches_params.append(params + tortuosities + bspline_tortuosities)

    columns = [
        "branch",
        "arc",
        "chord",
        "mean_calibre",
        "std_calibre",
        "mean_curvature",
        "std_curvature",
        "length_diameter_ratio",
    ]
    columns += TORTUOSITY_KEYS
    if all_bsplines is not None:
        columns += BSPLINE_TORTUOSITY_KEYS
    df = pd.DataFrame(branches_params, columns=columns)
    if all_bsplines is not None:
        df.insert(2, "arc_bspline", df.pop("arc_bspline"))

    df.index = df["branch"]
    if "strahler" in vgraph.branch_attr:
        df.insert(1, "strahler", vgraph.branch_attr["strahler"])
    if "rank" in vgraph.branch_attr:
        df.insert(1, "rank", vgraph.branch_attr["rank"])

    if fundus_data is not None:
        mid_yx = np.array(vgraph.geometric_data().branch_midpoint(df.index.to_numpy()))
        macula_center = fundus_data.infered_macula_center()
        if macula_center is not None:
            df.insert(4, "dist_macula", macula_center.distance(mid_yx))
        if fundus_data.od_center is not None:
            from .bifurcations import node_normalized_coordinates

            df.insert(4, "dist_od", fundus_data.od_center.distance(mid_yx))
            if macula_center is not None and fundus_data.od_diameter is not None:
                norm_coord, norm_dist_od = node_normalized_coordinates(
                    mid_yx, fundus_data.od_center, fundus_data.od_diameter, macula_center
                )
                df.insert(4, "norm_coord_x", norm_coord[:, 1])
                df.insert(4, "norm_coord_y", norm_coord[:, 0])
                df.insert(4, "norm_dist_od", norm_dist_od)

    return df


TORTUOSITY_KEYS = ["τHart", "τGrisan", "τTrucco", "n_curvatures_roots"]


def branch_tortuosity(
    curve: npt.NDArray[np.float64],
    curvature: npt.NDArray[np.float64],
    curvature_roots: Optional[npt.NDArray[np.int_]] = None,
    as_dict=True,
) -> Dict[str, float] | List[float]:
    """Compute the tortuosity of a branch.

    Parameters
    ----------
    curve : npt.NDArray[np.float64]
        The skeleton of the branch as a 2D array of shape (n, 2).

    curvature : npt.NDArray[np.float64]
        The curvature of the branch as a 1D array of shape (n,).

    Returns
    -------
    Dict[str, float] | List[float]
        A dictionary or list containing the tortuosity biomarkers.
    """

    if curvature_roots is None:
        curvature_roots = quantified_roots(curvature, 0.15, percentil_threshold=True)

    n = len(curvature_roots)
    Lc = arc(curve)
    Lx = chord(curve)
    Lci = [arc(_) for _ in np.split(curve, curvature_roots) if len(_) > 1]
    Lxi = [chord(_) for _ in np.split(curve, curvature_roots) if len(_) > 1]

    τHart = Lc / Lx - 1
    τGrisan = (n - 1) / Lc * np.sum([lci / lxi - 1 for lci, lxi in zip(Lci, Lxi, strict=True)])

    pTrucco = 4  # cf. Trucco et al. 2010, doi:10.1109/TBME.2010.2050771
    τTrucco = np.sum(abs(curvature) ** pTrucco) ** (1 / pTrucco)

    n_roots = len(curvature_roots)

    values = [τHart, τGrisan, τTrucco, n_roots]
    return dict(zip(TORTUOSITY_KEYS, values, strict=True)) if as_dict else values


BSPLINE_TORTUOSITY_KEYS = ["τHart_bspline", "τGrisan_bspline", "arc_bspline", "n_bspline"]


def branch_bspline_tortuosity(bspline: BSpline, as_dict=True) -> Dict[str, float] | List[float]:
    """Compute the tortuosity of a branch from its B-spline representation.

    Parameters
    ----------
    bspline : BSpline
        The B-spline representation of the branch.

    as_dict : bool, optional
        If True (bu default), the tortuosity is returned as a dictionary, otherwise as a list.

    Returns
    -------
    Dict[str, float] | List[float]
        A dictionary or list containing the tortuosity biomarkers.
    """

    bspline = bspline.ensure_constant_curvature()

    n = len(bspline)
    Lci = [_.arc_length() for _ in bspline]
    Lxi = [_.chord_length() for _ in bspline]
    Lc = sum(Lci)
    Lx = bspline.chord_length()

    τHart = Lc / Lx - 1
    τGrisan = (n - 1) / Lc * np.sum([lci / lxi - 1 for lci, lxi in zip(Lci, Lxi, strict=True)])

    values = [τHart, τGrisan, Lc, n]
    return dict(zip(BSPLINE_TORTUOSITY_KEYS, values, strict=True)) if as_dict else values


def chord(curve: np.ndarray):
    return np.linalg.norm(curve[0] - curve[-1])


def arc(curve: np.ndarray):
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
