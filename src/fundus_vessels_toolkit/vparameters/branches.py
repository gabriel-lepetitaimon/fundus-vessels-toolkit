from typing import Dict

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..vascular_data_objects import VBranchGeoData, VGraph


def parametrize_branches(vgraph: VGraph) -> pd.DataFrame:
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

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parameters of the branches.
    """
    branches_params = []

    for branch in vgraph.branches():
        if branch.arc_length() < 5:
            continue

        # === Get the calibres and curvature data for this branch ===
        calibres = branch.geodata(VBranchGeoData.Fields.CALIBRES)
        curvature = branch.geodata(VBranchGeoData.Fields.CURVATURES)

        if calibres is None or curvature is None:
            continue
        calibres = calibres.data
        curvature = curvature.data

        # === Compute parameters and biomarkers ===
        mean_calibre = np.mean(calibres)
        std_calibre = np.std(calibres)
        mean_curvature = np.mean(curvature)

        tortuosities = compute_tortuosity(branch.curve(), curvature)

        branches_params.append(
            {
                "branch": branch.id,
                "mean_calibre": mean_calibre,
                "std_calibre": std_calibre,
                "mean_curvature": mean_curvature,
                **tortuosities,
            }
        )

    return pd.DataFrame(branches_params)


def compute_tortuosity(curve: npt.NDArray[np.float_], curvature: npt.NDArray[np.float_]) -> Dict[str, float]:
    """Compute the tortuosity of a branch.

    Parameters
    ----------
    curve : npt.NDArray[np.float_]
        The skeleton of the branch as a 2D array of shape (n, 2).

    curvature : npt.NDArray[np.float_]
        The curvature of the branch as a 1D array of shape (n,).

    Returns
    -------
    pd.DataFrame
        _description_
    """

    curvatures_root = np.argwhere(np.sign(curvature[1:]) != np.sign(curvature[:-1])).flatten()
    n = len(curvatures_root)
    Lc = arc(curve)
    Lx = chord(curve)
    Lci = [arc(_) for _ in np.split(curve, curvatures_root) if len(_) > 1]
    Lxi = [chord(_) for _ in np.split(curve, curvatures_root) if len(_) > 1]

    return {
        "τHart": Lc / Lx - 1,
        "τGrisan": (n - 1) / Lc * np.sum([lci / lxi - 1 for lci, lxi in zip(Lci, Lxi, strict=True)]),
        "n_curvatures_roots": len(curvatures_root),
    }


def chord(curve: np.ndarray):
    return np.linalg.norm(curve[0] - curve[-1])


def arc(curve: np.ndarray):
    return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
