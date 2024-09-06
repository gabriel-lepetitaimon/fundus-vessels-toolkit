import numpy as np
import pandas as pd

from ..vascular_data_objects import VBranchGeoData, VGeometricData


def tortuosity_parameters(geodata: VGeometricData):
    branches_tortuosities = []

    curves = geodata.branch_curve()
    ids = [curve is not None and len(curve > 5) for curve in curves]
    curves = [curve for id, (curve, valid) in enumerate(zip(curves, ids, strict=True)) if valid]
    ids = np.argwhere(ids).flatten()

    curvatures = [_.data for _ in geodata.branch_data(VBranchGeoData.Fields.CURVATURES, ids)]

    def chord(curve):
        return np.linalg.norm(curve[0] - curve[-1])

    def arc(curve):
        return np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))

    for id, curve, curvature in zip(ids, curves, curvatures, strict=True):
        curvatures_root = np.argwhere(np.sign(curvature[1:]) != np.sign(curvature[:-1])).flatten()
        n = len(curvatures_root)
        Lc = arc(curve)
        Lx = chord(curve)
        Lci = [arc(_) for _ in np.split(curve, curvatures_root) if len(_) > 1]
        Lxi = [chord(_) for _ in np.split(curve, curvatures_root) if len(_) > 1]

        tortuosities = dict(
            branch_id=id,
            τHart=Lc / Lx - 1,
            n_curvatures_roots=len(curvatures_root),
            τGrisan=(n - 1) / Lc * np.sum([lci / lxi - 1 for lci, lxi in zip(Lci, Lxi)]),
        )
        branches_tortuosities.append(tortuosities)

    return pd.DataFrame(branches_tortuosities).set_index("branch_id")
