from typing import Literal

import numpy as np
import numpy.typing as npt
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize as skimage_skeletonize

type SkeletonizeMethod = Literal["medial_axis", "zhang", "lee", "fvt"]


def skeletonize(
    vessel_map: npt.NDArray[np.bool_] | npt.NDArray[np.uint8], method: SkeletonizeMethod = "lee"
) -> npt.NDArray[np.bool_]:
    """
    Args:
        vessel_map: Binary image containing the vessels.
        method: Method to use for skeletonization. One of: 'medial_axis', 'zhang', 'lee' (default).
    """
    if method == "fvt":
        import torch

        from ..utils.cpp_extensions import fvt_cpp

        if vessel_map.dtype == np.bool_:
            return fvt_cpp.skeletonize(torch.from_numpy(vessel_map)).numpy()
        elif vessel_map.dtype == np.uint8:
            return fvt_cpp.skeletonize_av(torch.from_numpy(vessel_map)).numpy()

    vessel_bin_map = vessel_map.astype(np.bool_)
    if method == "medial_axis":
        return medial_axis(vessel_bin_map)  # type: ignore
    else:
        return skimage_skeletonize(vessel_bin_map, method=method)
