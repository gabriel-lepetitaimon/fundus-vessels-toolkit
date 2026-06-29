from typing import Literal

import numpy as np
import numpy.typing as npt
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize as skimage_skeletonize

type SkeletonizeMethod = Literal["medial_axis", "zhang", "lee", "fvt", "fvt_av"]


def skeletonize(vessel_map: npt.NDArray[np.bool_], method: SkeletonizeMethod = "lee") -> npt.NDArray[np.bool_]:
    """
    Args:
        vessel_map: Binary image containing the vessels.
        method: Method to use for skeletonization. One of: 'medial_axis', 'zhang', 'lee' (default).
    """
    if method == "medial_axis":
        return medial_axis(vessel_map)
    elif method == "fvt":
        import torch

        from ..utils.cpp_extensions import fvt_cpp

        return fvt_cpp.skeletonize(torch.from_numpy(vessel_map)).numpy()
    elif method == "fvt_av":
        import torch

        from ..utils.cpp_extensions import fvt_cpp

        return fvt_cpp.skeletonize_av(torch.from_numpy(vessel_map)).numpy()
    else:
        return skimage_skeletonize(vessel_map, method=method)
