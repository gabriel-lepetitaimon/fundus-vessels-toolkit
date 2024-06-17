from enum import Enum

import numpy as np
import numpy.typing as npt
from skimage.morphology import medial_axis
from skimage.morphology import skeletonize as skimage_skeletonize


class SkeletonizeMethod(str, Enum):
    MEDIAL_AXIS = "medial_axis"
    ZHANG = "zhang"
    LEE = "lee"


def skeletonize(vessel_map: npt.NDArray[np.bool_], method: SkeletonizeMethod = "lee") -> npt.NDArray[np.bool_]:
    """
    Args:
        vessel_map: Binary image containing the vessels.
        method: Method to use for skeletonization. One of: 'medial_axis', 'zhang', 'lee' (default).
    """
    method = SkeletonizeMethod(method)
    if method is SkeletonizeMethod.MEDIAL_AXIS:
        return medial_axis(vessel_map)
    else:
        return skimage_skeletonize(vessel_map, method=method.value)
