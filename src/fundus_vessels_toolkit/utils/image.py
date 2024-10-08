from typing import Tuple

import numpy as np
import numpy.typing as npt


def crop_pad_center(img: npt.NDArray, shape: Tuple[int, ...]) -> npt.NDArray:
    H, W = img.shape[:2]
    h, w = shape[:2]

    out = np.zeros(shape=shape + img.shape[2:], dtype=img.dtype)

    if H < h:
        x0 = (h - H) // 2
        x1 = 0
        h0 = H
    else:
        x0 = 0
        x1 = (H - h) // 2
        h0 = h

    if W < w:
        y0 = (w - W) // 2
        y1 = 0
        w0 = W
    else:
        y0 = 0
        y1 = (W - w) // 2
        w0 = w

    out[x0 : x0 + h0, y0 : y0 + w0] = img[x1 : x1 + h0, y1 : y1 + w0]
    return out
