from ast import List
from enum import auto
from typing import Literal

import numpy as np
import torch

from .cpp_extensions import fvt_cpp
from .torch import autocast_torch


@autocast_torch
def first_index_of(array, search_for=True, out=None):
    array = array.cpu().int()
    if scalar := np.isscalar(search_for):
        search_for = torch.tensor(search_for, dtype=torch.int32, device=array.device).unsqueeze(0)
    search_for = search_for.cpu().int()
    if out is None:
        out = torch.empty((search_for.shape[0],), dtype=torch.int32)

    fvt_cpp.first_index_of(array, search_for, out)

    return out if not scalar else out[0]


@autocast_torch
def first_two_index_of(array, search_for, out=None):
    array = array.cpu().int()
    search_for = search_for.cpu().int()
    if out is None:
        out = torch.empty((search_for.shape[0], 2), dtype=torch.int32)

    fvt_cpp.first_two_index_of(array, search_for, out)

    return out


@autocast_torch
def discontiguous_index(curve) -> list[int]:
    curve = curve.cpu().int()
    out = fvt_cpp.discontiguous_index(curve)
    return out


@autocast_torch
def split_by(array, key, n=-1):
    assert array.shape == key.shape and array.ndim == 1, "Only 1D arrays of the same shape are supported"
    return fvt_cpp.split_by(array.cpu().int(), key.cpu().int(), int(n))


@autocast_torch
def smooth_binary_mask(
    mask: torch.Tensor, sigma: float = 1.0, tol: float = 1e-3, mode: Literal["full", "safe", "same"] = "same"
) -> torch.Tensor:
    """Smooth a binary mask with a Gaussian kernel."""
    assert mask.ndim == 2, "Only 2D masks are supported"
    assert mask.dtype == torch.bool, "Mask must be of type torch.bool"
    smooth_mask = fvt_cpp.smooth_binary_mask(mask, float(sigma), float(tol))
    if mode == "full":
        return smooth_mask
    H, W = smooth_mask.shape
    p = (H - mask.shape[0]) // 2
    if mode == "same":
        return smooth_mask[p : H - p, p : W - p]
    elif mode == "safe":
        return smooth_mask[2 * p : H - 2 * p, 2 * p : W - 2 * p]
