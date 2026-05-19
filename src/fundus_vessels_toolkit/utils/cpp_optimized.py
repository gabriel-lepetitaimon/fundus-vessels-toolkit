from ast import List
from enum import auto

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
