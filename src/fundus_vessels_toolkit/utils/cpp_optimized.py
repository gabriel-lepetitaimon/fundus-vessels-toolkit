import torch

from .cpp_extensions import fvt_cpp
from .torch import autocast_torch


@autocast_torch
def first_index_of(array, search_for, out=None):
    array = array.cpu().int()
    search_for = search_for.cpu().int()
    if out is None:
        out = torch.empty((search_for.shape[0],), dtype=torch.int32)

    fvt_cpp.first_index_of(array, search_for, out)

    return out


@autocast_torch
def first_two_index_of(array, search_for, out=None):
    array = array.cpu().int()
    search_for = search_for.cpu().int()
    if out is None:
        out = torch.empty((search_for.shape[0], 2), dtype=torch.int32)

    fvt_cpp.first_two_index_of(array, search_for, out)

    return out
