from __future__ import annotations

import functools
import inspect
import warnings
from typing import Callable, Optional, TypeVar, Union, get_args, get_origin

import numpy as np
import torch

TensorArray = TypeVar("TensorArray", bound=torch.Tensor | np.ndarray)


def torch_interp_bilinear(
    imgs: torch.Tensor, y: torch.Tensor, x: torch.Tensor, batch_idx: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """2D bilinear interpolation for a batch of images.

    Parameters
    ----------
    imgs : torch.Tensor
        A batch of images with shape (C, H, W) if batch_idx is None, or (B, C, H, W) otherwise.
    y : torch.Tensor
        A vector of y coordinates with shape (N1, N2, ...).
    x : torch.Tensor
        A vector of x coordinates with shape (N1, N2, ...).
    batch_idx : Optional[torch.Tensor], optional
        A vector of batch indices with shape (N1, N2, ...), by default None. If None, ``imgs`` is expected to have no batch dimension, and all coordinates in ``coord`` are assumed to belong to the same image.

    Returns
    -------
    torch.Tensor
        A batch of interpolated values with shape (N1, N2, ..., C).
    """  # noqa: E501
    y0 = torch.clamp(torch.floor(y).long(), 0, imgs.shape[-2] - 2)
    x0 = torch.clamp(torch.floor(x).long(), 0, imgs.shape[-1] - 2)

    y1, x1 = y0 + 1, x0 + 1

    dy0 = (y1 - y)[..., None]
    dy1 = (y - y0)[..., None]
    dx0 = (x1 - x)[..., None]
    dx1 = (x - x0)[..., None]
    if batch_idx is None:
        img_y0x0 = imgs[:, y0, x0].view(*y0.shape, -1)
        img_y1x0 = imgs[:, y1, x0].view(*y0.shape, -1)
        img_y0x1 = imgs[:, y0, x1].view(*y0.shape, -1)
        img_y1x1 = imgs[:, y1, x1].view(*y0.shape, -1)
    else:
        b = batch_idx.long()
        img_y0x0 = imgs[b, :, y0, x0].view(*y0.shape, -1)
        img_y1x0 = imgs[b, :, y1, x0].view(*y0.shape, -1)
        img_y0x1 = imgs[b, :, y0, x1].view(*y0.shape, -1)
        img_y1x1 = imgs[b, :, y1, x1].view(*y0.shape, -1)

    return img_y0x0 * (dy0 * dx0) + img_y1x0 * (dy1 * dx0) + img_y0x1 * (dy0 * dx1) + img_y1x1 * (dy1 * dx1)


def img_to_torch(x, device="cuda"):
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        raise TypeError(f"Unknown type: {type(x)}.\n Expected numpy.ndarray or torch.Tensor.")

    match x.shape:
        case s if len(s) == 3:
            if s[2] == 3:
                x = x.permute(2, 0, 1)
            x = x.unsqueeze(0)
        case s if len(s) == 4:
            if s[3] == 3:
                x = x.permute(0, 3, 1, 2)
            assert x.shape[1] == 3, f"Expected 3 channels, got {x.shape[1]}"

    return x.float().to(device=device)


def recursive_numpy2torch(x, device=None):
    if isinstance(x, torch.Tensor):
        return x.to(device) if device is not None else x
    if isinstance(x, np.ndarray):
        with warnings.catch_warnings(action="ignore"):
            try:
                r = torch.from_numpy(x)
            except ValueError:
                r = torch.from_numpy(x.copy())
        if device is not None:
            r = r.to(device)
        return r
    if isinstance(x, dict):
        return {k: recursive_numpy2torch(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_numpy2torch(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple([recursive_numpy2torch(v, device) for v in x])
    return x


def recursive_torch2numpy(x):
    if isinstance(x, torch.Tensor):
        r = x.cpu().numpy()
        return r
    if isinstance(x, dict):
        return {k: recursive_torch2numpy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [recursive_torch2numpy(v) for v in x]
    if type(x) is tuple:
        return type(x)(recursive_torch2numpy(v) for v in x)
    return x


def torch_apply(func, *args, device=None, **kwargs):
    from_numpy = None
    new_args = []
    for arg in args:
        if from_numpy is None:
            if isinstance(arg, torch.Tensor):
                from_numpy = False
            if isinstance(arg, np.ndarray):
                from_numpy = True
        new_args.append(recursive_numpy2torch(arg, device))
    for key, value in kwargs.items():
        if from_numpy is None:
            if isinstance(value, torch.Tensor):
                from_numpy = False
            if isinstance(value, np.ndarray):
                from_numpy = True
        kwargs[key] = recursive_numpy2torch(value, device)

    r = func(*new_args, **kwargs)

    return recursive_torch2numpy(r) if from_numpy else r


def autocast_torch(f) -> Callable:
    def decorated_f(*args, **kwargs):
        return torch_apply(f, *args, **kwargs)

    def recursive_replace(ann):
        if get_origin(ann) is Union:
            return Union[tuple(recursive_replace(a) for a in get_args(ann))]
        return ann if ann is not torch.Tensor else TensorArray

    functools.update_wrapper(decorated_f, f)
    f_signature = inspect.signature(f)
    decorated_f.__signature__ = f_signature.replace(
        parameters=[
            param.replace(annotation=recursive_replace(param.annotation)) for param in f_signature.parameters.values()
        ],
        return_annotation=recursive_replace(f_signature.return_annotation),
    )

    return decorated_f


def to_torch(x: TensorArray, device: str | None = "cpu", dtype=None) -> torch.Tensor:
    tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    if device is not None:
        tensor = tensor.to(device)
    return tensor if dtype is None else tensor.to(dtype)
