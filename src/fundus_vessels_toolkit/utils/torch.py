from __future__ import annotations

import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Callable, Literal, Optional, TypeVar, Union, get_args, get_origin, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

TensorArray = TypeVar("TensorArray", bound=Tensor | np.ndarray)


def torch_interp_bilinear(
    imgs: Tensor,
    y: Tensor,
    x: Tensor,
    batch_idx: Optional[Tensor] = None,
    img_shape: Optional[tuple[int, int]] = None,
    legacy: bool = True,
) -> Tensor:
    """2D bilinear interpolation for a batch of images.

    Parameters
    ----------
    imgs : Tensor
        A batch of images with shape (C, H, W) if batch_idx is None, or (B, C, H, W) otherwise.
    y : Tensor
        A vector of y coordinates with shape (N1, N2, ...).
    x : Tensor
        A vector of x coordinates with shape (N1, N2, ...).
    batch_idx : Optional[Tensor], optional
        A vector of batch indices with shape (N1, N2, ...), by default None. If None, ``imgs`` is expected to have no batch dimension, and all coordinates in ``coord`` are assumed to belong to the same image.

    img_shape: Optional[tuple[int, int]], optional
        The initial shape of the input images (H, W) that the coordinates refer to, by default None. If not None, the input coordinates are rescaled from the original image size to the actual image size in ``imgs``.

    legacy: bool
        If True, use legacy python implementation, otherwise use torch.nn.function.grid_sample.

    Returns
    -------
    Tensor
        A batch of interpolated values with shape (C, ..., N1, N2).

    Example
    -------
    >>> torch.set_printoptions(precision=1, sci_mode=False)
    >>> img = torch.tensor([[[0,0],[1,1]],[[0,1],[0,1]]]).float() # C=2, H=2, W=2
    >>> imgs = torch.stack([img, img+10], dim=0) # B=2, C=2, H=2, W=2
    >>> y = torch.tensor([[0, 0.5, 1.5], [0, 0.5, 1]]) # N=(2,3)
    >>> x = torch.tensor([[0, 0, 1], [0, 0, 1]])
    >>> batch_idx = torch.tensor([[0, 0, 0], [1, 1, 1]])
    >>> torch_interp_bilinear(imgs, y, x, batch_idx, legacy=False)
    tensor([[[ 0.0,  0.0],
             [ 0.5,  0.0],
             [ 1.0,  1.0]],
            [[10.0, 10.0],
             [10.5, 10.0],
             [11.0, 11.0]]])
    """  # noqa: E501
    H, W = imgs.shape[-2:]
    if img_shape is not None and img_shape != imgs.shape[-2:]:
        H_orig, W_orig = img_shape
    elif not y.dtype.is_floating_point and y.dtype.is_floating_point:
        return imgs[batch_idx, :, y, x] if batch_idx is not None else imgs[:, y, x]
    else:
        H_orig, W_orig = H, W

    y = torch.clamp(y, 0, H_orig - 1)
    x = torch.clamp(x, 0, W_orig - 1)

    if legacy:
        y = y * ((H - 1) / (H_orig - 1))
        x = x * ((W - 1) / (W_orig - 1))
        y0 = torch.floor(y).long()
        x0 = torch.floor(x).long()
        y1 = y0 + 1
        x1 = x0 + 1

        dy1 = (y - y0)[..., None]
        dy0 = 1 - dy1
        dx1 = (x - x0)[..., None]
        dx0 = 1 - dx1
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

        v = img_y0x0 * (dy0 * dx0) + img_y1x0 * (dy1 * dx0) + img_y0x1 * (dy0 * dx1) + img_y1x1 * (dy1 * dx1)
        return v.permute(-1, *range(len(y.shape)))  # C, N1, N2, ...
    else:
        Ns = y.shape
        C = imgs.shape[-3]
        if batch_idx is not None:
            B = imgs.shape[0]
            y += B * batch_idx
            H_orig = H * B
            imgs = imgs.permute(1, 0, 2, 3).flatten(1, 2)  # C, B*H, W
        x = x * (2 / (W_orig - 1)) - 1
        y = y * (2 / (H_orig - 1)) - 1
        imgs = imgs.unsqueeze(0)  # 1, C, H, W

        grid = torch.stack((x.flatten(), y.flatten()), dim=-1)[None, None, :, :]  # 1, 1, N, 2
        v = torch.nn.functional.grid_sample(imgs, grid, align_corners=True)  # 1, C, 1, N
        return v.squeeze(0, 2).reshape(C, *Ns)


def grid_indices(shape: tuple[int, int], device=None) -> Tensor:
    """Return the grid indices for a given shape.

    Parameters
    ----------
    shape : tuple[int, int]
        The shape of the grid (H, W).
    device : torch.device, optional
        The device to return the tensors on, by default None.

    Returns
    -------
    Tensor:
        A tensor of shape (H, W, 2) containing the grid indices for each pixel.
    """
    H, W = shape
    y = torch.arange(H, device=device)
    x = torch.arange(W, device=device)
    return torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)  # (H, W, 2)


def groupby_mean(x: Tensor, group_idx: Tensor, *, num_group: Optional[int] = None) -> Tensor:
    """Compute the mean of values in `x` grouped by `group_idx`.

    Parameters
    ----------
    x : Tensor
        A tensor of shape (N,) containing the values to be averaged.
    group_idx : Tensor
        A tensor of shape (N,) containing the group indices for each value in `x`. The values in `group_idx` should be non-negative integers.

    Returns
    -------
    Tensor
        A tensor of shape (G,) containing the mean values for each group, where G is the maximum group index + 1.
    """  # noqa: E501
    group_idx = group_idx.long()
    if num_group is None:
        num_group = int(group_idx.max().item()) + 1
    group_sum = torch.zeros(num_group, dtype=x.dtype, device=x.device).scatter_add_(0, group_idx, x)
    count = torch.bincount(group_idx, minlength=num_group)
    not_null_mask = count != 0
    group_sum[not_null_mask] /= count[not_null_mask].float()
    return group_sum


def rng_shuffle(x: Tensor, *, return_inverse: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """Randomly shuffle the elements of `x` along the first dimension.

    Parameters
    ----------
    x : Tensor
        A tensor of shape (N, ...) containing the values to be shuffled.
    return_inverse : bool, optional
        Whether to return the inverse permutation indices, by default False.

    Returns
    -------
    Tensor
        A tensor of shape (N, ...) containing the shuffled values.
    Tensor, optional
        If `return_inverse` is True, a tensor of shape (N,) containing the indices that can be used to restore the original order of `x`.
    """  # noqa: E501
    perm = torch.randperm(x.shape[0], device=x.device)
    if return_inverse:
        inverse_perm = torch.empty_like(perm)
        inverse_perm[perm] = torch.arange(len(perm), device=x.device)
        return x[perm], inverse_perm
    else:
        return x[perm]


def randperm_with_inverse(n: int, *, device=None) -> tuple[Tensor, Tensor]:
    """Randomly shuffle the elements of `x` along the first dimension.

    Parameters
    ----------
    x : Tensor
        A tensor of shape (N, ...) containing the values to be shuffled.
    return_inverse : bool, optional
        Whether to return the inverse permutation indices, by default False.

    Returns
    -------
    Tensor
        A tensor of shape (N, ...) containing the shuffled values.
    Tensor, optional
        If `return_inverse` is True, a tensor of shape (N,) containing the indices that can be used to restore the original order of `x`.
    """  # noqa: E501
    perm = torch.randperm(n, device=device)
    inverse_perm = torch.empty_like(perm)
    inverse_perm[perm] = torch.arange(len(perm), device=device)
    return perm, inverse_perm


def with_weight[**P](func: Callable[P, Tensor], w: float) -> Callable[P, Tensor]:
    if w == 1.0:
        return func

    @functools.wraps(func)
    def decorated_func(*args: P.args, **kwargs: P.kwargs) -> Tensor:
        return func(*args, **kwargs) * w

    return decorated_func


def lru_cache(maxsize=128):
    def decorator(func):
        func.cache = {}

        @functools.wraps(func)
        def wrapper(*args):
            if len(args) == 0 or not isinstance(args, torch.nn.Module):
                self = None
            else:
                self = id(args[0])
                args = args[1:]
            self_cache = func.cache.setdefault(self, OrderedDict())
            key = tuple(id(arg) + sum(arg.shape) if isinstance(arg, Tensor) else arg for arg in args)
            if (out := self_cache.get(key)) is not None:
                self_cache.move_to_end(key, last=False)
                return out

            out = func(*args)
            self_cache[key] = out
            if len(self_cache) > maxsize:
                self_cache.popitem(last=True)
            return out

        return wrapper

    return decorator


@overload
def unique_first(x: Tensor, *, return_inverse: Literal[False] = False) -> tuple[Tensor, Tensor]: ...
@overload
def unique_first(x: Tensor, *, return_inverse: Literal[True]) -> tuple[Tensor, Tensor, Tensor]: ...
def unique_first(x: Tensor, *, return_inverse: bool = False) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Return the indices of the first occurrence of each unique value in `x`.

    Parameters
    ----------
    x : Tensor
        An integer tensor of shape (N,) containing the values to be processed.

    Returns
    -------
    unique_values : Tensor
        A tensor of shape (G,) containing the unique values in `x`, where G is the number of unique values in `x`.
    first_indices : Tensor
        A tensor of shape (G,) containing the indices of the first occurrence of each unique value in `x`, where G is the number of unique values in `x`.
    """  #  # noqa: E501
    unique_values, inverse_idxs, counts = x.unique(return_inverse=True, return_counts=True)
    grouped_idxs = inverse_idxs.argsort(stable=True)
    group_start_idxs = counts.cumsum(0).roll(1)
    group_start_idxs[0] = 0
    if return_inverse:
        return unique_values, grouped_idxs[group_start_idxs], inverse_idxs
    else:
        return unique_values, grouped_idxs[group_start_idxs]


def img_to_torch(x, device="cuda"):
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        x = torch.from_numpy(x)
    elif not isinstance(x, Tensor):
        raise TypeError(f"Unknown type: {type(x)}.\n Expected numpy.ndarray or Tensor.")

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
    if isinstance(x, Tensor):
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
    if isinstance(x, Tensor):
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
            if isinstance(arg, Tensor):
                from_numpy = False
            if isinstance(arg, np.ndarray):
                from_numpy = True
        new_args.append(recursive_numpy2torch(arg, device))
    for key, value in kwargs.items():
        if from_numpy is None:
            if isinstance(value, Tensor):
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
        return ann if ann is not Tensor else TensorArray

    functools.update_wrapper(decorated_f, f)
    f_signature = inspect.signature(f)
    decorated_f.__signature__ = f_signature.replace(
        parameters=[
            param.replace(annotation=recursive_replace(param.annotation)) for param in f_signature.parameters.values()
        ],
        return_annotation=recursive_replace(f_signature.return_annotation),
    )

    return decorated_f


def to_torch(x: Tensor | npt.NDArray, device: str | None = "cpu", dtype=None) -> Tensor:
    tensor = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
    if device is not None:
        tensor = tensor.to(device)
    return tensor if dtype is None else tensor.to(dtype)
