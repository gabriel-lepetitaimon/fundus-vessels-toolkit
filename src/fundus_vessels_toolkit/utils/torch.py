from __future__ import annotations

import functools
import inspect
import warnings
from typing import Callable, Literal, Optional, TypeVar, Union, get_args, get_origin

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


def groupby_mean(x: torch.Tensor, group_idx: torch.Tensor, *, num_group: Optional[int] = None) -> torch.Tensor:
    """Compute the mean of values in `x` grouped by `group_idx`.

    Parameters
    ----------
    x : torch.Tensor
        A tensor of shape (N,) containing the values to be averaged.
    group_idx : torch.Tensor
        A tensor of shape (N,) containing the group indices for each value in `x`. The values in `group_idx` should be non-negative integers.

    Returns
    -------
    torch.Tensor
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


def unique_first(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the indices of the first occurrence of each unique value in `x`.

    Parameters
    ----------
    x : torch.Tensor
        An integer tensor of shape (N,) containing the values to be processed.

    Returns
    -------
    unique_values : torch.Tensor
        A tensor of shape (G,) containing the unique values in `x`, where G is the number of unique values in `x`.
    first_indices : torch.Tensor
        A tensor of shape (G,) containing the indices of the first occurrence of each unique value in `x`, where G is the number of unique values in `x`.
    """  #  # noqa: E501
    unique_values, inverse_idxs, counts = x.unique(return_inverse=True, return_counts=True)
    grouped_idxs = inverse_idxs.argsort(stable=True)
    group_start_idxs = counts.cumsum(0).roll(1)
    group_start_idxs[0] = 0
    return unique_values, grouped_idxs[group_start_idxs]


class GroupByCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
        label_smoothing: float = 0.0,
        invalid_metagroup_penalty: float = 0.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.invalid_metagroup_penalty = invalid_metagroup_penalty

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        group_idx: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        metagroup_idx: Optional[torch.Tensor] = None,
        other_group_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss which enforce the selection of one sample per group, given sample-wise logits and group indices.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of shape (N,) containing the predicted logits for each sample.

        target : torch.Tensor
            A tensor of shape (N,) containing whether each sample is the elected one in its group. The sum of target values for each group should be 1.

        group_idx : torch.Tensor
            A tensor of shape (N,) containing the group affiliation for each sample. The values in `group_idx` should be non-negative integers, and samples with the same group index belong to the same group.

        mask : Optional[torch.Tensor], optional
            A boolean tensor of shape (N,) indicating which group should be included in the loss computation. If None, all samples are included.

        metagroup_idx : Optional[torch.Tensor], optional
            A tensor of shape (G,) containing the metagroup affiliation for each group.

        other_group_idx : Optional[torch.Tensor], optional
            A tensor of shape (N,) containing the group affiliation for each sample according to another grouping scheme, used to apply a penalty to samples belonging to metagroup that are not consistent between the two grouping schemes.

        Returns
        -------
        torch.Tensor
            A scalar tensor containing the mean cross-entropy loss over all groups.
        """  # noqa: E501
        group_idx = group_idx.long()
        num_group = len(mask) if mask is not None else int(group_idx.max().item()) + 1
        group_size = torch.bincount(group_idx, minlength=num_group)
        assert torch.all(group_size > 0), "All groups must have at least one sample."

        # Normalize logits by group max for numerical stability
        group_max = torch.zeros(num_group, device=x.device).scatter_reduce_(0, group_idx, x, "amax", include_self=False)
        x_normed = x - group_max[group_idx]

        # Add penalty for invalid metagroups if provided
        if self.invalid_metagroup_penalty > 0 and metagroup_idx is not None and other_group_idx is not None:
            penalized_samples = (other_group_idx >= 0) & metagroup_idx[group_idx] != other_group_idx[group_idx]
            x_normed[penalized_samples] += self.invalid_metagroup_penalty

        # Compute the log-sum-exp
        group_sum_exp = torch.zeros(num_group, device=x.device).scatter_add_(0, group_idx, x_normed.exp())
        group_log_sum_exp = (group_sum_exp + 1e-16).log()

        # Compute the log-softmax
        log_softmax = x_normed - group_log_sum_exp[group_idx]

        # Compute the loss
        if self.label_smoothing > 0:
            log_softmax = log_softmax * (1 - self.label_smoothing) + self.label_smoothing / group_size[group_idx]
        loss = -log_softmax * target

        # Sum over samples in the same group
        group_loss = torch.zeros(num_group, device=x.device).scatter_add_(0, group_idx, loss)

        # Apply mask if provided
        if mask is not None:
            group_loss = group_loss[mask]

        # Reduce the loss according to the specified reduction method
        if self.reduction == "mean":
            group_loss = group_loss.sum() / num_group
        elif self.reduction == "sum":
            group_loss = group_loss.sum()
        return group_loss


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
