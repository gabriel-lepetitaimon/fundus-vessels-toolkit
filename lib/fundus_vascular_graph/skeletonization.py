import numpy as np
import torch
import torch.nn.functional as F
from kornia.contrib import distance_transform


def torch_medial_axis(vessel_maps: torch.Tensor, keep_distance=True) -> torch.Tensor:
    """Skeletonize a binary image.
    Args:
        vessel_maps (torch.Tensor): the input image with shape :math:`(B?, 1?, H, W)`.
    Returns:
        torch.Tensor: the skeletonized image with shape :math:`(B, 1, H, W)`.
    """
    shape = vessel_maps.shape
    vessel_maps = _check_vessel_map(vessel_maps)

    # Compute Distance Transform
    dt = distance_transform(vessel_maps.unsqueeze(1).float())

    @torch.jit.script
    def rift_maxima(patch):
        v = patch[4]
        zero = torch.zeros_like(v)
        if torch.all(v == zero):
            return v
        else:
            q = torch.quantile(patch, 7/9, interpolation='lower')
            return v if torch.all(zero < q <= v) else zero
    rift_maxima = torch.func.vmap(rift_maxima)

    dt = F.unfold(dt, kernel_size=3, padding=1)
    dt = dt.permute(0, 2, 1).reshape(-1, 9)
    dt = rift_maxima(dt)
    dt = dt.reshape(shape[0], -1, shape[-2], shape[-1])
    return dt if keep_distance else dt > 0

    # Mask pixel with 0 distance
    not_null_image, not_null_row, not_null_col = torch.where(vessel_maps != 0)
    not_null_index = not_null_image * np.prod(vessel_maps.shape[1:]) + not_null_row * vessel_maps.shape[2] + not_null_col
    not_null_index = not_null_index.reshape(-1)

    # Cancel out any pixels that do not belong the local maximas rifts
    unfold_dt = F.unfold(dt, kernel_size=3, padding=1)
    unfold_dt = unfold_dt.permute(0, 2, 1).reshape(-1, 9)[not_null_index]
    quantile = torch.quantile(unfold_dt, 7/9, dim=1, interpolation='lower')

    print(unfold_dt.shape, quantile.shape)
    not_local_maxima = torch.where((quantile > 0) & (unfold_dt[:, 4] > quantile))[0]
    dt = dt.reshape(-1)[not_null_index]
    dt[not_local_maxima] = 0

    final_shape = (shape[0],) + tuple(shape[-2:])
    skeleton = torch.zeros(np.prod(final_shape), dtype=dt.dtype, device=vessel_maps.device)
    skeleton[not_null_index] = dt
    skeleton = skeleton.reshape(final_shape)
    if not keep_distance:
        return skeleton > 0
    else:
        return skeleton


def _check_vessel_map(vessel_maps):
    if vessel_maps.ndim == 4:
        assert vessel_maps.shape[1] == 1, f"Expected 2D vessels map of shapes (B, 1, H, W), but provided maps has multiple channels."
        vessel_maps = vessel_maps.squeeze(1)
    elif vessel_maps.ndim == 2:
        vessel_maps = vessel_maps.unsqueeze(0)
    else:
        assert vessel_maps.ndim == 3, f"Expected 2D vessels maps of shapes (B?, 1?, H, W), got {vessel_maps.ndim}D tensor."

    if vessel_maps.dtype != torch.bool:
        vessel_maps = vessel_maps > 0.5
    return vessel_maps
