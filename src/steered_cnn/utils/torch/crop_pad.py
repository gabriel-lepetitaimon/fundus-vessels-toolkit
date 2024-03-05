import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple


def crop_pad(img, shape, center=(0.5, 0.5), pad_mode='constant', pad_value=0, broadcastable=False):
    H, W = img.shape[-2:]
    h, w = shape[-2:]
    y, x = (int(round((c % 1)*s)) if isinstance(c, float) and -1 <= c <= 1 else c
            for c, s in zip(center, (H, W)))

    if H == 1 and broadcastable:
        y1 = 0
        y2 = 1
        pad_y1 = 0
        pad_y2 = 0
    else:
        pad_y1, y1 = 0, y-h//2
        if y1 < 0:
            pad_y1, y1 = -y1, 0
        y2 = min(y1+h, H)
        pad_y2 = h-(y2-(y1-pad_y1))

    if W == 1 and broadcastable:
        x1 = 0
        x2 = 1
        pad_x1 = 0
        pad_x2 = 0
    else:
        pad_x1, x1 = 0, x - w // 2
        if x1 < 0:
            pad_x1, x1 = -x1, 0
        x2 = min(x1+w, W)
        pad_x2 = w-(x2-(x1-pad_x1))

    img = img[..., y1:y2, x1:x2]
    if pad_x1 or pad_x2 or pad_y1 or pad_y2:
        if isinstance(img, torch.Tensor):
            img = F.pad(img, (pad_x1, pad_x2, pad_y1, pad_y2), mode=pad_mode, value=pad_value)
        elif isinstance(img, np.ndarray):
            img = np.pad(img, ((0, 0),)*(img.ndim-2)+((pad_y1, pad_y2), (pad_x1, pad_x2)),
                         mode=pad_mode, constant_values=pad_value)
    return img


def clip_tensors(t1, t2):
    if t1.shape[-2:] == t2.shape[-2:]:
        return t1, t2
    h1, w1 = t1.shape[-2:]
    h2, w2 = t2.shape[-2:]
    dh = h1-h2
    dw = w1-w2
    i1 = max(dh, 0)
    j1 = max(dw, 0)
    h = h1 - i1
    w = w1 - j1
    i1 = i1 // 2
    j1 = j1 // 2
    i2 = i1 - dh//2
    j2 = j1 - dw//2

    t1 = t1[..., i1:i1+h, j1:j1+w]
    t2 = t2[..., i2:i2+h, j2:j2+w]
    return t1, t2


def pad_tensors(t1, t2, pad_mode='constant', pad_value=0):
    if t1.shape[-2:] == t2.shape[-2:]:
        return t1, t2

    def half_odd(v):
        return v//2, v % 2

    h1, w1 = t1.shape[-2:]
    h2, w2 = t2.shape[-2:]

    dh = h1-h2
    dh2 = max(dh, 0)
    dh1, dh1_odd = half_odd(dh2-dh)
    dh2, dh2_odd = half_odd(dh2)

    dw = w1-w2
    dw2 = max(dw, 0)
    dw1, dw1_odd = half_odd(dw2-dw)
    dw2, dw2_odd = half_odd(dw2)

    if dw1+dw1_odd or dh1+dh1_odd:
        t1 = F.pad(t1, (dh1, dh1+dh1_odd, dw1, dw1+dw1_odd), mode=pad_mode, value=pad_value)
    if dw2+dw2_odd or dh2+dh2_odd:
        t2 = F.pad(t2, (dh2, dh2+dh2_odd, dw2, dw2+dw2_odd), mode=pad_mode, value=pad_value)
    return t1, t2


def cat_crop(x1, x2):
    return torch.cat(clip_tensors(x1, x2), 1)


def select_pixels_by_mask(*tensors, mask):
    """
    Select pixels according to mask.
    :param tensors: Tensors of shape either:
        • [B, H, W] : then the returned tensors is vector of shape [M] (M being the count of non-zero element of mask)
        • [B, C, H, W] : then the shape of the returned tensors is [M, C]
    :param mask: a boolean matrix of shape [B, H, W]
    :return:
    """
    if mask is not None:
        mask = mask.to(torch.bool)
        clipped_mask = None
        selected = []
        for t in tensors:
            if clipped_mask is None or t.shape[-2:] != clipped_mask.shape:
                clipped_mask = crop_pad(mask, t.shape)

            if clipped_mask.ndim < t.ndim and t.shape[1] != 1:
                c = t.shape[1]
                t = torch.movedim(t, 1, 0).flatten(1)
                clipped_mask = clipped_mask.flatten().unsqueeze(0).expand(t.shape)
                masked_t = t.flatten()[clipped_mask.flatten()].reshape((c, -1))
                selected += [masked_t.T]
            else:
                if clipped_mask.ndim > t.ndim:
                    clipped_mask = clipped_mask.squeeze(1)
                elif clipped_mask.ndim < t.ndim:
                    t = t.unsqueeze(1)
                selected += [t[clipped_mask]]
    else:
        selected = [t.flatten() for t in tensors]
    return selected

