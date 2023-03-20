import torch
import torch.nn.functional as F
from typing import Union, Tuple


def clip_pad_center(tensor, shape, pad_mode='constant', pad_value=0, broadcastable=False):
    H, W = tensor.shape[-2:]
    h, w = shape[-2:]
    if H == 1 and broadcastable:
        y0 = 0
        y1 = 0
        h = 1
        yodd = 0
    else:
        y0 = (H-h)//2
        y1 = 0
        yodd = 0
        if y0 < 0:
            y1 = -y0
            y0 = 0
            yodd = (h-H) % 2

    if W == 1 and broadcastable:
        x0 = 0
        x1 = 0
        w = 1
        xodd = 0
    else:
        x0 = (W-w)//2
        x1 = 0
        xodd = 0
        if x0 < 0:
            x1 = -x0
            x0 = 0
            xodd = (w-W) % 2

    tensor = tensor[..., y0:y0+h, x0:x0+w]
    if x1 or y1:
        tensor = F.pad(tensor, (x1-xodd, x1, y1-yodd, y1), mode=pad_mode, value=pad_value)
    return tensor


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


def normalize_vector(vector: Union[Tuple[torch.Tensor], torch.Tensor], epsilon: int = 1e-8):
    """
    Normalize a vector field to unitary norm.
    Args:
        vector:
        epsilon:

    Shape:
        vector: [2, ...]

    Returns: The vector of unitary norm,
             the norm maxtrix.

    """
    if isinstance(vector, (list, tuple)):
        vectors = []
        norms = []
        for v in vector:
            v, norm = normalize_vector(v, epsilon)
            vectors += [v]
            norms += [norm]
        return vectors, norms
    d = torch_norm2d(vector)
    return vector / (d+epsilon), d


_norm2d = [None]
def torch_norm2d(xy):
    if _norm2d[0] is None:
        import torch
        def linalg_norm(xy):
            return torch.linalg.norm(xy, dim=0)
        def legacy_norm(xy):
            return torch.norm(xy, dim=0)
        try:
            d = linalg_norm(xy)
            _norm2d[0] = linalg_norm
        except AttributeError:
            d = legacy_norm(xy)
            _norm2d[0] = legacy_norm
        return d
    else:
        return _norm2d[0](xy)
