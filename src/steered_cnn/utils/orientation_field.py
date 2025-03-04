import numpy as np
import torch
from scipy import stats as st


def gkern(n=21, sigma=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-sigma, sigma, n + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


_G_xy_cached = {}


def G_xy(std):
    if std in _G_xy_cached:
        return _G_xy_cached[std]
    n = std * 12 + 1
    x = np.linspace(-std * 3, std * 3, n)
    y, x = np.meshgrid(x, x)
    G = (gkern(n, std) + 1e-6) / (np.sqrt(x * x + y * y) + 1e-8)
    x *= G
    y *= G
    G = np.stack((x, y))
    _G_xy_cached[std] = G
    return G


def compute_field_torch(skeleton, std=None):
    if std is None:
        std = int(np.ceil(max(skeleton.shape[-2:]) / 20))  # std = ceil(max(h,w)/20)
    with torch.no_grad():
        skeleton = torch.from_numpy(skeleton).cuda().double()
        G = torch.from_numpy(G_xy(std))[:, None, :, :].cuda()
        f = torch.conv2d(skeleton[:, None, :, :], G, padding="same")
        f = f[:, 0] + 1j * f[:, 1]
        return f.cpu().numpy()
