import numpy as np
import scipy.stats as st
from scipy.ndimage import gaussian_filter


def fundus_preprocessing(x):
    from fundus_toolkits.utils.safe_import import cv2

    k = np.max(x.shape) // 20 * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k // 2 + 1, k // 2 + 1))

    mask_org = (x[0, :, :] > 10 / 255.0).astype(np.uint8)
    mask = cv2.erode(mask_org, np.ones((15, 15), np.uint8))

    mask = np.expand_dims(mask, 0).astype(np.uint8)
    mask_org = np.expand_dims(mask_org, 2)
    x_cv = (x * 255).transpose((1, 2, 0)).astype(np.uint8)
    dilation = cv2.dilate(x_cv, kernel, iterations=1)
    dilation = dilation.astype(np.float32).transpose((2, 0, 1)) / 255.0
    fundus = preprocess(dilation * (1 - mask) + mask * x) * mask
    return fundus


def preprocess(img):
    sigma = np.max(img.shape) / 60
    blur = np.stack(
        [
            gaussian_filter(img[0], sigma, truncate=6.5),
            gaussian_filter(img[1], sigma, truncate=6.5),
            gaussian_filter(img[2], sigma, truncate=6.5),
        ]
    )
    return (img - blur - 0.0022501) / 0.02771


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
    n = std * 6 + 1
    x = np.linspace(-std * 3, std * 3, n)
    y, x = np.meshgrid(x, x)
    G = (gkern(n, std) + 1e-6) / (np.sqrt(x * x + y * y) + 1e-8)
    x *= G
    y *= G
    G = np.stack((x, y))
    _G_xy_cached[std] = G
    return G


def compute_skeleton_field(skeleton, std=None):
    if std is None:
        std = int(np.ceil(max(skeleton.shape[-2:]) / 20))  # std = ceil(max(h,w)/20)

    G = G_xy(std)
    n = G.shape[-1]
    half = n // 2

    sk_grad = np.zeros(shape=(2,) + skeleton.shape, dtype=np.float32)
    h, w = skeleton.shape
    for i, j in np.argwhere(skeleton):
        i1 = max(half - i, 0)
        i2 = min(n, h + half - i)
        h0 = i2 - i1
        i0 = i + i1 - half

        j1 = max(half - j, 0)
        j2 = min(n, w + half - j)
        w0 = j2 - j1
        j0 = j + j1 - half
        sk_grad[:, i0 : i0 + h0, j0 : j0 + w0] += G[:, i1:i2, j1:j2]

    return sk_grad
