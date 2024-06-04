import numpy as np


def ensure_superior_multiple(x, m=32):
    """
    Return y such that y >= x and y is a multiple of m.
    """
    return m - (x - 1) % m + x - 1


def gaussian(x, sigma):
    return np.exp(-(x**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def gaussian_filter1d(x, sigma, integrate=False):
    from scipy.signal import convolve

    t = np.arange(-sigma * 3, sigma * 3 + 1)
    kernel = gaussian(t, sigma)
    kernel = kernel / kernel.sum()
    return convolve(x, kernel, mode="same")
