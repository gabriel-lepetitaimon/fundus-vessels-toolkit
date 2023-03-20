import numpy as np


def symlog(start, stop, num, log_base=10):
    linthresh = (stop-start) / num
    linscale = 1
    start, stop = symlog_transform((start, stop), linthresh, linscale, log_base=log_base)
    space = np.linspace(start, stop, num)
    return symlog_itransform(space, linthresh, linscale, log_base=log_base)


def symlog_transform(a, linthresh, linscale, log_base=10):
    linscale_adj = linscale / (1 - 1/log_base)

    sign = np.sign(a)
    masked = np.ma.masked_inside(a, -linthresh, linthresh,
                                 copy=False)
    log = sign * linthresh * (linscale_adj + np.ma.log(np.abs(masked) / linthresh) / log_base)
    if masked.mask.any():
        return np.ma.where(masked.mask, a * linscale_adj, log)
    else:
        return log


def symlog_itransform(a, linthresh, linscale, log_base=10):
    linscale_adj = linscale / (1 - 1/log_base)
    invlinthresh = symlog_transform(linthresh, linthresh, linscale, log_base=log_base)

    sign = np.sign(a)
    masked = np.ma.masked_inside(a, -invlinthresh, invlinthresh, copy=False)
    exp = sign * linthresh * (np.ma.power(log_base, (sign * (masked / linthresh)) - linscale_adj))
    if masked.mask.any():
        return np.ma.where(masked.mask, a / linscale_adj, exp)
    else:
        return exp