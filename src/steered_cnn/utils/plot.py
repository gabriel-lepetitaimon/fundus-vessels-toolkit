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


#####################################################################################
#                           Custom pyplot Scales                                    #
#####################################################################################
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import matplotlib.ticker as ticker


class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'sqrt'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        #mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0., vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


class SquareScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    """

    name = 'sqr'

    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        #mscale.ScaleBase.__init__(self, axis)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax

    class SquareTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a)**2

        def inverted(self):
            return SquareScale.InvertedSquareTransform()

    class InvertedSquareTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform(self, a):
            return np.array(a)**.5

        def inverted(self):
            return SquareScale.SquareTransform()

    def get_transform(self):
        return self.SquareTransform()


mscale.register_scale(SquareRootScale)
mscale.register_scale(SquareScale)
