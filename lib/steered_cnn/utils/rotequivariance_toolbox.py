import numpy as np


def cartesian_space(size, center=None):
    if not isinstance(size, tuple):
        size = (size, size)
    if center is None:
        center = tuple((_-1) / 2 for _ in size)

    x = np.arange(-center[0], size[0] - center[0], dtype=np.float32)
    y = np.arange(-center[1], size[1] - center[1], dtype=np.float32)
    x, y = np.meshgrid(x, y)
    return y, x


def r_space(size, center=None):
    return np.linalg.norm(cartesian_space(size, center=center), axis=0)


def polar_space(size, center=None):
    x, y = cartesian_space(size, center=center)
    rho = np.linalg.norm((x, y), axis=0)
    phi = np.arctan2(y, x)
    return rho, phi


def spectral_power(arr: 'θ.hw', plot=False, split=False, sort=True, mask=None):
    from scipy.fftpack import fft
    import matplotlib.pyplot as plt

    if mask == 'disk':
        r = min(arr.shape[-2:])
        mask = (r_space(arr.shape[-2:])*2) <= r
    if mask is not None:
        if mask.dtype != np.bool:
            mask = mask > 0
        arr = arr[..., mask != 0]

    spe = fft(arr, axis=0)
    spe = abs(spe) ** 2
    if split:
        spe = spe.reshape(spe.shape[:2] + (-1,)).sum(axis=-1)
    else:
        spe = spe.reshape(spe.shape[:1] + (-1,)).sum(axis=1)
    if plot:
        fig = None
        scale = False
        if isinstance(plot, str):
            scale = plot
            plot = True
        if plot is True:
            fig, plot = plt.subplots()

        N = spe.shape[0] // 2 + 1

        if split:
            W = 0.8
            w = W / spe.shape[1]

            spe = spe[:N]
            if split == 'normed':
                spe = spe / spe.sum(axis=tuple(_ for _ in range(spe.ndim) if _ != 1))[None, :]
            else:
                spe = spe / spe.sum(axis=-1).mean(axis=tuple(_ for _ in range(spe.ndim)
                                                             if _ not in (1, spe.ndim-1)))[None, :]
            if sort:
                idx = spe[0].argsort()
                spe = spe[:, idx[::-1]]
            for i in range(spe.shape[1]):
                y = spe[:, i]
                x = np.arange(len(y))
                plot.bar(x + w / 2 - W / 2 + i * w, y, width=w, bottom=0.001, zorder=10)
        else:
            y = spe[:N] / spe[:N].sum()
            x = np.arange(N)
            plot.bar(x, y, width=.8, bottom=0.001, zorder=10, color='gray')

        plot.spines['top'].set_visible(False)
        plot.spines['right'].set_visible(False)
        plot.spines['left'].set_visible(False)

        plot.set_xticks(np.arange(0, N, 1))
        xlabels = ['Equivariant'] + [f'${repr_pi_fraction(2,k)}$' for k in range(1, N)]
        plot.set_xticklabels(xlabels)

        plot.set_ylabel('Polar Spectral Power Density')
        plot.set_ylim([0.001, 1])
        plot.set_yticks([.25, .5, .75, 1])
        plot.set_yticklabels(['25%', '50%', '75%', '100%'])
        plot.yaxis.grid()
        if scale:
            plot.set_yscale(scale)
        plot.grid(which='minor', color='#bbbbbb', linestyle='-', linewidth=1, zorder=1)

        if fig is not None:
            fig.show()
    return spe


def polar_spectral_power(arr: '.hw', theta=8, plot=False, split=False, mask=None):
    if mask == 'auto':
        pad = 'auto'
        mask = None
    elif mask == 'disk':
        r = min(arr.shape[-2:])
        mask = (r_space(arr.shape[-2:]) * 2) <= r
        pad = 0
    elif isinstance(mask, np.ndarray):
        pad = int(np.max(mask*np.ceil(r_space(mask.shape[-2:]))))
    else:
        pad = 0
    arr = rotate(arr, theta, pad=pad)
    return spectral_power(arr, plot=plot, split=split, mask=mask)


#####################################################################################
#                       Rotation Equivariance Measures                              #
#####################################################################################
DEFAULT_ROT_ANGLE = np.arange(10, 360, 10)


def rotate(arr, angles=DEFAULT_ROT_ANGLE, pad=0):
    from skimage.transform import rotate as imrotate
    import math
    if pad == 'auto':
        pad = math.ceil(max(arr.shape[-2:])/math.sqrt(2))
    if isinstance(pad, int):
        pad = ((0, 0),)*(arr.ndim-2) + ((pad, pad),)*2
        arr = np.pad(arr, pad)

    shape = arr.shape
    if isinstance(angles, int):
        angles = np.linspace(0, 360, angles, endpoint=False)[1:]
    arr = arr.reshape((-1,) + arr.shape[-2:]).transpose((1, 2, 0))
    arr = np.stack([arr] + [imrotate(arr, -a) for a in angles])
    return arr.transpose((0, 3, 1, 2)).reshape((len(angles) + 1,) + shape)


def unrotate(arr: 'θ.hw', angles=None, pad=0) -> 'θ.hw':
    from skimage.transform import rotate as imrotate
    import math
    if pad == 'auto':
        pad = math.ceil(max(arr.shape[-2:]) / math.sqrt(2))
    if isinstance(pad, int):
        pad = ((0, 0),) * (arr.ndim - 2) + ((pad, pad),) * 2
        arr = np.pad(arr, pad)

    if angles is None:
        angles = np.linspace(0, 360, arr.shape[0], endpoint=False)[1:]
    elif isinstance(angles, int):
        angles = np.linspace(0, 360, angles)[1:]

    shape = arr.shape
    arr = arr.reshape((arr.shape[0], -1) + arr.shape[-2:]).transpose((0, 2, 3, 1))
    arr = np.stack([arr[0]] +
                   [imrotate(ar, ang) for ar, ang in zip(arr[1:], angles)])
    return arr.transpose((0, 3, 1, 2)).reshape(shape)


def rotate_vect(arr_xy, angles=DEFAULT_ROT_ANGLE, reproject=True):
    if isinstance(angles, int):
        angles = np.linspace(0, 360, angles, endpoint=False)[1:]

    x,y = arr_xy
    x = rotate(x, angles)
    y = rotate(y, angles)

    z = x+1j*y
    angle_offset = np.concatenate([[0], angles])
    while angle_offset.ndim < z.ndim:
        angle_offset = np.expand_dims(angle_offset, -1)
    theta = (np.angle(z, deg=True) + angle_offset)
    r = np.abs(z)
    if reproject:
        theta *= np.pi/180
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    else:
        return theta, r


def unrotate_vect(arr_xy, angles=None, reproject=True):
    if angles is None:
        angles = np.linspace(0, 360, arr_xy.shape[1], endpoint=False)[1:]
    elif isinstance(angles, int):
        angles = np.linspace(0, 360, angles)[1:]

    x, y = arr_xy
    x = unrotate(x, angles)
    y = unrotate(y, angles)

    z = x + 1j * y
    angle_offset = np.concatenate([[0], angles])
    while angle_offset.ndim < z.ndim:
        angle_offset = np.expand_dims(angle_offset, -1)
    theta = (np.angle(z, deg=True) - angle_offset)
    r = np.abs(z)
    if reproject:
        theta *= np.pi / 180
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y
    else:
        return theta, r


def simplify_angle(angles, mod=1, deg=True):
    mod = (360 if deg else 2*np.pi)/mod
    angles = np.mod(angles, mod)
    angles = np.stack([angles, angles-mod])
    a_idx = np.argmin(np.abs(angles), axis=0)
    angles = np.take_along_axis(angles, np.expand_dims(a_idx, axis=0), axis=0).squeeze(0)
    return angles


def repr_pi_fraction(num, den):
    if den == 0:
        raise ZeroDivisionError
    if num == 0:
        return "0"

    gcd = np.gcd(num, den)
    sign = "" if num*den>0 else "-"
    num = abs(num)//gcd
    den = abs(den)//gcd

    if den > 1:
        if num > 1:
            return sign + "\\dfrac{%i\\pi}{%i}" % (num, den)
        else:
            return sign + "\\dfrac{\\pi}{%i}" % den
    else:
        if num > 1:
            return sign + "%s\\pi" % num
        else:
            return sign + "\\pi"


def clip_pad_center(array, shape, pad_mode='constant', broadcastable=False, **kwargs):
    s = array.shape
    h, w = shape[-2:]

    if s[-2] == 1 and broadcastable:
        y0 = 0
        y1 = 0
        h = 1
        yodd = 0
    else:
        y0 = (s[-2]-h)//2
        y1 = 0
        yodd = (h-s[-2]) % 2
        if y0 < 0:
            y1 = -y0
            y0 = 0

    if s[-1] == 1 and broadcastable:
        x0 = 0
        x1 = 0
        w = 1
        xodd = 0
    else:
        x0 = (s[-1]-w)//2
        x1 = 0
        xodd = (w-s[-1]) % 2
        if x0 < 0:
            x1 = -x0
            x0 = 0

    tensor = array[..., y0:y0+h, x0:x0+w]
    if x1 or y1:
        tensor = np.pad(tensor, (y1-yodd, y1, x1-xodd, x1), mode=pad_mode, **kwargs)
    return tensor


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
        return  max(0., vmin), vmax

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
