from __future__ import annotations

import abc
from typing import Dict, List, Literal, Mapping, Optional, Self, Tuple, Type

import numpy as np
import numpy.typing as npt

from fundus_toolkits.utils.geometric import Point, Rect
import torch

from ..utils import if_none
from ..utils.numpy import GAUSSIAN_KERNEL_5x5, np_interp_bilinear
from ..utils.safe_import import import_cv2
from ..utils.typing import Float2DArray, Float3DArray
from ..utils.cpp_extensions.fvt_cpp import inverse_displacement, vec_bilinear_interpolate


def _np_short_str(arr: npt.NDArray[np.floating]) -> str:
    if arr.ndim == 2:
        return "[" + "| ".join([" ".join(f"{v:.2f}" for v in row) for row in arr]) + "]"
    elif arr.ndim == 1:
        return "[" + " ".join(f"{v:.2f}" for v in arr) + "]"
    return str(arr)


class FundusProjection(abc.ABC):
    @classmethod
    def identity(cls) -> FundusProjection:
        """
        Returns the identity projection model.

        Returns
        -------
        projection : FundusProjection
            The identity projection model.
        """
        return IdentityProjection()

    @classmethod
    def fit(cls, src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating]) -> Tuple[Self, float]:
        """
        Fits a projection model to map points from ``src`` to ``dst``.

        Parameters
        ----------
        src : npt.NDArray[np.floating]
            The source points coordinates (N x 2) where N is the number of points.

        dst : npt.NDArray[np.floating]
            The destination points coordinates (N x 2).

        Returns
        -------
        projection : Self
            The fitted projection model.

        error : float
            The mean square error of the fitted model.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement the 'fit' method")

    @classmethod
    def fit_to_projection(
        cls,
        src: npt.NDArray[np.floating],
        dst: npt.NDArray[np.floating],
        projection: Type[Self] | Dict[int, Type[Self]],
    ) -> Tuple[Self, float]:
        """Fits a given projection model to map points from ``src`` to ``dst``.

        Parameters
        ----------
        src : npt.NDArray[np.floating]
            The source points coordinates (N, 2) where N is the number of points.

        dst : npt.NDArray[np.floating]
            The destination points coordinates (N, 2).

        projection : Type[Self] | Dict[int, Type[Self]]
            The projection model to fit. If a dictionary is provided, the key is the minimum number of inliers required to use the corresponding projection.

        Returns
        -------
        Tuple[Self, float]
            The fitted projection model and the mean square error of the fitted model.

        """  # noqa: E501

        if isinstance(projection, Mapping):
            projection = {k: projection[k] for k in sorted(projection.keys(), reverse=True)}

            n_inliers = src.shape[0]
            proj = None
            for k, p in projection.items():
                proj = p
                if k <= n_inliers:
                    break
            else:
                raise ValueError("No projection model matches the number of inliers")
            return proj.fit(src, dst)
        elif issubclass(projection, FundusProjection):
            return projection.fit(src, dst)

        raise ValueError("projection must be a projection model or a dictionary of projection models")

    def compose(self, T1: FundusProjection) -> FundusProjection:
        """
        Composes this projection model with another one.

        Parameters
        ----------
        T1 : FundusProjection
            The other projection model to compose with.

        Returns
        -------
        T : FundusProjection
            The composed projection model: T = self @ T1.
        """
        return ProjectionComposition.simplify_composition(self, T1)

    def invert(self) -> FundusProjection:
        """
        Inverts this projection model.

        Returns
        -------
        T : Self
            The inverted projection model: T
        """
        return ProjectionInverse(self)

    @property
    def is_exact(self) -> bool:
        """
        Whether this projection model is exact (it doesn't provide an approximation for example using Newton algorithm).
        """
        return True

    @property
    def is_inverse_exact(self) -> bool:
        """
        Whether the inverse of this projection model is exact.
        (I.e. it doesn't provide an approximation for example using Newton algorithm).
        """
        return True

    @abc.abstractmethod
    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Transforms a set of points with this projection model.

        Parameters
        ----------
        src : npt.NDArray[np.floating]
            The source points coordinates (N x 2) where N is the number of points.

        Returns
        -------
        dst : npt.NDArray[np.floating]
            The transformed points coordinates (N x 2).
        """
        pass

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """
        Transforms a set of points with the inverse of this projection model.

        Parameters
        ----------
        dst : npt.NDArray[np.floating]
            The destination points coordinates (N x 2) where N is the number of points.

        Returns
        -------
        src : npt.NDArray[np.floating]
            The source points coordinates (N x 2).
        """
        invert_t = self.invert()
        if isinstance(invert_t, ProjectionInverse):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement the 'transform_inverse' nor the 'invert' methods"
            )
        return invert_t.transform(dst)

    def transform_domain(self, moving_domain: Rect) -> Rect:
        """
        Transforms a domain with this projection model.

        Parameters
        ----------
        src_domain : Rect
            The source domain to transform

        Returns
        -------
        Rect
            The transformed domain.
        """
        corners = self.transform(np.array(moving_domain.corners()))
        return Rect.from_points(tuple(np.amin(corners, axis=0)), tuple(np.amax(corners, axis=0))).to_int()

    def inverse_transform_domain(self, fixed_domain: Rect) -> Rect:
        """
        Transforms a domain with the inverse of this projection model.

        Parameters
        ----------
        dst_domain : Rect
            The destination domain to transform

        Returns
        -------
        Rect
            The transformed domain.
        """
        corners = self.transform_inverse(np.array(fixed_domain.corners()))
        return Rect.from_points(tuple(np.amin(corners, axis=0)), tuple(np.amax(corners, axis=0))).to_int()

    def quadratic_error(
        self, src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating], mean: bool = False
    ) -> npt.NDArray[np.floating] | float:
        """
        Calculates the quadratic error of the projection model when mapping points from ``src`` to ``dst``.

        Parameters
        ----------
        src : npt.NDArray[np.floating]
            The source points coordinates (N x 2) where N is the number of points.

        dst : npt.NDArray[np.floating]
            The destination points coordinates (N x 2).

        mean : bool, optional
            Whether to return the mean error. By default False.

        Returns
        -------
        error : npt.NDArray[np.floating] | float
            The quadratic error of each point or the mean error if ``mean`` is True.
        """
        errors = np.sum((dst - self.transform(src)) ** 2, axis=1)
        return np.mean(errors) if mean else errors

    def warp(
        self,
        src_img: npt.NDArray[np.uint8 | np.float32],
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Tuple[npt.NDArray[np.uint8 | np.float32], Rect]:
        """
        Warps an image using this projection model.

        Parameters
        ----------
        src_img : npt.NDArray[np.uint8] | npt.NDArray[np.float32]
            The source image to warp. The image must be cv2 compatible: shape=(H x W [x C]) and dtype=np.uint8|np.float32.

        src_top_left : Point | tuple[int, int]
            The top-left corner of the source image domain. The ``src_domain`` is defined as a Rect with this top-left corner and the size of the source image.

        warped_domain : Rect | Literal["full", "same"], optional
            The domain of the destination image.
            - "full": the destination domain is computed by transforming ``src_domain``;
            - "same": the destination domain is the same as ``src_domain``;
            - or any Rect manually defining the requested destination domain.

        Returns
        -------
        dst_img : npt.NDArray[np.uint8] | npt.NDArray[np.float32]
            The warped image.

        warped_domain : Rect
            The domain of the warped image.
        """  # noqa: E501
        cv2 = import_cv2()

        warped_domain = self.warped_domain(src_img, src_top_left, warped_domain)

        yy, xx = np.mgrid[warped_domain.slice()]
        dst_yx = np.column_stack((yy.ravel(), xx.ravel()))
        dst_yx -= Point(*src_top_left).numpy()[None, :]  # TODO: why is this not after transform_inverse?
        src_map = self.transform_inverse(dst_yx).reshape(warped_domain.shape + (2,))
        src_map = src_map.astype(np.float32)[..., ::-1]

        def warp(img):
            return cv2.remap(img, src_map, None, cv2.INTER_LINEAR)

        return warp(src_img) if isinstance(src_img, np.ndarray) else [warp(_) for _ in src_img], warped_domain

    def warped_domain(
        self,
        src_img: npt.NDArray,
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Rect:
        """
        Computes the domain of the warped image using this projection model.

        Parameters
        ----------
        src_img : npt.NDArray
            The source image to warp. Only the shape of the image is used.

        src_top_left : Point | tuple[int, int]
            The top-left corner of the source image domain. The ``src_domain`` is defined as a Rect with this top-left corner and the size of the source image.

        warped_domain : Rect | Literal["full", "same"], optional
            The domain of the destination image.
            - "full": the destination domain is computed by transforming ``src_domain``;
            - "same": the destination domain is the same as ``src_domain``;
            - or any Rect manually defining the requested destination domain.

        Returns
        -------
        warped_domain : Rect
            The domain of the warped image.
        """  # noqa: E501
        src_domain = Rect.from_size((src_img.shape[0], src_img.shape[1])).translate(*src_top_left)
        if warped_domain == "full":
            warped_domain = self.transform_domain(src_domain)
        elif warped_domain == "same":
            warped_domain = src_domain
        return warped_domain

    def select_warped_region[T: np.generic](
        self,
        src_img: npt.NDArray[T],
        src_top_left: Point | tuple[int, int],
        warped_domain: Rect | Literal["full", "same"],
    ) -> tuple[npt.NDArray[T], Rect]:
        """
        Selects a region to warp from an image using this projection model.

        Parameters
        ----------
        src_img : npt.NDArray[T]
            The source image to select the region from of shape (H, W[, C]).

        src_top_left : Point | tuple[int, int]
            The position of the top-left corner of the source image.

        warped_domain : Rect | Literal["full", "same"]
            The domain of the region to select.

        Returns
        -------
        src_img_to_warp : npt.NDArray[T]
            The selected region from the source image to warp.
        """
        src_domain = Rect.from_size((src_img.shape[0], src_img.shape[1])).translate(*src_top_left)
        if warped_domain == "full":
            warped_domain = self.transform_domain(src_domain)
        elif warped_domain == "same":
            warped_domain = src_domain

        inv_warped_domain = self.inverse_transform_domain(warped_domain).translate(-src_top_left[0], -src_top_left[1])
        return inv_warped_domain.crop_pad_image(src_img, channel_last=True), warped_domain


class ProjectionComposition(FundusProjection):
    def __init__(self, *Ts: FundusProjection) -> None:
        self.Ts = Ts
        super().__init__()

    def __repr__(self) -> str:
        return f"ProjectionComposition({', '.join(repr(T) for T in self.Ts)})"

    def __str__(self) -> str:
        return " @ ".join(str(T) for T in self.Ts)

    @staticmethod
    def simplify_composition(*Ts: FundusProjection) -> FundusProjection:
        expanded_transforms: list[FundusProjection] = []
        for T in Ts:
            if isinstance(T, ProjectionComposition):
                expanded_transforms.extend(T.Ts)
            elif not isinstance(T, IdentityProjection):
                expanded_transforms.append(T)
        Ts_ = list(expanded_transforms)

        simplified = True
        while simplified:
            simplified = False
            i = 0
            while i < len(Ts_) - 1:
                T1, T2 = Ts_[i], Ts_[i + 1]
                if (isinstance(T1, ProjectionInverse) and T1.T is T2) or (
                    isinstance(T2, ProjectionInverse) and T2.T is T1
                ):
                    simplified = True
                    del Ts_[i + 1]
                    del Ts_[i]
                else:
                    i += 1

        if not Ts_:
            return IdentityProjection()
        if len(Ts_) == 1:
            return Ts_[0]
        return ProjectionComposition(*Ts_)

    def compose(self, T1: FundusProjection) -> FundusProjection:
        if isinstance(T1, ProjectionComposition):
            return ProjectionComposition.simplify_composition(*self.Ts, *T1.Ts)
        return ProjectionComposition.simplify_composition(*self.Ts, T1)

    @property
    def is_exact(self) -> bool:
        return all(T.is_exact for T in self.Ts)

    @property
    def is_inverse_exact(self) -> bool:
        return all(T.is_inverse_exact for T in self.Ts)

    def invert(self) -> Self:
        return type(self)(*(T.invert() for T in reversed(self.Ts)))

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        for T in self.Ts:
            src = T.transform(src)
        return src

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        for T in reversed(self.Ts):
            dst = T.transform_inverse(dst)
        return dst


class ProjectionInverse(FundusProjection):
    def __init__(self, T: FundusProjection) -> None:
        self.T = T
        super().__init__()

    def __repr__(self) -> str:
        return f"ProjectionInverse({self.T})"

    def __str__(self) -> str:
        return f"Inv[{self.T}]"

    def invert(self) -> FundusProjection:
        return self.T

    @property
    def is_exact(self) -> bool:
        return self.T.is_inverse_exact

    @property
    def is_inverse_exact(self) -> bool:
        return self.T.is_exact

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self.T.transform_inverse(src)

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self.T.transform(dst)


class IdentityProjection(FundusProjection):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "IdentityProjection()"

    def __str__(self) -> str:
        return "I"

    def invert(self) -> Self:
        return type(self)()

    def compose(self, T1: FundusProjection) -> FundusProjection:
        return T1

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return src

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return dst

    def warp(
        self,
        src_img: npt.NDArray[np.uint8 | np.float32],
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Tuple[npt.NDArray[np.uint8 | np.float32], Rect]:
        return self.select_warped_region(src_img, src_top_left, warped_domain)


class Translation(FundusProjection):
    """A projection model that translates points by a given vector.

    Example
    -------
    >>> T = Translation((2, 1))
    >>> src = np.array([[0, 0], [-2, -1]])
    >>> dst = T.transform(src)
    >>> dst
    array([[2,  1],
           [0,  0]])

    >>> img = np.zeros((5, 5), dtype=np.uint8)
    >>> img[0,0] = 255
    >>> warped_img, warped_domain = T.warp(img)
    >>> np.all(warped_img == img)
    np.True_
    >>> warped_domain
    Rect(y=2, x=1, h=5, w=5)

    >>> warped_img, warped_domain = T.warp(img, src_top_left=(1,1), warped_domain="same")
    >>> np.argwhere(warped_img==255)
    array([[2, 1]])
    >>> warped_domain
    Rect(y=1, x=1, h=5, w=5)

    >>> warped_img, warped_domain = T.warp(img, warped_domain=Rect(y=1, x=1, h=2, w=2))
    >>> warped_img
    array([[  0,   0],
           [255, 0]], dtype=uint8)
    >>> warped_domain
    Rect(y=1, x=1, h=2, w=2)

    >>> np.all(T.transform_inverse(dst) == src) and np.all(T.invert().transform(dst) == src)
    np.True_

    """

    def __init__(self, t: npt.NDArray[np.floating] | tuple[float, float]) -> None:
        self.t = np.asarray(t)
        assert self.t.shape == (2,), "t must be a 2D vector"
        super().__init__()

    def __repr__(self) -> str:
        return f"Translation(t={self.t})"

    def __str__(self) -> str:
        return f"Trans(t={_np_short_str(self.t)})"

    def invert(self) -> Self:
        return self.__class__(-self.t)

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return src + self.t

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return dst - self.t

    @classmethod
    def identity(cls) -> FundusProjection:
        return cls(np.array([0.0, 0.0]))

    @classmethod
    def fit(cls, src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating]) -> Tuple[Self, float]:
        src, dst = np.asarray(src), np.asarray(dst)
        assert src.ndim == 2 and src.shape[1] == 2, "src must be a 2D array of 2D coordinates"
        assert src.shape == dst.shape, "src and dst must have the same shape"
        t = np.mean(dst - src, axis=0)
        return cls(t), np.mean(np.sum((dst - (src + t)) ** 2, axis=1))

    def warp(
        self,
        src_img: npt.NDArray[np.uint8 | np.float32],
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Tuple[npt.NDArray[np.uint8 | np.float32], Rect]:
        return self.select_warped_region(src_img, src_top_left, warped_domain)


class FlipProjection(FundusProjection):
    def __init__(self, center: tuple[int, int], horizontal: bool = True, vertical: bool = False) -> None:
        """
        A projection model that flips points horizontally and/or vertically around a center point.

        Parameters
        ----------
        center : tuple[int, int]
            The center point (y, x) around which to flip the points.
        horizontal : bool, optional
            Whether to flip points horizontally. Default is True.
        vertical : bool, optional
            Whether to flip points vertically. Default is False.
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.center = center
        super().__init__()

    def __repr__(self) -> str:
        return f"FlipProjection(center={self.center}, horizontal={self.horizontal}, vertical={self.vertical})"

    def __str__(self) -> str:
        flips = []
        if self.horizontal:
            flips.append("H")
        if self.vertical:
            flips.append("V")
        return "Flip(" + ",".join(flips) + ")"

    def invert(self) -> Self:
        return self.__class__(self.center, self.horizontal, self.vertical)

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        dst = np.asarray(src, copy=True)
        if self.horizontal:
            dst[:, 1] = 2 * self.center[1] - dst[:, 1]
        if self.vertical:
            dst[:, 0] = 2 * self.center[0] - dst[:, 0]
        return dst

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self.transform(dst)

    def warp(
        self,
        src_img: npt.NDArray[np.uint8 | np.float32],
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Tuple[npt.NDArray[np.uint8 | np.float32], Rect]:
        dst_img, warped_domain = self.select_warped_region(src_img, src_top_left, warped_domain)

        if self.horizontal:
            dst_img = np.fliplr(dst_img)
        if self.vertical:
            dst_img = np.flipud(dst_img)
        return dst_img, warped_domain


class ResizeTranslateProjection(FundusProjection):
    def __init__(self, r: float, t: Optional[npt.NDArray[np.floating]] = None) -> None:
        assert r > 0, "r must be positive"
        assert t is None or t.shape == (2,), "t must be a 2D vector"
        self.r = r
        self.t = if_none(t, np.zeros(2))
        super().__init__()

    @classmethod
    def translate_resize(self, t: npt.NDArray[np.floating], r: float) -> Self:
        return self(r, t * r)

    @classmethod
    def fit(cls, src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating]) -> Tuple[Self, float]:
        src, dst = np.asarray(src), np.asarray(dst)
        assert src.ndim == 2 and src.shape[1] == 2, "src must be a 2D array of 2D coordinates"
        assert src.shape == dst.shape, "src and dst must have the same shape"
        N = src.shape[0]
        A = np.zeros((2 * N, 3))
        b = np.zeros((2 * N,))
        A[0::2, 0] = src[:, 0]
        A[0::2, 1] = 1
        A[1::2, 0] = src[:, 1]
        A[1::2, 2] = 1
        b[0::2] = dst[:, 0]
        b[1::2] = dst[:, 1]
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        r = x[0]
        t = x[1:3]
        error = np.sum((dst - (r * src + t)) ** 2, axis=1)
        return cls(r, t), np.mean(error)

    def __repr__(self) -> str:
        return f"ResizeTranslateProjection(r={self.r}, t={self.t})"

    def __str__(self) -> str:
        return f"ResizeTranslateProjection(r={self.r}, t={_np_short_str(self.t)})"

    def invert(self) -> Self:
        return self.__class__(1 / self.r, -self.t / self.r)

    def compose(self, T1: FundusProjection) -> FundusProjection:
        if isinstance(T1, ResizeTranslateProjection):
            r = self.r * T1.r
            t = self.r * T1.t + self.t
            return self.__class__(r, t)
        return super().compose(T1)

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self.r * src + self.t

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return (dst - self.t) / self.r

    def warp(
        self,
        src_img: npt.NDArray[np.uint8 | np.float32],
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Tuple[npt.NDArray[np.uint8 | np.float32], Rect]:
        cv2 = import_cv2()

        warped_region, warped_domain = self.select_warped_region(src_img, src_top_left, warped_domain)
        return cv2.resize(warped_region, dsize=warped_domain.shape, fx=self.r, fy=self.r), warped_domain  # type: ignore


class AffineProjection(FundusProjection):
    def __init__(self, R: npt.NDArray[np.floating], t: npt.NDArray[np.floating]) -> None:
        assert R.shape == (2, 2) and t.shape == (2,), "R must be a 2x2 matrix and t must be a 2D vector"
        self.R = R
        self.t = t
        super().__init__()

    def __repr__(self) -> str:
        return f"AffineProjection(R={self.R}, t={self.t})"

    def __str__(self) -> str:
        return f"Affine(R={_np_short_str(self.R)}, t={_np_short_str(self.t)})"

    @classmethod
    def rotate(cls, theta: float, center: npt.NDArray[np.floating] | tuple[float, float] = (0, 0)) -> Self:
        """Create an affine transformation that rotates by theta and translates by t.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees. Positive values rotate clockwise.
        center : npt.NDArray[np.floating] | tuple[float, float]
            Center of rotation.

        Returns
        -------
        AffineProjection
            The corresponding affine transformation.
        """
        theta = np.deg2rad(theta)
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        center = np.asarray(center, dtype=np.float64)
        t = center - R @ center
        return cls(R, t)

    @classmethod
    def fit(cls, src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating]) -> Tuple[Self, float]:
        src, dst = np.asarray(src), np.asarray(dst)
        assert src.ndim == 2 and src.shape[1] == 2, "src must be a 2D array of 2D coordinates"
        assert src.shape == dst.shape, "src and dst must have the same shape"
        src = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
        X, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
        R, t = X[:2].T, X[2]
        error = np.sum((dst - src @ X) ** 2, axis=1)
        return cls(R, t), np.mean(error)

    def compose(self, T1: FundusProjection) -> FundusProjection:
        if isinstance(T1, AffineProjection):
            return self.__class__(self.R @ T1.R, self.R @ T1.t + self.t)
        return super().compose(T1)

    def invert(self) -> Self:
        X = np.concatenate((self.R, self.t[:, None]), axis=1)
        X = np.concatenate((X, [[0, 0, 1]]), axis=0)
        X_inv = np.linalg.inv(X)
        R_inv, t_inv = X_inv[:2, :2], X_inv[:2, 2]
        return self.__class__(R_inv, t_inv)

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        src = np.asarray(src)
        return (self.R @ src.T + self.t[:, None]).T

    def warp(
        self,
        src_img: npt.NDArray[np.uint8]
        | npt.NDArray[np.float32]
        | List[npt.NDArray[np.uint8]]
        | List[npt.NDArray[np.float32]],
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Tuple[
        npt.NDArray[np.uint8] | npt.NDArray[np.float32] | List[npt.NDArray[np.uint8]] | List[npt.NDArray[np.float32]],
        Rect,
    ]:
        cv2 = import_cv2()

        warped_domain = self.warped_domain(src_img, src_top_left, warped_domain)
        src_top_left = Point(*src_top_left)
        t = self.t - warped_domain.top_left + src_top_left
        M = np.concatenate((self.R[::-1, ::-1], t[::-1, None]), axis=1)

        def warp(img):
            return cv2.warpAffine(img, M, warped_domain.size.xy, flags=cv2.INTER_LINEAR)

        return warp(src_img) if isinstance(src_img, np.ndarray) else [warp(_) for _ in src_img], warped_domain


class QuadraticProjection(FundusProjection):
    """
    A quadratic projection model that maps points from a source to a destination using a quadratic transformation.

    The transformation is defined as:
        dst = [Q, R, t] @ [src.y², src.x², src.x*src.y, src.y, src.x, 1].T

    """

    def __init__(self, Q: npt.NDArray[np.floating], R: npt.NDArray[np.floating], t: npt.NDArray[np.floating]) -> None:
        Q, R, t = np.asarray(Q), np.asarray(R), np.asarray(t)
        assert Q.shape == (2, 3) and R.shape == (2, 2) and t.shape == (2,), (
            "Q must be a 2x3 matrix, R must be a 2x2 matrix and t must be a 2D vector"
        )
        self.Q = Q
        self.R = R
        self.t = t
        self._inverse_transform: Literal[False] | None | QuadraticProjection = False
        super().__init__()

    def __repr__(self) -> str:
        return f"QuadraticProjection(Q={self.Q}, R={self.R}, t={self.t})"

    def __str__(self) -> str:
        return f"Quadratic(Q={_np_short_str(self.Q)}, R={_np_short_str(self.R)}, t={_np_short_str(self.t)})"

    @property
    def is_inverse_exact(self) -> bool:
        return False

    @classmethod
    def fit(cls, src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating]) -> Tuple[Self, float]:
        src_y = src[:, 0]
        src_x = src[:, 1]
        src_ = np.stack((src_y**2, src_x**2, src_x * src_y, src_y, src_x, np.ones((src.shape[0],))), axis=1)
        X, _, _, _ = np.linalg.lstsq(src_, dst, rcond=None)
        Q, R, t = X[:3].T, X[3:5].T, X[5]

        # if np.any(abs(Q) > 1e-5):
        error = np.sum((dst - src_ @ X) ** 2, axis=1)
        return cls(Q, R, t), np.mean(error)
        # else:
        #    T = AffineProjection(R.T, t)
        #    return T, np.mean(T.quadratic_error(src, dst))

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        src = np.asarray(src)
        src_y, src_x = src[:, 0], src[:, 1]
        src_yy_xx_yx = np.stack((src_y**2, src_x**2, src_x * src_y), axis=1)
        return (self.Q @ src_yy_xx_yx.T + self.R @ src.T + self.t[:, None]).T

    def jacobian(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        src = np.asarray(src)
        return self.R[None, :, :] + (self.Q[None, :, 2, None] + 2 * self.Q[None, :, :2]) * src[:, None, :]

    def _eval_inverse_transform(self) -> QuadraticProjection | None:
        # Sample points to estimate the inverse transformation
        src = np.mgrid[0:1000:100, 0:1000:100].reshape(2, -1).T
        dst = self.transform(src)
        invT, error = QuadraticProjection.fit(dst, src)
        invT._inverse_transform = self
        return None if error > 1 else invT

    def transform_inverse_newton(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        # initial guess using only the affine part
        x = AffineProjection(self.R, self.t).transform_inverse(dst)

        NITERS = 20
        TOL = 1

        # Newton's method
        for _ in range(NITERS):
            p = self.transform(x)
            t = dst - p
            if np.all(np.linalg.norm(t, axis=-1) < TOL):
                break

            J = self.jacobian(x)
            dx = np.linalg.solve(J, t)
            x += dx
            if np.all(np.linalg.norm(dx, axis=-1) < TOL):
                break

        return x

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        if self._inverse_transform is False:
            self._inverse_transform = self._eval_inverse_transform()
        if self._inverse_transform is None:
            return self.transform_inverse_newton(dst)
        return self._inverse_transform.transform(dst)


class ElasticProjection(FundusProjection):
    def __init__(self, displacement: npt.NDArray[np.float32], reversed: bool = False) -> None:
        assert displacement.ndim == 3 and displacement.shape[2] == 2, "displacement must be a 2D map of 2D vectors"
        self.displacement = np.asarray(displacement, dtype=np.float32)

        self.reversed = reversed
        super().__init__()

    def __repr__(self) -> str:
        return f"ElasticProjection(displacement: {self.displacement.shape})"

    def __str__(self) -> str:
        return "Elastic"

    @classmethod
    def random(
        cls,
        shape: Tuple[int, int],
        displacement_std: float = 10,
        smoothing_size: Optional[float] = 2,
        *,
        rng: Optional[np.random.Generator] = None,
        reversed: bool = True,
    ) -> Self:
        if rng is None:
            rng = np.random.default_rng()
        if smoothing_size is not None and smoothing_size <= 0:
            smoothing_size = None
        subsampling = smoothing_size // 2 if smoothing_size is not None else 1
        disp_map = rng.normal(0, displacement_std, size=(int(shape[0] // subsampling), int(shape[1] // subsampling), 2))
        disp_map = disp_map.astype(np.float32)
        if smoothing_size is not None:
            cv2 = import_cv2()

            kernel = GAUSSIAN_KERNEL_5x5
            disp_map_ = np.empty(shape + (2,), dtype=disp_map.dtype)
            for i in range(2):
                smooth_disp = cv2.filter2D(disp_map[..., i], -1, kernel, borderType=cv2.BORDER_REPLICATE)
                disp_map_[..., i] = cv2.resize(smooth_disp, dsize=shape[::-1], interpolation=cv2.INTER_LINEAR)
            disp_map = disp_map_

        return cls(disp_map, reversed=reversed)

    @classmethod
    def fit(cls, src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating]) -> Tuple[Self, float]:
        raise NotImplementedError("ElasticProjection does not implement the 'fit' method")

    def invert(self) -> Self:
        return type(self)(self.displacement, reversed=not self.reversed)

    def transform(self, src: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self._transform(self.displacement, src, reversed=self.reversed)

    def transform_inverse(self, dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        return self._transform(self.displacement, dst, reversed=not self.reversed)

    @classmethod
    def _transform(
        cls, displacement: npt.NDArray[np.floating], src: npt.NDArray | None = None, reversed: bool = False
    ) -> npt.NDArray[np.floating]:
        disp_t = torch.from_numpy(displacement)

        if not reversed:

            def interp_displacement(pos: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
                pos_t = torch.from_numpy(pos).reshape(-1, 2)
                return vec_bilinear_interpolate(disp_t, pos_t).numpy().reshape(pos.shape)

            if src is None:
                return np.indices(displacement.shape[:2]).transpose(1, 2, 0) + displacement
            elif np.issubdtype(src.dtype, np.integer):
                src[..., 0] = np.clip(src[..., 0], 0, displacement.shape[0] - 1)
                src[..., 1] = np.clip(src[..., 1], 0, displacement.shape[1] - 1)
                return src + displacement[src[..., 0], src[..., 1]]
            else:
                return src + interp_displacement(src.astype(np.float32))

        if src is None:
            src = np.indices(displacement.shape[:2]).transpose(1, 2, 0)

        src_t = torch.from_numpy(src.astype(np.float32)).reshape(-1, 2)

        # Inverse displacement field through fixed-point iteration
        inv_d = inverse_displacement(disp_t, src_t, 50, 0.5).numpy()

        return src + inv_d.reshape(src.shape)

    def warp(
        self,
        src_img: npt.NDArray[np.uint8 | np.float32],
        src_top_left: Point | tuple[int, int] = (0, 0),
        warped_domain: Rect | Literal["full", "same"] = "full",
    ) -> Tuple[npt.NDArray[np.uint8 | np.float32], Rect]:
        cv2 = import_cv2()

        warped_domain = self.warped_domain(src_img, src_top_left, warped_domain)
        src_remap = self._transform(
            self.displacement, src=warped_domain.grid_indices(), reversed=not self.reversed
        ).astype(np.float32)
        return cv2.remap(src_img, src_remap[..., ::-1], None, cv2.INTER_LINEAR), warped_domain  # type: ignore


def ransac_fit_projection(
    fix: npt.NDArray[np.floating],
    moving: npt.NDArray[np.floating],
    sampling_probability: Optional[npt.NDArray[np.floating]] = None,
    initial_projection: Type[FundusProjection] = AffineProjection,
    final_projection: Optional[Type[FundusProjection] | Dict[int, Type[FundusProjection]]] = None,
    *,
    n: int = 4,
    initial_inliers_tolerance: float = 5,
    min_initial_inliers: int | float = 0.5,
    final_inliers_tolerance: Optional[float] = None,
    max_iterations: int = 300,
    early_stop_mean_error: float = 1,
    early_stop_min_inliers: int = 0.5,
    rng: np.random.Generator = None,
) -> Tuple[FundusProjection, float, npt.NDArray[np.integer]]:
    """
    Estimates a 2D transformation matrix that maps points from ``src`` to ``dst`` using the RANSAC algorithm.

    Parameters
    ----------
        fix: npt.NDArray[np.floating]
            Coordinates of the fix points (N x 2) where N is the number of points.

        moving: npt.NDArray[np.floating]
            Coordinates of the moving points (N x 2) where N is the number of points.

        sampling_probability: Optional[npt.NDArray[np.floating]]
            Probability of sampling each point. If None, all points are sampled with the same probability.

        initial_projection: Type[FundusProjection], optional
            The type of projection to use for the initial estimation.

        final_projection: Type[FundusProjection] | Dict[int, Type[FundusProjection]], optional
            The type of projection to use for the final estimation.

            - If a dictionary is provided, the key is the minimum number of inliers required to use the corresponding projection.
            - If None (by default), the initial projection is used for the final estimation.

        n: int, optional
            Number of points to sample for each iteration.

        initial_inliers_tolerance: float, optional
            Maximum distance between the transformed points and the destination points to consider them as inliers.

        min_initial_inliers: int | float, optional
            Minimum number of inliers required to consider the transformation as valid. If a float, it is interpreted as a ratio of the total number of points.

        final_inliers_tolerance: float, optional
            Maximum distance between the transformed points and the destination points to consider them as inliers in the final estimation. The returned transformation is the one with the most of such inliers.

        max_iterations: int, optional
            Maximum number of iterations.

        early_stop_mean_error: float, optional
            Mean distance under which the algorithm should stop early.

        early_stop_min_inliers: int | float, optional
            Minimum number of inliers required to stop the algorithm early. If a float, it is interpreted as a ratio of the total number of points.

        rng: np.random.Generator, optional
            Random number generator.

    Returns
    -------
        T: FundusProjection
            The best transformation of type ``final_projection`` found to map the moving points to the fix points.

        error: float
            Mean distance of the best transformation.

        inliers: npt.NDArray[np.integer]
            Indices of the points that are considered inliers.

    Raises
    ------
        ValueError
            If no transformation matches the criteria
    """  # noqa: E501
    assert moving.ndim == 2 and moving.shape[1] == 2, "moving must be a 2D array of 2D coordinates"
    assert moving.shape == fix.shape, "moving and fix must have the same shape"
    N = moving.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    if isinstance(final_projection, Mapping):
        final_projection[-1] = initial_projection
    elif final_projection is None:
        final_projection = initial_projection

    initial_inliers_tolerance = initial_inliers_tolerance**2
    final_inliers_tolerance = (
        initial_inliers_tolerance if final_inliers_tolerance is None else final_inliers_tolerance**2
    )
    early_stop_mean_error = early_stop_mean_error**2

    if min_initial_inliers < 1:
        min_initial_inliers = int(min_initial_inliers * (N - n))
    if early_stop_min_inliers < 1:
        early_stop_min_inliers = int(early_stop_min_inliers * (N - n))

    best_T = None
    best_mean_error = np.inf
    best_inliers = []

    for _ in range(max_iterations):
        # Sample n points
        if sampling_probability is not None:
            idx = rng.choice(N, n, replace=False, p=sampling_probability)
            idx = np.concatenate([idx, np.setdiff1d(np.arange(N), idx)])
        else:
            idx = np.arange(N)
            rng.shuffle(idx)

        # Estimate initial transformation
        iniT, _ = initial_projection.fit(moving[idx[:n]], fix[idx[:n]])

        # Apply initial transformation to all other points and calculate error
        errors = iniT.quadratic_error(moving[idx[n:]], fix[idx[n:]])

        # Check if initial transformation m is valid
        n_inliers = np.sum(errors < initial_inliers_tolerance) + n
        if n_inliers < max(min_initial_inliers + n, len(best_inliers)):
            continue

        # Optimize transformation using all inliers and the final projection
        inliers = np.concatenate([idx[:n], idx[n:][errors < initial_inliers_tolerance]])
        T, mean_error = FundusProjection.fit_to_projection(moving[inliers], fix[inliers], final_projection)
        errors = np.sum((fix - T.transform(moving)) ** 2, axis=1)

        inliers = np.where(errors < final_inliers_tolerance)[0]

        # Save transformation if it is better
        if len(inliers) > len(best_inliers) or mean_error < best_mean_error:
            best_T = T
            best_inliers = inliers
            best_mean_error = mean_error

            # Early stop if the error is below the tolerance
            if best_mean_error < early_stop_mean_error and len(best_inliers) >= early_stop_min_inliers:
                break
    else:
        # If the loop completes without early stopping, recompute the transformation using all best inliers
        T, mean_error = FundusProjection.fit_to_projection(moving[best_inliers], fix[best_inliers], final_projection)
        if mean_error < best_mean_error:
            best_T = T
            best_mean_error = mean_error

    if best_T is None:
        raise ValueError("RANSAC algorithm failed: no transformation matched the criteria")

    return best_T, np.sqrt(best_mean_error), best_inliers


########################################################################################################################
def fit_mse_affine_tranform(src: npt.NDArray[np.floating], dst: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Calculates the least-squares best-fit translation and rotation that maps corresponding points ``source`` to ``dest``.

    Return the transformation matrix X which solve: dst.T = T @ [src, 1].T

    Parameters
    ----------
    src : npt.NDArray[np.floating]
        Source points (N x m) where N is the number of points and m is the number of dimensions.

    dst : npt.NDArray[np.floating]
        Destination points (N x m).

    Returns
    -------
    T : npt.NDArray[np.floating]
        Homogeneous transformation matrix (m x m+1). E.g. for 2D points, the matrix is:
            [[cos(theta), -sin(theta), ty],
             [sin(theta),  cos(theta), tx]]

    """  # noqa: E501

    assert src.ndim == 2 and dst.ndim == 2, "src and dst must be 2D arrays"
    assert src.shape == dst.shape, "src and dst must have the same shape"
    N, m = src.shape

    src = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
    # dst = np.concatenate((dst, np.ones((dst.shape[0], 1))), axis=1)
    return np.linalg.lstsq(src, dst, rcond=None)[0].T


def apply_affine_transform(src: npt.NDArray[np.floating], T: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Applies a 2D affine transformation matrix to a set of points following the formula:
        ``dst.T = T @ [src, 1].T``.

    Parameters
    ----------
    src : npt.NDArray[np.floating]
        Source points (N x m) where N is the number of points and m is the number of dimensions.

    T : npt.NDArray[np.floating]
        Affine transformation matrix (m x m+1).

    Returns
    -------
    dst : npt.NDArray[np.floating]
        Transformed points (N x m).
    """
    src = np.asarray(src)
    assert src.ndim == 2, "src must be a 2D array"
    _, m = src.shape
    T_square = T[:, :m]
    dst = T_square @ src.T + T[:, m, None]
    return dst.T


def compose_affine_transforms(T1: npt.NDArray[np.floating], T2: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Composes two 2D affine transformation matrices.

    Parameters
    ----------
    T1 : npt.NDArray[np.floating]
        First affine transformation matrix (m x m+1).

    T2 : npt.NDArray[np.floating]
        Second affine transformation matrix (m x m+1).

    Returns
    -------
    T : npt.NDArray[np.floating]
        Composed affine transformation matrix (m x m+1).
    """
    assert T1.ndim == T2.ndim == 2, "T1 and T2 must be 2D arrays"
    assert T1.shape == T2.shape, "T1 and T2 must have the same shape"

    R1, t1 = T1[:, :-1], T1[:, -1]
    R2, t2 = T2[:, :-1], T2[:, -1]

    return np.concatenate((R1 @ R2, R1 @ t2[:, None] + t1[:, None]), axis=1)
