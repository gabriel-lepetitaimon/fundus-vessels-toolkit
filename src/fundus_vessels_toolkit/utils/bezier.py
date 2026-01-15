from __future__ import annotations, print_function

import itertools
from typing import Iterable, List, Literal, NamedTuple, Optional, Self, Tuple, overload

import numpy as np
import numpy.typing as npt
import torch

from fundus_toolkits.utils.geometric import Point
from fundus_vessels_toolkit.utils.numpy import as_1d_array
from fundus_vessels_toolkit.utils.typing import Float1DArray, Int2DArray

from ..utils.fundus_projections import FundusProjection
from .graph.measures import curve_tangent
from .math import intercept_segment
from .torch import autocast_torch


class BezierCubic(NamedTuple):
    p0: Point
    c0: Point
    c1: Point
    p1: Point

    def __str__(self) -> str:
        return f"BezierCubic(p0={self.p0}, c0={self.c0}, c1={self.c1}, p1={self.p1})"

    @classmethod
    def from_array(cls, curve: npt.ArrayLike) -> BezierCubic:
        curve = np.asarray(curve, dtype=float)
        assert curve.shape == (4, 2), "BezierCubic must be defined with a 2D array of shape (4, 2)"
        assert not np.isnan(curve).any(), "BezierCubic control points cannot be NaN"
        return cls(*[Point(float(p[0]), float(p[1])) for p in curve])

    def to_path(self, offset: Optional[Point] = None) -> str:
        if self.p0.is_nan() or self.c0.is_nan() or self.c1.is_nan() or self.p1.is_nan():
            return ""
        oy, ox = offset if offset is not None else (0, 0)
        return (
            f"M {self.p0.x - ox},{self.p0.y - oy} "
            f"C {self.c0.x - ox},{self.c0.y - oy} {self.c1.x - ox},{self.c1.y - oy} "
            f"{self.p1.x - ox},{self.p1.y - oy}"
        )

    def to_array(self) -> npt.NDArray[np.float64]:
        return np.array([self.p0, self.c0, self.c1, self.p1])

    def chord_length(self) -> float:
        return self.p0.distance(self.p1)

    def arc_length(self, fast_approximation=False):
        if fast_approximation:
            return (
                np.linalg.norm(self.p0 - self.p1)
                + np.linalg.norm(self.c0 - self.p0)
                + np.linalg.norm(self.c1 - self.c0)
                + np.linalg.norm(self.p1 - self.c1)
            ) / 2
        else:
            u = np.linspace(0, 1, int(self.arc_length(fast_approximation=True)) + 1, endpoint=True)
            bezier = self.to_array()
            return np.sum(np.linalg.norm(q(bezier, u[1:]) - q(bezier, u[:-1]), axis=1))

    def parametrize(
        self, yx_points: np.ndarray, initial_u: Optional[np.ndarray] = None, error=2, max_iteration=20
    ) -> npt.NDArray[np.float64]:
        bezier = self.to_array()
        u = chordLengthParameterize(yx_points) if initial_u is None else initial_u
        for _ in range(max_iteration):
            u = reparameterize(bezier, yx_points, u)
            maxError, splitPoint = computeMaxError(yx_points, bezier, u)
            if maxError < error**2:
                break
        return np.asarray(u)

    def projection(
        self,
        yx_points: np.ndarray,
        return_distance=False,
        return_u=False,
        fast_approximation=False,
        max_iteration=20,
        error=2,
    ) -> npt.NDArray[np.float64] | Tuple[npt.NDArray[np.float64], ...]:
        """Project a set of points on the Bezier curve.

        Parameters
        ----------
        yx_points : np.ndarray
            The points to project on the Bezier curve.

        return_distance : bool, optional
            If True, also return the distance between the projected points and the original points, by default False.

        return_u : bool, optional
            If True, also return the parameter u of the projected points on the Bezier curve, by default False.

        fast_approximation : bool, optional
            If True, use a fast approximation by projecting the points on the trapezoid defined by the control points, by default False.

        max_iteration : int, optional
            Maximum number of iterations of the Newton-Raphson algorithm, by default 20.
            If fast_approximation is True, this parameter is ignored.

        error : int, optional
            Target error in pixel for the projection stopping preemptively the Newton-Raphson algorithm, by default 2.
            If fast_approximation is True, this parameter is ignored.

        Returns
        -------
        projected_points : np.ndarray
            The projected points on the Bezier curve as a (n, 2) array.

        distance : np.ndarray
            If return_distance is True, also return the distance between the projected points and the original points as a (n,) array.

        u : np.ndarray
            If return_u is True, also return the parameter u of the projected points on the Bezier curve as a (n,) array.
        """  # noqa: E501

        yx_points = np.atleast_2d(yx_points).astype(float)
        assert yx_points.ndim == 2 and yx_points.shape[1] == 2, "yx_points must be a 2D array of shape (n, 2)"
        n = yx_points.shape[0]

        # Compute projection on the trapezoid defined by (p0, c0, c1, p1)
        p0, c0, c1, p1 = self.to_array()

        if np.all(p0 == c0) and np.all(p0 == c1) and np.all(p0 == p1):
            proj = np.full_like(yx_points, p0)
            out = (proj,)
            if return_distance:
                dist = np.linalg.norm(proj - yx_points, axis=1)
                out = out + (dist,)
            if return_u:
                out = out + (np.zeros(len(yx_points)),)
            return proj if len(out) == 1 else out

        if np.any(p0 != c0):
            p0c0_u = ((c0 - p0) / (p0c0norm := np.linalg.norm(c0 - p0)))[None, :]
            l_p0c0 = np.clip((yx_points - p0) @ p0c0_u.T, 0, p0c0norm)
            p0c0_proj = l_p0c0 * p0c0_u + p0[None, :]
        else:
            p0c0_proj, l_p0c0, p0c0norm = p0[None, :], np.zeros((1, n)), 0
        if np.any(c1 != c0):
            c0c1_u = ((c1 - c0) / (c0c1norm := np.linalg.norm(c1 - c0)))[None, :]
            l_c0c1 = np.clip((yx_points - c0) @ c0c1_u.T, 0, c0c1norm)
            c0c1_proj = l_c0c1 * c0c1_u + c0[None, :]
        else:
            c0c1_proj, l_c0c1, c0c1norm = c0[None, :], np.zeros((1, n)), 0
        if np.any(p1 != c1):
            p1c1_u = ((c1 - p1) / (p1c1norm := np.linalg.norm(c1 - p1)))[None, :]
            l_p1c1 = np.clip((yx_points - p1) @ p1c1_u.T, 0, p1c1norm)
            p1c1_proj = l_p1c1 * p1c1_u + p1[None, :]
        else:
            p1c1_proj, l_p1c1, p1c1norm = p1[None, :], np.zeros((1, n)), 0

        proj = np.stack([p0c0_proj, c0c1_proj, p1c1_proj], axis=0)
        dist = np.linalg.norm(proj - yx_points[None, :, :], axis=2)
        closest_segment = np.argmin(dist, axis=0)

        if not fast_approximation or return_u:
            u = np.stack([l_p0c0, l_c0c1 + p0c0norm, l_p1c1 + p0c0norm + c0c1norm], axis=0).squeeze(axis=2)
            u = np.take_along_axis(u, closest_segment[None, :], axis=0).squeeze(axis=0)
            u /= p0c0norm + c0c1norm + p1c1norm

        if fast_approximation:
            proj = np.take_along_axis(proj, closest_segment[None, :, None], axis=0).squeeze(axis=0)
            out = (proj,)
            if return_distance:
                dist = np.take_along_axis(dist, closest_segment[None, :], axis=0).squeeze(axis=0)
                out = out + (dist,)
            if return_u:
                out = out + (u,)
            return proj if len(out) == 1 else out
        else:
            bezier = self.to_array()
            for _ in range(20):
                u = newtonRaphsonRootFind(bezier, yx_points, u)
                maxError, splitPoint = computeMaxError(yx_points, bezier, u)
                if maxError < 2**2:
                    break
            u = np.clip(u, 0, 1)

            proj = self.evaluate(u)
            out = (proj,)
            if return_distance:
                dist = np.linalg.norm(proj - yx_points, axis=1)
                out = out + (dist,)
            if return_u:
                out = out + (u,)
            return proj if len(out) == 1 else out

    def intercept(
        self, origins: npt.ArrayLike, tangents: npt.ArrayLike, fast_approximation=False, half_line=True
    ) -> npt.NDArray[np.float64]:
        """Compute the intercept of the Bezier curve with a set of lines defined by their origins and tangents.

        Parameters
        ----------
        origins : np.array
            The origins of the lines as a (n, 2) array.

        tangents : np.array
            The tangents of the lines as a (n, 2) array.

        fast_approximation : bool, optional
            If True, use a fast approximation by intercepting the trapezoid defined by the control points instead of the true bezier curve, by default False.

        half_line : bool, optional
            If True, the intercept may only be on the half-line defined by the origin and the tangent, by default True.

        Returns
        -------
        np.array
            The intercept of the Bezier curve with the lines as a (n, 2) array. If the intercept does not exist, the point is set to NaN.
        """  # noqa: E501
        origins = np.atleast_2d(origins).astype(float)
        tangents = np.atleast_2d(tangents).astype(float)

        assert origins.ndim == 2 and origins.shape[1] == 2, "origins must be a 2D array of shape (n, 2)"
        assert tangents.ndim == 2 and tangents.shape[1] == 2, "tangents must be a 2D array of shape (n, 2)"

        # Compute the intercept of the Bezier curve with the trapezoid defined by (p0, c0, c1, p1)
        p0, c0, c1, p1 = self.to_array()
        n = max(round(self.arc_length(fast_approximation=True) / 30), 1)
        segment = q((p0, c0, c1, p1), np.linspace(0, 1, n + 1, endpoint=True))
        intercepts = intercept_segment(
            segment[:-1], segment[1:], origins, origins + tangents, b1_bound=False, b0_bound=half_line
        )

        # Find the closest intercept to the origin
        dist = np.linalg.norm(intercepts - origins[None, :, :], axis=-1)
        closest_segment = np.full(len(origins), -1, dtype=int)
        intercept = np.full((len(origins), 2), np.nan)

        no_intercept = np.isnan(dist).all(axis=0)
        if np.any(~no_intercept):
            closest_segment[~no_intercept] = np.nanargmin(dist[:, ~no_intercept], axis=0)
            intercept[~no_intercept] = np.take_along_axis(
                intercepts[:, ~no_intercept], closest_segment[None, ~no_intercept, None], axis=0
            ).squeeze(axis=0)

            if not fast_approximation:
                # Project the approximative intercept on the Bezier curve
                # TODO: Compute an exact intercept instead of a projection
                intercept[~no_intercept] = self.projection(intercept[~no_intercept], fast_approximation=True)

        return intercept

    def evaluate(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return q(self.to_array(), t)

    def evaluate_tangent(self, t: float, normalized=False):
        tangent = qprime(self.to_array(), t)
        return tangent / np.linalg.norm(tangent, axis=1)[:, None] if normalized else tangent

    def flip(self) -> BezierCubic:
        return BezierCubic(self.p1, self.c1, self.c0, self.p0)

    def has_nan(self) -> bool:
        return any(p.is_nan() for p in self)

    def is_null(self) -> bool:
        return self.p0 == self.p1 and self.c0 == self.p0 and self.c1 == self.p1

    def c0_sym(self, d: float = -1, relative=True) -> Point:
        """Compute the symmetric control point of c0 with respect to p0.

        The argument d allows to move the symmetric control point along the line p0-c0.

        Parameters
        ----------
        d : int, optional
            Distance from c0 to the symmetric control point, by default -1 (i.e. symmetric of c0 with respect to p0).

        relative : bool, optional
            If True (by default), d is multiplied by the distance between c0 and p0.
            Otherwise d is considered as an absolute distance (in pixel).

        Returns
        -------
        Point
            The symmetric control point of c0 with respect to p0 or any point along the line p0-c0.
        """
        v = self.c0 - self.p0
        if not relative:
            v = v.normalized()
        return self.p0 + v * d

    def c1_sym(self, d: float = -1, relative=True) -> Point:
        """Compute the symmetric control point of c1 with respect to p1.

        The argument d allows to move the symmetric control point along the line p1-c1.

        Parameters
        ----------
        d : int, optional
            Distance from c1 to the symmetric control point, by default -1 (i.e. symmetric of c1 with respect to p1).

        relative : bool, optional
            If True (by default), d is multiplied by the distance between c1 and p1.
            Otherwise d is considered as an absolute distance (in pixel).

        Returns
        -------
        Point
            The symmetric control point of c1 with respect to p1 or any point along the line p1-c1.
        """
        v = self.c1 - self.p1
        if not relative:
            v = v.normalized()
        return self.p1 + v * d

    def is_curvature_constant(self) -> bool:
        c0_elevation = (self.p1 - self.p0).cross(self.c0 - self.p0, normalize=True)
        c1_elevation = -(self.p0 - self.p1).cross(self.c1 - self.p1, normalize=True)
        return bool(np.isclose(c0_elevation, c1_elevation)) or np.sign(c1_elevation) == np.sign(c0_elevation)

    def split(self, t=0.5) -> Tuple[BezierCubic, BezierCubic]:
        p01 = self.p0 + (self.c0 - self.p0) * t
        p12 = self.c0 + (self.c1 - self.c0) * t
        p23 = self.c1 + (self.p1 - self.c1) * t
        p012 = p01 + (p12 - p01) * t
        p123 = p12 + (p23 - p12) * t
        p0123 = p012 + (p123 - p012) * t
        return (BezierCubic(self.p0, p01, p012, p0123), BezierCubic(p0123, p123, p23, self.p1))

    def rasterize(self, out: npt.NDArray | Tuple[int, int], width: float = 1.0, fill_value=1) -> npt.NDArray[np.int_]:
        """Rasterize the Bezier curve into a binary mask.

        Parameters
        ----------
        image_shape : Tuple[int, int]
            The shape of the output binary mask (height, width).
        width : float, optional
            The width of the curve in pixels, by default 1.0.

        Returns
        -------
        npt.NDArray[np.float32]
            A binary mask of shape `image_shape` with the rasterized Bezier curve.
        """
        import cv2

        # Sample points along the Bezier curve
        n = int(self.arc_length(fast_approximation=True))
        step = 1 / n
        t_values = np.arange(0, 1 + 0.5 * step, step / 2)
        bezier_points = np.round(self.evaluate(t_values)).astype(np.int32)
        bezier_points = np.unique(bezier_points, axis=0)

        # Drop out of bounds points
        H, W = out if isinstance(out, tuple) else out.shape
        max_shape = np.array([[H, W]])
        bezier_points = bezier_points[np.all((bezier_points >= 0) & (bezier_points < max_shape), axis=1)]

        out_array = np.zeros(out, dtype=np.int_) if isinstance(out, tuple) else out

        # Apply Gaussian blur to simulate width
        if width > 1.0:
            k = int(width * 3)
            half_k = k // 2
            K = cv2.getGaussianKernel(k, width / 3)
            K = K * K.T
            K /= K[half_k, half_k]  # Normalize the kernel

            mask_smooth = np.zeros((H, W), dtype=np.float32)
            for y, x in bezier_points:
                y0, x0 = max(0, y - half_k), max(0, x - half_k)
                ky, kx = half_k - (y - y0), half_k - (x - x0)
                h, w = min(k - ky, H - y0), min(k - kx, W - x0)
                mask_smooth[y0 : y0 + h, x0 : x0 + w] += K[ky : ky + h, kx : kx + w]
            mask = mask_smooth > 0.5
            out_array[mask] = fill_value
        else:
            out_array[bezier_points[:, 0], bezier_points[:, 1]] = fill_value
        return out_array


class BSpline(tuple[BezierCubic]):
    def __new__(cls, iterable: Iterable[BezierCubic] = ()) -> BSpline:
        assert all(isinstance(_, BezierCubic) for _ in iterable), "All elements of a BSpline must be BezierCubic"
        return super().__new__(cls, tuple(iterable))

    def __repr__(self) -> str:
        return f"BSpline({','.join([repr(_) for _ in self])})"

    def __str__(self) -> str:
        descr = f"BSpline({len(self)} curves):\n\t"
        return descr + "\n\t".join(str(curve) for curve in self)

    def to_path(self, offset: Optional[Point] = None) -> str:
        return "\n".join(curve.to_path(offset) for curve in self)

    def __add__(self, other: tuple[BezierCubic]) -> Self:
        return self.__class__(super().__add__(other))

    def __radd__(self, other: tuple[BezierCubic]) -> BSpline:
        if not isinstance(other, BSpline):
            other = BSpline(other)
        return other + self

    @classmethod
    def fit(
        cls,
        yx_points: npt.NDArray,
        tangents: Optional[npt.NDArray[np.float32]] = None,
        curvature_roots: Optional[npt.NDArray[np.int_]] = None,
        *,
        max_error: float = 3.0,
    ) -> Tuple[Self, float]:
        import torch

        from .cpp_extensions.fvt_cpp import fit_bspline

        if yx_points.shape[0] == 0:
            return cls(), float("inf")

        curve = torch.from_numpy(yx_points.copy()).int()
        tangent_torch = (
            torch.from_numpy(tangents.copy()).float()
            if tangents is not None
            else torch.empty((0, 2), dtype=torch.float)
        )

        curv_roots_torch = torch.tensor([-1], dtype=torch.int)
        if curvature_roots is not None:
            curv_roots_torch = torch.from_numpy(curvature_roots.copy()).int()

        opt = {"bspline_target_error": max_error, "ignore_gaps": 2.0, "curvature_roots_percentile_threshold": 0.1}

        bsplines, max_error = fit_bspline(curve, tangent_torch, curv_roots_torch, opt)
        return cls.from_array(bsplines.numpy()), max_error

    @classmethod
    def fit_legacy(cls, yx_points: npt.NDArray, max_error: float, split_on=None, tangent_std=2) -> Self:
        if len(yx_points) < 4:
            return cls([BezierCubic(*[Point(*yx_points[i]) for i in (0, -1, 0, -1)])])

        if split_on is None:
            inflexions = []
        elif isinstance(split_on, (list, np.ndarray)):
            inflexions = list(split_on)
            split_on = "manual"

        tangents = curve_tangent(yx_points, eval_for=[0] + inflexions + [len(yx_points) - 1], std=tangent_std)
        if split_on is None:
            curves = fitCubic(yx_points, tangents[0], tangents[-1], max_error)
        else:
            curves = []
            points = [0] + inflexions + [len(yx_points) - 1]
            for id_p0, id_p1, t0, t1 in zip(points[:-1], points[1:], tangents[:-1], tangents[1:], strict=True):
                curves += fitCubic(yx_points[id_p0 : id_p1 + 1], t0, -t1, max_error, split_on_failed=False)
        return cls([BezierCubic.from_array(curve) for curve in curves])

    @classmethod
    def from_array(cls, curves: npt.NDArray) -> Self:
        return cls([curve if isinstance(curve, BezierCubic) else BezierCubic.from_array(curve) for curve in curves])

    def to_array(self) -> npt.NDArray[np.float64]:
        return np.stack([curve.to_array() for curve in self])

    @overload
    def intermediate_points(self, return_tangent: Literal[False] = False) -> npt.NDArray[np.float64]: ...
    @overload
    def intermediate_points(
        self, return_tangent: Literal[True]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def intermediate_points(
        self, return_tangent=False
    ) -> npt.NDArray[np.float64] | Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        points: List[Point] = []
        tangents: List[Point] = []
        for curve in self:
            points.append(curve.p0)
            if return_tangent:
                tangents.append(curve.c0 - curve.p0)
                points.append(curve.p1)
                tangents.append(curve.c1 - curve.p1)
        if return_tangent:
            return np.array(points), np.array(tangents)
        else:
            points.append(self[-1].p1)
        return np.array(points)

    def flip(self) -> Self:
        return self.__class__([curve.flip() for curve in reversed(self)])

    def filling_curves(
        self, start: Optional[Point] = None, end: Optional[Point] = None, *, smoothing: int | float = 0
    ) -> List[BezierCubic]:
        filling = []
        if len(self) == 0:
            if start is not None and end is not None:
                filling.append(BezierCubic(start, start, end, end))
            return filling

        if start is not None:
            p_start = self[0].p0
            start_c = self[0].c0_sym(-start.distance(p_start) * smoothing, relative=False) if smoothing else p_start
            filling.append(BezierCubic(start, start, start_c, p_start))

        for prev, next in zip(self[:-1], self[1:], strict=True):
            if smoothing:
                dist = prev.p1.distance(next.p0)
                prev_c = prev.c1_sym(-dist * smoothing, relative=False)
                next_c = next.c0_sym(-dist * smoothing, relative=False)
            else:
                prev_c, next_c = prev.p1, next.p0
            filling.append(BezierCubic(prev.p1, prev_c, next_c, next.p0))

        if end is not None:
            p_last = self[-1].p1
            last_c = self[-1].c1_sym(-end.distance(p_last) * smoothing, relative=False) if smoothing else p_last
            filling.append(BezierCubic(p_last, last_c, end, end))

        return filling

    def interpolate_missing_curves(
        self, start: Optional[Point] = None, end: Optional[Point] = None, *, smoothing: int | float = 0
    ) -> BSpline:
        filling_curves = self.filling_curves(start, end, smoothing=smoothing)
        curves = list(self)

        bsplines = []
        if start is not None:
            bsplines.append(filling_curves.pop(0))
        for curve, filling in itertools.zip_longest(curves, filling_curves):
            bsplines.append(curve)
            if filling is not None:
                bsplines.append(filling)
        if end is not None and filling_curves:
            bsplines.append(filling_curves.pop())

        return BSpline([b for b in bsplines if not b.is_null()])

    def extend_bspline(self, start: Optional[Point] = None, end: Optional[Point] = None, *, smoothing=0.2) -> BSpline:
        extended = []
        if len(self) == 0:
            if start is not None and end is not None:
                extended.append(BezierCubic(start, start, end, end))
            return BSpline(extended)

        if start is not None:
            p_start = self[0].p0
            start_c = self[0].c0_sym(-start.distance(p_start) * smoothing, relative=False) if smoothing else p_start
            extended.append(BezierCubic(start, start, start_c, p_start))

        for prev, next in zip(self[:-1], self[1:], strict=True):
            if prev.p1 == next.p0:
                extended.append(prev)
            else:
                if smoothing:
                    dist = prev.p1.distance(next.p0)
                    prev_c = prev.c1_sym(-dist * smoothing, relative=False)
                    next_c = next.c0_sym(-dist * smoothing, relative=False)
                else:
                    prev_c, next_c = prev.p1, next.p0
                extended.append(BezierCubic(prev.p1, prev_c, next_c, next.p0))
        extended.append(self[-1])
        if end is not None:
            p_last = self[-1].p1
            last_c = self[-1].c1_sym(-end.distance(p_last) * smoothing, relative=False) if smoothing else p_last
            extended.append(BezierCubic(p_last, last_c, end, end))

        return BSpline(extended)

    def discretize(self, flat_tolerance: float = 0) -> tuple[Int2DArray, Float1DArray]:
        """Discretize the BSpline into a set of integer points.

        Parameters
        ----------
        flat_tolerance : float, optional
            The flatness tolerance for the discretization. If 0 (default), return consecutive points to form a continuous line.

        Returns
        -------
        points: Int2DArray
            The discretized points as a (n, 2) array of integers.

        u: Float1DArray
            The corresponding parameters on the BSpline for each discretized point as a (n,) array of floats.
        """  # noqa: E501
        from .cpp_extensions.fvt_cpp import discretize_bspline

        if len(self) == 0:
            return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float64)

        bspline = torch.tensor(self.to_array(), dtype=torch.float32)
        points, u = discretize_bspline(bspline, float(flat_tolerance))

        return points.numpy().astype(np.int32), u.numpy().astype(np.float64)

    def evaluate(self, t: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the position of the BSpline for a set of parameters t.

        Parameters
        ----------
        t : npt.ArrayLike
            The parameters to evaluate the BSpline at. Each parameter corresponds to a position on the BSpline, where the integer part is the index of the Bezier curve and the fractional part is the relative position on that Bezier curve.

        Returns
        -------
        npt.NDArray[np.float64]
            The evaluated points on the BSpline as a (n, 2) array if t is a vector, or as a (2,) array if t is scalar.
        """  # noqa: E501
        t, is_single = as_1d_array(t, dtype=float)
        assert t.ndim == 1, "t must be a 1D array"

        points = np.empty((len(t), 2), dtype=float)
        curves_id = np.clip(np.floor(t).astype(int), 0, len(self) - 1)
        u = t - curves_id
        for c in np.unique(curves_id):
            mask = curves_id == c
            points[mask] = self[c].evaluate(u[mask])

        return points[0] if is_single else points

    def evaluate_tangent(self, t: npt.ArrayLike, normalized=False) -> npt.NDArray[np.float64]:
        """Evaluate the tangent of the BSpline for a set of parameters t.

        Parameters
        ----------
        t : npt.ArrayLike
            The parameters to evaluate the BSpline at. Each parameter corresponds to a position on the BSpline, where the integer part is the index of the Bezier curve and the fractional part is the relative position on that Bezier curve.

        normalized : bool, optional
            If True, return normalized tangents, by default False.

        Returns
        -------
        npt.NDArray[np.float64]
            The evaluated tangents on the BSpline as a (n, 2) array if t is a vector, or as a (2,) array if t is scalar.
        """  # noqa: E501
        t, is_single = as_1d_array(t, dtype=float)
        assert t.ndim == 1, "t must be a 1D array"

        tangents = np.empty((len(t), 2), dtype=float)
        curves_id = np.clip(np.floor(t).astype(int), 0, len(self) - 1)
        u = t - curves_id
        for c in np.unique(curves_id):
            mask = curves_id == c
            tangents[mask] = self[c].evaluate_tangent(u[mask], normalized=normalized)

        return tangents[0] if is_single else tangents

    def relative_pos_to_t(self, pos: npt.ArrayLike, fast_approximation=False) -> npt.NDArray[np.float64]:
        """
        Convert relative positions on the BSpline to the corresponding parameter t.

        Parameters
        ----------
        pos : npt.ArrayLike
            The relative positions on the BSpline (between 0 and 1).

        Returns
        -------
        npt.NDArray[np.float64]
            The corresponding parameter t on the BSpline.
        """
        pos, is_single = as_1d_array(pos, dtype=float)
        assert pos.ndim == 1, "pos must be a 1D array"

        lengths = np.array([curve.arc_length(fast_approximation=fast_approximation) for curve in self])
        lengths = np.concatenate(([0], np.cumsum(lengths)))
        lengths /= lengths[-1]  # Normalize to [0, 1]

        curve_ids = np.clip(np.searchsorted(lengths, pos, side="right") - 1, 0, len(self) - 1)
        curve_start = lengths[curve_ids]
        curve_end = lengths[curve_ids + 1]
        t = curve_ids + (pos - curve_start) / (curve_end - curve_start + 1e-10)

        return t[0] if is_single else t

    def tips_tangents(self, normalize=True) -> Tuple[Point, Point]:
        t0 = self[0].c0 - self[0].p0
        t1 = self[-1].c1 - self[-1].p1
        return (t0.normalized(), t1.normalized()) if normalize else (t0, t1)

    def chord_length(self) -> float:
        if len(self) == 0:
            return 0
        return self[0].p0.distance(self[-1].p1)

    def arc_length(self, fast_approximation=False):
        return sum(curve.arc_length(fast_approximation) for curve in self)

    def ensure_constant_curvature(self) -> BSpline:
        if all(curve.is_curvature_constant() for curve in self):
            return self

        bezier_cubics = []
        for bezier in self:
            if bezier.is_curvature_constant():
                bezier_cubics.append(bezier)
            else:
                bezier_cubics.extend(bezier.split(0.5))
        return BSpline(bezier_cubics)

    def transform(self, projection: FundusProjection) -> Self:
        bspline_data = self.to_array().reshape(-1, 2)
        bspline_data = projection.transform(bspline_data).reshape(-1, 4, 2)
        return self.from_array(bspline_data)

    @overload
    def projection(
        self,
        points: npt.ArrayLike,
        *,
        fast_approximation=False,
        return_dist: Literal[False] = False,
        return_u: Literal[False] = False,
    ) -> npt.NDArray[np.float64]: ...
    @overload
    def projection(
        self,
        points: npt.ArrayLike,
        *,
        fast_approximation=False,
        return_dist: Literal[True],
        return_u: Literal[False] = False,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def projection(
        self,
        points: npt.ArrayLike,
        *,
        fast_approximation=False,
        return_dist: Literal[False] = False,
        return_u: Literal[True],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    @overload
    def projection(
        self, points: npt.ArrayLike, *, fast_approximation=False, return_dist: Literal[True], return_u: Literal[True]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
    def projection(
        self, points: npt.ArrayLike, *, fast_approximation=False, return_dist=False, return_u=False
    ) -> np.ndarray | Tuple[np.ndarray, ...]:
        """Project a set of points on the Bezier curve.

        Parameters
        ----------
        yx_points : np.ndarray
            The points to project on the Bezier curve.
        fast_approximation : bool, optional
            If True, use a fast approximation by projecting the points on the trapezoid defined by the control points, by default False.
        return_dist : bool, optional
            If True, also return the distance between the projected points and the original points, by default False.
        return_u : bool, optional
            If True, also return the parameter u of the projected points on the Bezier curve, by default False

        Returns
        -------
        projected_points : np.ndarray
            The projected points on the Bezier curve as a (n, 2) array.

        distance : np.ndarray
            If return_dist is True, also return the distance between the projected points and the original points as a (n,) array.

        u : np.ndarray
            If return_u is True, also return the parameter u of the projected points on the Bezier curve as a (n,) array. ``floor(u)`` is the index of the Bezier curve in the BSpline, and ``u - floor(u)`` is the relative position on the Bezier curve.

        """  # noqa: E501
        points = np.atleast_2d(points).astype(float)
        assert points.ndim == 2 and points.shape[1] == 2, "split_coord must be a 2D array of shape (n, 2)"
        n_split = points.shape[0]

        if len(self) == 0:
            out = (np.full(n_split, np.nan),)
            if return_dist:
                out = out + (np.full(n_split, np.nan),)
            if return_u:
                out = out + (np.full(n_split, np.nan),)
            return out[0] if len(out) == 1 else out

        # === Find the closest bezier curve for each split point ===
        closest_points, dist, u = zip(
            *[
                bezier.projection(points, return_distance=True, return_u=True, fast_approximation=fast_approximation)
                for bezier in self
            ],
            strict=True,
        )
        if len(self) == 1:
            out = (list(closest_points)[0],)
            if return_dist:
                out = out + (list(dist)[0],)
            if return_u:
                out = out + (list(u)[0],)
            return out[0] if len(out) == 1 else out
        else:
            closest_bezier = np.argmin(list(dist), axis=0)
            out = (np.take_along_axis(np.array(closest_points), closest_bezier[None, :, None], axis=0).squeeze(axis=0),)
            if return_dist:
                out = out + (np.take_along_axis(np.array(dist), closest_bezier[None, :], axis=0).squeeze(axis=0),)
            if return_u:
                u = np.take_along_axis(np.array(u), closest_bezier[None, :], axis=0).squeeze(axis=0)
                u += closest_bezier
                out = out + (u,)

            return out[0] if len(out) == 1 else out

    def intercept(
        self, origins: npt.ArrayLike, tangents: npt.ArrayLike, fast_approximation=False, half_line=True
    ) -> np.ndarray:
        """Compute the intercept of the Bezier curve with a set of lines defined by their origins and tangents.

        Parameters
        ----------
        origins : np.array
            The origins of the lines as a (n, 2) array.

        tangents : np.array
            The tangents of the lines as a (n, 2) array.

        fast_approximation : bool, optional
            If True, use a fast approximation by intercepting the trapezoid defined by the control points instead of the true bezier curve, by default False.

        Returns
        -------
        np.array
            The intercept of the Bezier curve with the lines as a (n, 2) array. If the intercept does not exist, the point is set to NaN.
        """  # noqa: E501
        origins = np.atleast_2d(origins).astype(float)
        intercept = np.full((len(origins), 2), np.nan)
        if len(self) == 0:
            return intercept

        intercepts = np.stack([curve.intercept(origins, tangents, fast_approximation, half_line) for curve in self])

        if len(self) == 1:
            return intercepts[0]
        no_intercept = np.isnan(intercepts).any(axis=2).all(axis=0)
        if np.all(no_intercept):
            return intercept

        # Find the closest intercept to the origin
        dist = np.linalg.norm(intercepts - origins[None, :, :], axis=2)
        intercept[~no_intercept] = np.take_along_axis(
            intercepts[:, ~no_intercept], np.nanargmin(dist[:, ~no_intercept], axis=0)[None, :, None], axis=0
        ).squeeze(axis=0)

        return intercept

    def split_into_multiple_bsplines(
        self, split_coord: Optional[npt.ArrayLike] = None, u: Optional[npt.ArrayLike] = None
    ) -> List[BSpline]:
        if u is None:
            if split_coord is None:
                raise ValueError("Either split_coord or u must be provided")
            else:
                _, u = self.projection(split_coord, return_u=True, fast_approximation=False)
        else:
            u = np.atleast_1d(u).astype(float).flatten()

        closest_bezier = np.floor(u).astype(int)
        u = u - closest_bezier
        previous_end = (u == 0) & (closest_bezier > 0)
        closest_bezier[previous_end] -= 1
        u[previous_end] = 1

        # === Split the closest bezier curve at the closest point ===
        beziers = [[]]
        for i, bezier in enumerate(self):
            if i not in closest_bezier:
                beziers[-1].append(bezier)
                continue

            # Select the split points and their parameter u for the current bezier curve
            # split_points, split_u = split_coord[closest_bezier == i], u[closest_bezier == i]
            split_u = u[closest_bezier == i]

            # Remove points that are at the extremities of the bezier curve
            p0_points, p1_points = np.isclose(split_u, 0), np.isclose(split_u, 1)
            not_extremities = ~(p0_points | p1_points)
            # split_points, split_u = split_points[not_extremities], split_u[not_extremities]
            split_u = split_u[not_extremities]

            # Refine the parameter u and remove any additional extremities
            # split_u = np.clip(np.sort(bezier.parametrize(split_points, initial_u=split_u)), 0, 1)
            # p0_points, p1_points = np.isclose(split_u, 0), np.isclose(split_u, 1)
            # not_extremities = ~(p0_points | p1_points)
            # split_points, split_u = split_points[not_extremities], split_u[not_extremities]

            # Create empty segment for any points before the bezier curve
            for _ in range(np.count_nonzero(p0_points)):
                beziers.append([])

            # Split the bezier curve at the split points
            if len(split_u):
                acc_t = 0
                for t in split_u:
                    b0, bezier = bezier.split((t - acc_t) / (1 - acc_t))
                    acc_t += t
                    # Append the beginning of the bezier curve (before the split) to the current segment ...
                    beziers[-1].append(b0)
                    # ... and start a new segment
                    beziers.append([])

            # Append the remaining bezier curve to the current segment
            beziers[-1].append(bezier)

            # Create empty segment for any points after the bezier curve
            for _ in range(np.count_nonzero(p1_points)):
                beziers.append([])

        return [BSpline(_) for _ in beziers]

    def split(self, split_coord: Optional[npt.ArrayLike] = None, u: Optional[npt.ArrayLike] = None) -> BSpline:
        return BSpline(itertools.chain(*self.split_into_multiple_bsplines(split_coord, u)))


@autocast_torch
def fit_bezier_cubic(
    curve: torch.Tensor,
    tangents: Optional[torch.Tensor] = None,
    max_error: float = 2,
    tangent_std: float = 2,
    start: Optional[int] = 0,
    end: Optional[int] = 0,
) -> Tuple[BezierCubic, float, torch.Tensor]:
    from .cpp_extensions.fvt_cpp import fit_bezier_cubic as fit_bezier__cubic_cpp

    curve = curve.cpu().int()
    tangents = tangents.cpu().float() if tangents is not None else torch.empty((0, 2), dtype=torch.float32)

    bezier, max_error, u = fit_bezier__cubic_cpp(curve, tangents, max_error, tangent_std, start, end)
    return BezierCubic.from_array(bezier), max_error, u


# Fit one (ore more) Bezier curves to a set of points
def fitCurve(points, maxError):
    leftTangent = normalize(points[1] - points[0])
    rightTangent = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTangent, maxError)


def fitCubic(points, leftTangent, rightTangent, error, split_on_failed=True, return_parametrization=False):
    # Use heuristic if region only has two points in it
    if len(points) == 2:
        dist = np.linalg.norm(points[0] - points[1]) / 3.0
        bezCurve = [points[0], points[0] + leftTangent * dist, points[1] + rightTangent * dist, points[1]]
        return [bezCurve]

    # Parameterize points, and attempt to fit curve
    u = chordLengthParameterize(points)
    bezCurve = generateBezier(points, u, leftTangent, rightTangent)
    # Find max deviation of points to fitted curve
    maxError, splitPoint = computeMaxError(points, bezCurve, u)
    if maxError < error:
        return [bezCurve]

    # If error not too large, try some reparameterization and iteration
    if maxError < error**2:
        for _ in range(20):
            uPrime = reparameterize(bezCurve, points, u)
            bezCurve = generateBezier(points, uPrime, leftTangent, rightTangent)
            maxError, splitPoint = computeMaxError(points, bezCurve, uPrime)
            if maxError < error:
                return [bezCurve]
            u = uPrime

    if not split_on_failed:
        return [bezCurve] if not return_parametrization else ([bezCurve], u)

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    centerTangent = normalize(points[splitPoint - 1] - points[splitPoint + 1])
    beziers += fitCubic(points[: splitPoint + 1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)

    return beziers if not return_parametrization else (beziers, u)


def generateBezier(points, parameters, leftTangent, rightTangent):
    bezCurve = [points[0], None, None, points[-1]]

    # compute the A's
    A = np.zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = leftTangent * 3 * (1 - u) ** 2 * u
        A[i][1] = rightTangent * 3 * (1 - u) * u**2

    # Create the C and X matrices
    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters, strict=True)):
        C[0][0] += np.dot(A[i][0], A[i][0])
        C[0][1] += np.dot(A[i][0], A[i][1])
        C[1][0] += np.dot(A[i][0], A[i][1])
        C[1][1] += np.dot(A[i][1], A[i][1])

        tmp = point - q([points[0], points[0], points[-1], points[-1]], u)

        X[0] += np.dot(A[i][0], tmp)
        X[1] += np.dot(A[i][1], tmp)

    # Compute the determinants of C and X
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    # Finally, derive alpha values
    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    # If alpha negative, use the Wu/Barsky heuristic (see text) */
    # (if alpha is 0, you get coincident control points that lead to
    # divide by zero in any subsequent NewtonRaphsonRootFind() call. */
    segLength = np.linalg.norm(points[0] - points[-1])
    epsilon = 1.0e-6 * segLength
    if alpha_l < epsilon or alpha_r < epsilon:
        # fall back on standard (probably inaccurate) formula, and subdivide further if needed.
        bezCurve[1] = bezCurve[0] + leftTangent * (segLength / 3.0)
        bezCurve[2] = bezCurve[3] + rightTangent * (segLength / 3.0)

    else:
        # First and last control points of the Bezier curve are
        # positioned exactly at the first and last data points
        # Control points 1 and 2 are positioned an alpha distance out
        # on the tangent vectors, left and right, respectively
        bezCurve[1] = bezCurve[0] + leftTangent * alpha_l
        bezCurve[2] = bezCurve[3] + rightTangent * alpha_r

    return bezCurve


def reparameterize(bezier, points, parameters):
    return [newtonRaphsonRootFind(bezier, point, u) for point, u in zip(points, parameters, strict=True)]


def newtonRaphsonRootFind(bez, point, u):
    """
    Newton's root finding algorithm calculates f(x)=0 by reiterating
    x_n+1 = x_n - f(x_n)/f'(x_n)

    We are trying to find curve parameter u for some point p that minimizes
    the distance from that point to the curve. Distance point to curve is d=q(u)-p.
    At minimum distance the point is perpendicular to the curve.
    We are solving
    f = q(u)-p * q'(u) = 0
    with
    f' = q'(u) * q'(u) + q(u)-p * q''(u)

    gives
    u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
    """
    d = q(bez, u) - point
    numerator = (d * qprime(bez, u)).sum()
    denominator = (qprime(bez, u) ** 2 + d * qprimeprime(bez, u)).sum()

    if denominator == 0.0:
        return u
    else:
        return u - numerator / denominator


def chordLengthParameterize(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    u = np.linalg.norm(np.diff(points, axis=0), axis=1)
    u = np.cumsum(u)
    u /= u[-1]
    return np.concatenate(([0.0], u))
    # LEGACY
    # u = [0.0]
    # for i in range(1, len(points)):
    #     u.append(u[i - 1] + np.linalg.norm(points[i] - points[i - 1]))
    # for i, _ in enumerate(u):
    #     u[i] = u[i] / u[-1]
    # return u


def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points) / 2
    for i, (point, u) in enumerate(zip(points, parameters, strict=True)):
        dist = np.linalg.norm(q(bez, u) - point) ** 2
        if dist > maxDist:
            maxDist = dist
            splitPoint = i

    return maxDist, splitPoint


def normalize(v):
    return v / np.linalg.norm(v)


# evaluates cubic bezier at t, return point
def q(ctrlPoly, t):
    if isinstance(t, np.ndarray):
        t = t[:, np.newaxis]
        ctrlPoly = np.asarray(ctrlPoly)[:, np.newaxis, :]
    return (
        (1.0 - t) ** 3 * ctrlPoly[0]
        + 3 * (1.0 - t) ** 2 * t * ctrlPoly[1]
        + 3 * (1.0 - t) * t**2 * ctrlPoly[2]
        + t**3 * ctrlPoly[3]
    )


# evaluates cubic bezier first derivative at t, return point
def qprime(ctrlPoly, t):
    if isinstance(t, np.ndarray):
        t = t[:, np.newaxis]
        ctrlPoly = np.asarray(ctrlPoly)[:, np.newaxis, :]
    return (
        3 * (1.0 - t) ** 2 * (ctrlPoly[1] - ctrlPoly[0])
        + 6 * (1.0 - t) * t * (ctrlPoly[2] - ctrlPoly[1])
        + 3 * t**2 * (ctrlPoly[3] - ctrlPoly[2])
    )


# evaluates cubic bezier second derivative at t, return point
def qprimeprime(ctrlPoly, t):
    if isinstance(t, np.ndarray):
        t = t[:, np.newaxis]
        ctrlPoly = np.asarray(ctrlPoly)[:, np.newaxis, :]
    return 6 * (1.0 - t) * (ctrlPoly[2] - 2 * ctrlPoly[1] + ctrlPoly[0]) + 6 * (t) * (
        ctrlPoly[3] - 2 * ctrlPoly[2] + ctrlPoly[1]
    )
