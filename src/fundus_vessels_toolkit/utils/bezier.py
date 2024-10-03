from __future__ import annotations, print_function

from typing import List, Literal, NamedTuple, Optional, Tuple, overload

import numpy as np
import numpy.typing as npt
import torch

from .geometric import Point
from .graph.measures import curve_tangent
from .torch import autocast_torch


class BezierCubic(NamedTuple):
    p0: Point
    c0: Point
    c1: Point
    p1: Point

    def __str__(self) -> str:
        return f"BezierCubic(p0={self.p0}, c0={self.c0}, c1={self.c1}, p1={self.p1})"

    @classmethod
    def from_array(cls, curve: np.array) -> BezierCubic:
        points = [Point(float(p[0]), float(p[1])) for p in curve[:4]]
        return cls(*points)

    def to_path(self, offset: Optional[Point] = None) -> str:
        if self.p0.is_nan() or self.c0.is_nan() or self.c1.is_nan() or self.p1.is_nan():
            return ""
        oy, ox = offset if offset is not None else (0, 0)
        return (
            f"M {self.p0.x-ox},{self.p0.y-oy} "
            f"C {self.c0.x-ox},{self.c0.y-oy} {self.c1.x-ox},{self.c1.y-oy} "
            f"{self.p1.x-ox},{self.p1.y-oy}"
        )

    def to_array(self) -> np.array:
        return np.array([self.p0, self.c0, self.c1, self.p1])

    def chord_length(self) -> float:
        return np.linalg.norm(self.p0 - self.p1)

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
    ) -> np.array:
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
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
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
        n = len(yx_points)

        # Compute projection on the trapezoid defined by (p0, c0, c1, p1)
        p0, c0, c1, p1 = self.to_array()
        p0c0_u = ((c0 - p0) / (p0c0norm := np.linalg.norm(c0 - p0)))[None, :]
        c0c1_u = ((c1 - c0) / (c0c1norm := np.linalg.norm(c1 - c0)))[None, :]
        p1c1_u = ((c1 - p1) / (p1c1norm := np.linalg.norm(c1 - p1)))[None, :]

        l_p0c0 = np.clip((yx_points - p0) @ p0c0_u.T, 0, p0c0norm)
        l_c0c1 = np.clip((yx_points - c0) @ c0c1_u.T, 0, c0c1norm)
        l_p1c1 = np.clip((yx_points - p1) @ p1c1_u.T, 0, p1c1norm)
        p0c0_proj = l_p0c0 * p0c0_u + p0[None, :]
        c0c1_proj = l_c0c1 * c0c1_u + c0[None, :]
        p1c1_proj = l_p1c1 * p1c1_u + p1[None, :]

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

            proj = self.evaluate(u)
            out = (proj,)
            if return_distance:
                dist = np.linalg.norm(proj - yx_points, axis=1)
                out = out + (dist,)
            if return_u:
                out = out + (u,)
            return proj if len(out) == 1 else out

    def evaluate(self, t: float):
        return q(self.to_array(), t)

    def evaluate_tangent(self, t: float, normalized=False):
        tangent = qprime(self.to_array(), t)
        return tangent / np.linalg.norm(tangent, axis=1)[:, None] if normalized else tangent

    def flip(self) -> BezierCubic:
        return BezierCubic(self.p1, self.c1, self.c0, self.p0)

    def has_nan(self) -> bool:
        return any(p.is_nan() for p in self)

    def c0_sym(self, d=-1, relative=True) -> Point:
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

    def c1_sym(self, d=-1, relative=True) -> Point:
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
        return np.isclose(c0_elevation, c1_elevation) or np.sign(c1_elevation) == np.sign(c0_elevation)

    def split(self, t=0.5) -> BSpline:
        p01 = self.p0 + (self.c0 - self.p0) * t
        p12 = self.c0 + (self.c1 - self.c0) * t
        p23 = self.c1 + (self.p1 - self.c1) * t
        p012 = p01 + (p12 - p01) * t
        p123 = p12 + (p23 - p12) * t
        p0123 = p012 + (p123 - p012) * t
        return BSpline((BezierCubic(self.p0, p01, p012, p0123), BezierCubic(p0123, p123, p23, self.p1)))


class BSpline(tuple[BezierCubic]):
    def __new__(cls, iterable: np.Iterable[BezierCubic] = ()) -> BSpline:
        assert all(isinstance(_, BezierCubic) for _ in iterable), "All elements of a BSpline must be BezierCubic"
        return super(BSpline, cls).__new__(cls, tuple(iterable))

    def __repr__(self) -> str:
        return f"BSpline({','.join([repr(_) for _ in self])})"

    def __str__(self) -> str:
        descr = f"BSpline({len(self)} curves):\n\t"
        return descr + "\n\t".join(str(curve) for curve in self)

    def to_path(self, offset: Optional[Point] = None) -> str:
        return "\n".join(curve.to_path(offset) for curve in self)

    def __add__(self, other: BSpline) -> BSpline:
        return BSpline(super().__add__(other))

    def __radd__(self, other: BSpline) -> BSpline:
        return BSpline(other + super())

    @classmethod
    def fit(cls, yx_points: np.array, max_error: float, split_on=None, tangent_std=2) -> BSpline:
        if len(yx_points) < 4:
            return cls([BezierCubic(*[Point(*yx_points[i]) for i in (0, -1, 0, -1)])])

        if split_on is None:
            inflexions = []
        elif isinstance(split_on, (list, np.ndarray)):
            inflexions = list(split_on)
            split_on = "manual"

        tangents = curve_tangent(yx_points, [0] + inflexions + [len(yx_points) - 1], std=tangent_std)
        if split_on is None:
            curves = fitCubic(yx_points, tangents[0], tangents[-1], max_error)
        else:
            curves = []
            points = [0] + inflexions + [len(yx_points) - 1]
            for id_p0, id_p1, t0, t1 in zip(points[:-1], points[1:], tangents[:-1], tangents[1:], strict=True):
                curves += fitCubic(yx_points[id_p0 : id_p1 + 1], t0, -t1, max_error, split_on_failed=False)
        return cls([BezierCubic.from_array(curve) for curve in curves])

    @classmethod
    def from_array(cls, curves: np.array) -> BSpline:
        return cls([curve if isinstance(curve, BezierCubic) else BezierCubic.from_array(curve) for curve in curves])

    def intermediate_points(self, return_tangent=False) -> np.array | Tuple[np.array, np.array]:
        points = []
        tangents = []
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

    def flip(self) -> BSpline:
        return BSpline([curve.flip() for curve in reversed(self)])

    def filling_curves(
        self, start: Optional[Point] = None, end: Optional[Point] = None, *, smoothing=0
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
            if prev.p1 == next.p0:
                continue
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

    def tips_tangents(self, normalize=True) -> Tuple[Point, Point]:
        t0 = self[0].c0 - self[0].p0
        t1 = self[-1].c1 - self[-1].p1
        return (t0.normalized(), t1.normalized()) if normalize else (t0, t1)

    def chord_length(self) -> float:
        if len(self) == 0:
            return 0
        return np.linalg.norm(self[0].p0 - self[-1].p1)

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

    def parametrize(yx_points: np.ndarray, error=2, max_iteration=20) -> np.array:
        pass

    @overload
    def split(self, split_coord: npt.ArrayLike[float], return_individual_bspline: Literal[True]) -> List[BSpline]: ...
    @overload
    def split(self, split_coord: npt.ArrayLike[float], return_individual_bspline: Literal[False]) -> BSpline: ...
    def split(self, split_coord: npt.ArrayLike[float], return_individual_bspline=False) -> List[BSpline] | BSpline:
        split_coord = np.atleast_2d(split_coord).astype(float)
        assert split_coord.ndim == 2 and split_coord.shape[1] == 2, "split_coord must be a 2D array of shape (n, 2)"
        n_split = split_coord.shape[0]

        if len(self) == 0:
            return [BSpline() for _ in range(n_split)] if return_individual_bspline else self

        # === Find the closest bezier curve for each split point ===
        closest_points, dist, u = zip(
            *[
                bezier.projection(split_coord, return_distance=True, return_u=True, fast_approximation=True)
                for bezier in self
            ],
            strict=True,
        )
        if len(self) == 1:
            closest_points = list(closest_points)[0]
            u = list(u)[0]
            closest_bezier = np.full(n_split, 0)
        else:
            closest_bezier = np.argmin(list(dist), axis=0)
            u = np.take_along_axis(np.array(u), closest_bezier[None, :], axis=0).squeeze(axis=0)
            closest_points = np.take_along_axis(
                np.array(closest_points), closest_bezier[None, :, None], axis=0
            ).squeeze(axis=0)

        # === Split the closest bezier curve at the closest point ===
        beziers = [[]]
        for i, bezier in enumerate(self):
            if i not in closest_bezier:
                beziers[-1].append(bezier)
                continue

            # Select the split points and their parameter u for the current bezier curve
            split_points, split_u = split_coord[closest_bezier == i], u[closest_bezier == i]

            # Remove points that are at the extremities of the bezier curve
            p0_points, p1_points = np.isclose(split_u, 0), np.isclose(split_u, 1)
            not_extremities = ~(p0_points | p1_points)
            split_points, split_u = split_points[not_extremities], split_u[not_extremities]

            # Refine the parameter u and remove any additional extremities
            split_u = np.clip(np.sort(bezier.parametrize(split_points, initial_u=split_u)), 0, 1)
            p0_points, p1_points = np.isclose(split_u, 0), np.isclose(split_u, 1)
            not_extremities = ~(p0_points | p1_points)
            split_points, split_u = split_points[not_extremities], split_u[not_extremities]

            # Create empty segment for any points before the bezier curve
            for _ in range(np.sum(p0_points)):
                beziers.append([])

            # Split the bezier curve at the split points
            if len(split_points):
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
            for _ in range(np.sum(p1_points)):
                beziers.append([])

        return [BSpline(_) for _ in beziers] if return_individual_bspline else BSpline(sum(beziers, []))


@autocast_torch
def fit_bezier_cubic(
    curve: torch.Tensor, tangents: torch.Tensor, max_error: float, start: Optional[int] = 0, end: Optional[int] = 0
) -> Tuple[BezierCubic, float, torch.Tensor]:
    from .cpp_extensions.graph_cpp import fit_bezier as fit_bezier_cpp

    curve = curve.cpu().int()
    tangents = tangents.cpu().float()

    bezier, max_error, u = fit_bezier_cpp(curve, tangents, max_error, start, end)
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
        for i in range(20):
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
    return [newtonRaphsonRootFind(bezier, point, u) for point, u in zip(points, parameters)]


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


def chordLengthParameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i - 1] + np.linalg.norm(points[i] - points[i - 1]))

    for i, _ in enumerate(u):
        u[i] = u[i] / u[-1]

    return u


def computeMaxError(points, bez, parameters):
    maxDist = 0.0
    splitPoint = len(points) / 2
    for i, (point, u) in enumerate(zip(points, parameters)):
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
