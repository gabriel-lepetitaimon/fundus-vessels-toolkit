from __future__ import annotations, print_function

from typing import NamedTuple, Tuple

import numpy as np

from .geometric import Point
from .yx_curves import compute_tangent


class BezierSpline:
    def __init__(self, curves: List[BezierCubic]):
        self.curves = curves

    def to_path(self) -> str:
        return "\n".join(curve.to_path() for curve in self.curves)

    @classmethod
    def fit(cls, yx_points: np.array, max_error: float, split_on=None, tangent_std=2) -> BezierSpline:
        if len(yx_points) < 4:
            return cls([BezierCubic(*[Point(*yx_points[i]) for i in (0, -1, 0, -1)])])

        if split_on is None:
            inflexions = []
        elif isinstance(split_on, (list, np.ndarray)):
            inflexions = list(split_on)
            split_on = "manual"

        tangents = compute_tangent(yx_points, [0] + inflexions + [len(yx_points) - 1], std=tangent_std)
        if split_on is None:
            curves = fitCubic(yx_points, tangents[0], tangents[-1], max_error)
        else:
            curves = []
            points = [0] + inflexions + [len(yx_points) - 1]
            for id_p0, id_p1, t0, t1 in zip(points[:-1], points[1:], tangents[:-1], tangents[1:], strict=True):
                curves += fitCubic(yx_points[id_p0 : id_p1 + 1], t0, -t1, max_error, split_on_failed=False)
        return cls([BezierCubic.from_numpy(curve) for curve in curves])

    def intermediate_points(self, return_tangent=False) -> np.array | Tuple[np.array, np.array]:
        points = []
        tangents = []
        for curve in self.curves:
            points.append(curve.p0)
            if return_tangent:
                tangents.append(curve.c0 - curve.p0)
                points.append(curve.p1)
                tangents.append(curve.c1 - curve.p1)
        if return_tangent:
            return np.array(points), np.array(tangents)
        else:
            points.append(self.curves[-1].p1)
        return np.array(points)


class BezierCubic(NamedTuple):
    p0: Point
    c0: Point
    c1: Point
    p1: Point

    @classmethod
    def from_numpy(cls, curve: np.array) -> BezierCubic:
        return cls(Point(*curve[0]), Point(*curve[1]), Point(*curve[2]), Point(*curve[3]))

    def to_path(self) -> str:
        return f"M {self.p0.x},{self.p0.y} C {self.c0.x},{self.c0.y} {self.c1.x},{self.c1.y} {self.p1.x},{self.p1.y}"


# Fit one (ore more) Bezier curves to a set of points
def fitCurve(points, maxError):
    leftTangent = normalize(points[1] - points[0])
    rightTangent = normalize(points[-2] - points[-1])
    return fitCubic(points, leftTangent, rightTangent, maxError)


def fitCubic(points, leftTangent, rightTangent, error, split_on_failed=True):
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
        return [bezCurve]

    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    centerTangent = normalize(points[splitPoint - 1] - points[splitPoint + 1])
    beziers += fitCubic(points[: splitPoint + 1], leftTangent, centerTangent, error)
    beziers += fitCubic(points[splitPoint:], -centerTangent, rightTangent, error)

    return beziers


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

    for i, (point, u) in enumerate(zip(points, parameters)):
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
    return (
        (1.0 - t) ** 3 * ctrlPoly[0]
        + 3 * (1.0 - t) ** 2 * t * ctrlPoly[1]
        + 3 * (1.0 - t) * t**2 * ctrlPoly[2]
        + t**3 * ctrlPoly[3]
    )


# evaluates cubic bezier first derivative at t, return point
def qprime(ctrlPoly, t):
    return (
        3 * (1.0 - t) ** 2 * (ctrlPoly[1] - ctrlPoly[0])
        + 6 * (1.0 - t) * t * (ctrlPoly[2] - ctrlPoly[1])
        + 3 * t**2 * (ctrlPoly[3] - ctrlPoly[2])
    )


# evaluates cubic bezier second derivative at t, return point
def qprimeprime(ctrlPoly, t):
    return 6 * (1.0 - t) * (ctrlPoly[2] - 2 * ctrlPoly[1] + ctrlPoly[0]) + 6 * (t) * (
        ctrlPoly[3] - 2 * ctrlPoly[2] + ctrlPoly[1]
    )
