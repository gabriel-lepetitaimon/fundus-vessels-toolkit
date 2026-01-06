/*
An Algorithm for Automatically Fitting Digitized Curves
by Philip J. Schneider
from "Graphics Gems", Academic Press, 1990
*/

#include "bezier.h"

/*
 *  FitCubic :
 *  	Fit a Bezier curve to a (sub)set of digitized points
 *
 * CurveYX d : Array of digitized points
 * int first, last : Indices of first and last  (included!) pts in region
 * Vector t0, t1 : Unit tangent vectors at endpoints
 * double error : User-defined error squared
 *
 * Returns: tuple of
 *   BezierCubic bezCurve : Fitted Bezier curve
 *   double maxSqrError : Maximum squared error of fit
 *   std::vector<double> sqrErrors : Squared errors for each point
 *   std::vector<double> u : Parameterization of points
 */
std::tuple<BezierCubic, double, std::vector<double>, std::vector<double>> fit_bezier(
    const CurveYX& d, const std::vector<Point>& tangent, double targetSqrError, std::size_t first, std::size_t last) {
    BezierCubic bezCurve;          /*Control points of fitted Bezier curve*/
    const int MAX_ITERATIONS = 10; /*  Max times to try iterating  */

    double iterationError = targetSqrError * 4.0; /* fixed issue 23 */
    if (last == 0) last = d.size() - 1;
    std::size_t nPts = last - first + 1;

    /*  Use heuristic if region only has two points in it */
    if (nPts == 2) {
        // float dist = distance(d[last], d[first]) / 3.0;
        // Point mid = (d[first] + d[last]) / 2.0;

        bezCurve[0] = d[first];
        bezCurve[3] = d[last];
        bezCurve[1] = d[first];  // d[first] + t0 * dist;
        bezCurve[2] = d[last];   // d[last] + t1 * dist;
        return {bezCurve, 0, {0.0, 0.0}, {0.0, 1.0}};
    } else if (nPts == 3) {
        bezCurve[0] = d[first];
        bezCurve[1] = d[first + 1];
        bezCurve[2] = d[first + 1];
        bezCurve[3] = d[last];

        std::vector<double> u = {0.0, 0.5, 1.0};
        auto const& [sqrErrors, maxSqrError, _] = computeMaxError(d, first, last, bezCurve, u);
        return {bezCurve, maxSqrError, sqrErrors, u};
    } else if (nPts == 4) {
        bezCurve[0] = d[first];
        bezCurve[1] = d[first + 1];
        bezCurve[2] = d[first + 2];
        bezCurve[3] = d[last];

        std::vector<double> u = {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0};
        auto const& [sqrErrors, maxSqrError, _] = computeMaxError(d, first, last, bezCurve, u);
        return {bezCurve, maxSqrError, sqrErrors, u};
    }

    Vector t0 = tangent[first], t1 = -tangent[last];

    /*  Parameterize points, and attempt to fit curve */
    std::vector<double> u = chordLengthParameterize(d, first, last);
    bezCurve = bezier_regression(d, first, last, u, t0, t1);

    /*  Find max deviation of points to fitted curve */
    auto const& [sqrErrorIni, maxSqrErrorIni, _] = computeMaxError(d, first, last, bezCurve, u);

    if (maxSqrErrorIni < targetSqrError || maxSqrErrorIni > iterationError)
        return {bezCurve, maxSqrErrorIni, sqrErrorIni, u};

    /*  If error not too large, try some reparameterization and iteration */
    for (int i = 0; i <= MAX_ITERATIONS; i++) {
        reparameterize(u, bezCurve, d, first, last);
        bezCurve = bezier_regression(d, first, last, u, t0, t1);
        auto const& [sqrError, maxSqrError, split] = computeMaxError(d, first, last, bezCurve, u);
        if (maxSqrError < targetSqrError || i == MAX_ITERATIONS) return {bezCurve, maxSqrError, sqrError, u};
    }

    return {bezCurve, maxSqrErrorIni, sqrErrorIni, u};
}

/*
 *  GenerateBezier :
 *  Use least-squares method to find Bezier control points for region.
 *
 * Point2 *d : Array of digitized points
 * int first, last : Indices defining region
 * double *uPrime : Parameter values for region
 * Vector2 tHat1, tHat2 : Unit tangents at endpoints
 */
BezierCubic bezier_regression(const CurveYX& d, std::size_t first, std::size_t last, const std::vector<double>& uPrime,
                              const Vector& t0, const Vector& t1) {
    int nPts = last - first + 1;

    /* Compute the A's	*/
    /*
    std::vector<std::array<Vector, 2>> A; // Precomputed rhs for eqn
    A.reserve(nPts);
    for (int i = 0; i < nPts; i++) {
        double u = uPrime[i];
        A.push_back({t0 * B1(u), t1 * B2(u)});
    }
    */

    /* Create the C and X matrices	*/
    Matrix2<double> C = {{{0.0, 0.0}, {0.0, 0.0}}};
    std::array<double, 2> X = {0.0, 0.0};

    for (int i = 0; i < nPts; i++) {
        const double& u = uPrime[i];
        std::array<Vector, 2> A = {t0 * B1(u), t1 * B2(u)};
        C[0][0] += A[0].dot(A[0]);
        C[0][1] += A[0].dot(A[1]);
        /*					C[1][0] += V2Dot(&A[i][0], &A[i][1]); // C[1][0] = C[0][1]*/
        C[1][1] += A[1].dot(A[1]);

        Vector tmp = d[first + i];
        tmp -= d[first] * (B0(u) + B1(u)) + d[last] * (B2(u) + B3(u));

        X[0] += A[0].dot(tmp);
        X[1] += A[1].dot(tmp);
    }
    C[1][0] = C[0][1];

    /* Compute the determinants of C and X	*/
    double det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1];
    double det_C0_X = C[0][0] * X[1] - C[1][0] * X[0];
    double det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1];

    /* Finally, derive alpha values	*/
    double alpha_l = (det_C0_C1 == 0) ? 0.0 : det_X_C1 / det_C0_C1;
    double alpha_r = (det_C0_C1 == 0) ? 0.0 : det_C0_X / det_C0_C1;

    /* Create the Bezier Curve*/
    BezierCubic bezCurve = {d[first], 0, 0, d[last]};

    double segLength = distance(d[last], d[first]);
    double epsilon = 1.0e-6 * segLength;
    if (alpha_l < epsilon || alpha_r < epsilon) {
        /* If alpha negative, use the Wu/Barsky heuristic (see text) */
        /* (if alpha is 0, you get coincident control points that lead to
         * divide by zero in any subsequent NewtonRaphsonRootFind() call. */
        /* fall back on standard (probably inaccurate) formula, and subdivide further if needed. */
        double dist = segLength / 3.0;
        bezCurve[1] = bezCurve[0] + t0 * dist;
        bezCurve[2] = bezCurve[3] + t1 * dist;
        return bezCurve;
    } else {
        /*  First and last control points of the Bezier curve are */
        /*  positioned exactly at the first and last data points */
        /*  Control points 1 and 2 are positioned an alpha distance out */
        /*  on the tangent vectors, left and right, respectively */
        bezCurve[1] = bezCurve[0] + t0 * alpha_l;
        bezCurve[2] = bezCurve[3] + t1 * alpha_r;
        return bezCurve;
    }
}

PointList evaluate_bezier(const BezierCubic& bezCurve, const std::vector<double>& u) {
    PointList curve;
    curve.reserve(u.size());
    for (int i = 0; i < (int)u.size(); i++) curve.push_back(bezierPolynomialTriangle(bezCurve, u[i]));
    return curve;
}
Point evaluate_bezier(const BezierCubic& bezCurve, const double& u) { return bezierPolynomialTriangle(bezCurve, u); }

PointList evaluate_bezier_tangent(const BezierCubic& bezCurve, const std::vector<double>& u) {
    PointList curve;
    curve.reserve(u.size());

    // Precompute the tangent control polynomial
    std::array<Point, 3> Q;
    for (int i = 0; i <= 2; i++) Q[i] = (bezCurve[i + 1] - bezCurve[i]) * 3.0;

    for (int i = 0; i < (int)u.size(); i++) curve.push_back(bezierPolynomialTriangle(Q, u[i]));
    return curve;
}

/*
 *  ChordLengthParameterize :
 *	Assign parameter values to digitized points
 *	using relative distances between points.
 *
 * Point2 *d : Array of digitized points
 * int first, last : Indices defining region
 */
std::vector<double> chordLengthParameterize(const CurveYX& d, std::size_t first, std::size_t last) {
    int nPts = last - first + 1;
    std::vector<double> u(nPts, 0); /*  Parameterization		*/

    for (int i = 1; i < nPts; i++) u[i] = u[i - 1] + distance(d[i + first], d[i + first - 1]);

    double lastU = u[nPts - 1];
    for (int i = 1; i < nPts - 1; i++) u[i] /= lastU;
    u[nPts - 1] = 1.0;

    return u;
}

/*
 *  DiscretizeBezier
 *	Discretize a bezier cubic curve into a succession of "flat" lines.
 *  Iteratively subdivides the curve until the maximum distance between
 *  the curve and the line is below a threshold.
 */
std::tuple<PointList, std::vector<double>> discretizeBezier(const BezierCubic& bezCurve) {
    PointList points = {bezCurve[0]};
    std::vector<double> us = {0.0};
    _recursiveDiscretizeBezier(bezCurve, 0.0, 1.0, points, us);

    return {points, us};
}

/*
 *  SubdivideBezier
 *	Subdivide a bezier cubic curve into two halves at parameter u, using de Casteljau's algorithm.
 */
std::pair<BezierCubic, BezierCubic> subdivideBezier(const BezierCubic& curve, const double& u) {
    BezierCubic left, right;
    std::array<Point, 7> V;

    // Initialize V with control points
    for (int i = 0; i < 4; i++) V[i] = curve[i];

    // de Casteljau's algorithm
    for (int i = 1; i <= 3; i++) {
        for (int j = 0; j <= 3 - i; j++) {
            V[j] = V[j] * (1.0 - u) + V[j + 1] * u;
        }
        left[i] = V[0];
        right[3 - i] = V[3 - i];
    }
    left[0] = curve[0];
    right[3] = curve[3];

    return {left, right};
}

bool _isFlatEnough(const BezierCubic& curve) {
    const Point& a = (curve[1] * 3.0 - curve[0] * 2.0 - curve[3]).sqr();
    const Point& b = (curve[2] * 3.0 - curve[3] * 2.0 - curve[0]).sqr();
    return std::max(a.x, b.x) + std::max(a.y, b.y) <= 8;
}

void _recursiveDiscretizeBezier(const BezierCubic& curveSegment, double u_start, double u_end, PointList& points,
                                std::vector<double>& us) {
    // Check flatness
    const auto &p0 = curveSegment[0], p3 = curveSegment[3];
    if (distanceSqr(p0, p3) <= 2 || _isFlatEnough(curveSegment)) {
        // Add endpoint to the list
        points.push_back(p3);
        us.push_back(u_end);
    } else {
        // Subdivide curve at midpoint
        double u_mid = findNewtonRaphsonRoot(curveSegment, evaluate_bezier(curveSegment, 0.5).toInt(), 0.5);
        auto [left, right] = subdivideBezier(curveSegment, u_mid);
        u_mid = u_start + (u_end - u_start) * u_mid;
        _recursiveDiscretizeBezier(left, u_start, u_mid, points, us);
        _recursiveDiscretizeBezier(right, u_mid, u_end, points, us);
    }
}
/*
 *  Reparameterize:
 *	Given set of points and their parameterization, try to find
 *   a better parameterization.
 *
 * Point2 *d : Array of digitized points
 * int first, last : Indices defining region
 * double *u : Current parameter values
 * BezierCubic bezCurve : Current fitted curve
 */
void reparameterize(std::vector<double>& u, const BezierCubic& bezCurve, const CurveYX& d, std::size_t first,
                    std::size_t last) {
    int nPts = (int)u.size();
    for (int i = 0; i < nPts; i++) u[i] = findNewtonRaphsonRoot(bezCurve, d[linspace_int(i, first, last, nPts)], u[i]);
}

/*
 *  NewtonRaphsonRootFind :
 *	Use Newton-Raphson iteration to find better root.
 *
 * BezierCubic Q : Current fitted curve
 * Point2 P : Digitized point
 * double u : Parameter value for "P"
 */
double findNewtonRaphsonRoot(const BezierCubic& Q, const Point& P, double u) {
    double numerator, denominator;
    std::vector<Point> Q1;
    std::vector<Point> Q2; /*  Q' and Q''			*/
    Point Q_u, Q1_u, Q2_u; /*u evaluated at Q, Q', & Q''	*/
    double uPrime;         /*  Improved u			*/
    int i;

    /* Compute Q(u)	*/
    Q_u = BezierII(std::vector<Point>(Q.begin(), Q.end()), u);

    /* Generate control vertices for Q'	*/
    for (i = 0; i <= 2; i++) Q1.push_back((Q[i + 1] - Q[i]) * 3.0);

    /* Generate control vertices for Q'' */
    for (i = 0; i <= 1; i++) Q2.push_back((Q1[i + 1] - Q1[i]) * 2.0);

    /* Compute Q'(u) and Q''(u)	*/
    Q1_u = BezierII(Q1, u);
    Q2_u = BezierII(Q2, u);

    /* Compute f(u)/f'(u) */
    numerator = (Q_u.x - P.x) * (Q1_u.x) + (Q_u.y - P.y) * (Q1_u.y);
    denominator = (Q1_u.x) * (Q1_u.x) + (Q1_u.y) * (Q1_u.y) + (Q_u.x - P.x) * (Q2_u.x) + (Q_u.y - P.y) * (Q2_u.y);
    if (denominator == 0.0f) return u;

    /* u = u - f(u)/f'(u) */
    uPrime = u - (numerator / denominator);
    return uPrime;
}

/*
 *  Bezier :
 *  	Evaluate a Bezier curve at a particular parameter value
 *
 * int degree : Degree of bezier curve
 * BezierCubic V : Control points
 * double t : Parametric value to find point for
 */
Point BezierII(std::vector<Point> V, double t) {
    int i, j;
    const int degree = (int)V.size() - 1;

    /* Triangle computation	*/
    for (i = 1; i <= degree; i++) {
        for (j = 0; j <= degree - i; j++) {
            V[j] = V[j] * (1.0 - t) + V[j + 1] * t;
        }
    }
    return V[0];
}

/*
 *  B0, B1, B2, B3 :
 *	Bezier multipliers
 */
double B0(double u) {
    double u0 = 1.0 - u;
    return u0 * u0 * u0;
}

double B1(double u) {
    double u0 = 1.0 - u;
    return 3 * u * u0 * u0;
}

double B2(double u) {
    double u0 = 1.0 - u;
    return 3 * u * u * u0;
}

double B3(double u) { return u * u * u; }

/*
 *  ComputeMaxError :
 *	Find the maximum squared distance of digitized points
 *	to fitted curve.
 *
 * Point2 *d : Array of digitized points
 * int first, last : Indices defining region
 * BezierCubic bezCurve : Fitted Bezier curve
 * double *u : Parameterization of points
 * int *splitPoint : Point of maximum error
 */
std::tuple<std::vector<double>, double, std::size_t> computeMaxError(const CurveYX& d, std::size_t first,
                                                                     std::size_t last, const BezierCubic& bezCurve,
                                                                     const std::vector<double>& u) {
    double maxDist; /*  Maximum error		*/
    double dist;    /*  Current error		*/
    Point P;        /*  Point on curve		*/
    Vector v;       /*  Vector from point to curve	*/

    std::vector<double> sqrErrors;
    sqrErrors.reserve(last - first + 1);
    std::size_t splitPoint = (last - first + 1) / 2;
    maxDist = 0.0;
    for (std::size_t i = first; i <= last; i++) {
        P = bezierPolynomialTriangle(bezCurve, u[i - first]);
        v = P - d[i];
        dist = v.squaredNorm();
        sqrErrors.push_back(dist);
        if (dist >= maxDist) {
            maxDist = dist;
            splitPoint = i;
        }
    }
    return {sqrErrors, maxDist, splitPoint};
}

torch::Tensor bspline_to_tensor(const BezierCubic& bspline) {
    torch::Tensor spline_t = torch::zeros({4, 2}, torch::kFloat);
    auto spline_acc = spline_t.accessor<float, 2>();
    for (int j = 0; j < 4; j++) {
        spline_acc[j][0] = bspline[j].y;
        spline_acc[j][1] = bspline[j].x;
    }
    return spline_t;
}

torch::Tensor bspline_to_tensor(const BSpline& bspline) {
    torch::Tensor spline_t = torch::zeros({(long)bspline.size(), 4, 2}, torch::kFloat);
    auto spline_acc = spline_t.accessor<float, 3>();
    for (int i = 0; i < (int)bspline.size(); i++) {
        for (int j = 0; j < 4; j++) {
            spline_acc[i][j][0] = bspline[i][j].y;
            spline_acc[i][j][1] = bspline[i][j].x;
        }
    }
    return spline_t;
}

std::vector<torch::Tensor> bsplines_to_tensor(const std::vector<BSpline>& bsplines) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(bsplines.size());
    for (const auto& bspline : bsplines) tensors.push_back(bspline_to_tensor(bspline));
    return tensors;
}