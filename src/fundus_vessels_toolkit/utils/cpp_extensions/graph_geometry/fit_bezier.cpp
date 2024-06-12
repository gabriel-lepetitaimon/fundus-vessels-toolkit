/*
An Algorithm for Automatically Fitting Digitized Curves
by Philip J. Schneider
from "Graphics Gems", Academic Press, 1990
*/

#include "fit_bezier.h"

/*
 *  FitCubic :
 *  	Fit a Bezier curve to a (sub)set of digitized points
 *
 * CurveYX d : Array of digitized points
 * int first, last : Indices of first and last pts in region
 * Vector t0, t1 : Unit tangent vectors at endpoints
 * double error : User-defined error squared
 */
BSpline fit_bspline(const CurveYX& d, const std::vector<Point>& tangent, double error, int first, int last) {
    auto [bezCurve, maxError, maxErrorPoint] = fit_bezier(d, tangent, error, first, last);

    if (maxError < error) {
        /*  If error is acceptable, return this curve */
        return {bezCurve};
    } else {
        /* Fitting failed -- split at max error point and fit recursively */
        auto bspline0 = fit_bspline(d, tangent, error, first, maxErrorPoint);
        auto const& bspline1 = fit_bspline(d, tangent, error, maxErrorPoint, last);
        bspline0.insert(bspline0.end(), bspline1.begin(), bspline1.end());
        return bspline0;
    }
}

/*
 *  FitCubic :
 *  	Fit a Bezier curve to a (sub)set of digitized points
 *
 * CurveYX d : Array of digitized points
 * int first, last : Indices of first and last pts in region
 * Vector t0, t1 : Unit tangent vectors at endpoints
 * double error : User-defined error squared
 */
std::tuple<BezierCurve, double, int> fit_bezier(const CurveYX& d, const std::vector<Point>& tangent, double error,
                                                int first, int last) {
    BezierCurve bezCurve;          /*Control points of fitted Bezier curve*/
    const int MAX_ITERATIONS = 20; /*  Max times to try iterating  */

    double iterationError = error * 4.0; /* fixed issue 23 */
    if (last < 0) last += (int)d.size();
    int nPts = last - first + 1;

    Vector t0 = tangent[first], t1 = -tangent[last];

    /*  Use heuristic if region only has two points in it */
    if (nPts == 2) {
        float dist = distance(d[last], d[first]) / 3.0;

        bezCurve[0] = d[first];
        bezCurve[3] = d[last];
        bezCurve[1] = d[first] + t0 * dist;
        bezCurve[2] = d[last] + t1 * dist;
        return {bezCurve, 0, -1};
    }

    /*  Parameterize points, and attempt to fit curve */
    std::vector<double> u = chordLengthParameterize(d, first, last);
    bezCurve = bezier_regression(d, first, last, u, t0, t1);

    /*  Find max deviation of points to fitted curve */
    auto [maxError, splitPoint] = computeMaxError(d, first, last, bezCurve, u);

    /*  If error not too large, try some reparameterization and iteration */
    if (maxError > error && maxError < iterationError) {
        for (int i = 0; i < MAX_ITERATIONS; i++) {
            reparameterize(u, bezCurve, d, first, last);
            bezCurve = bezier_regression(d, first, last, u, t0, t1);
            std::tie(maxError, splitPoint) = computeMaxError(d, first, last, bezCurve, u);
            if (maxError < error) break;
        }
    }
    return {bezCurve, maxError, splitPoint};
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
BezierCurve bezier_regression(const CurveYX& d, int first, int last, const std::vector<double>& uPrime,
                              const Vector& t0, const Vector& t1) {
    int nPts = last - first + 1;

    /* Compute the A's	*/
    std::vector<std::array<Vector, 2>> A; /* Precomputed rhs for eqn	*/
    A.reserve(nPts);
    for (int i = 0; i < nPts; i++) {
        double u = uPrime[i];
        A.push_back({t0 * B1(u), t1 * B2(u)});
    }

    /* Create the C and X matrices	*/
    Matrix2<double> C = {{{0.0, 0.0}, {0.0, 0.0}}};
    std::array<double, 2> X = {0.0, 0.0};

    for (int i = 0; i < nPts; i++) {
        C[0][0] += A[i][0].dot(A[i][0]);
        C[0][1] += A[i][0].dot(A[i][1]);
        /*					C[1][0] += V2Dot(&A[i][0], &A[i][1]); // C[1][0] = C[0][1]*/
        C[1][1] += A[i][1].dot(A[i][1]);

        Vector tmp = d[first + i];
        tmp -= d[first] * (B0(uPrime[i]) + B1(uPrime[i])) + d[last] * (B2(uPrime[i]) + B3(uPrime[i]));

        X[0] += A[i][0].dot(tmp);
        X[1] += A[i][1].dot(tmp);
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
    BezierCurve bezCurve = {d[first], 0, 0, d[last]};

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

/*
 *  ChordLengthParameterize :
 *	Assign parameter values to digitized points
 *	using relative distances between points.
 *
 * Point2 *d : Array of digitized points
 * int first, last : Indices defining region
 */
std::vector<double> chordLengthParameterize(const CurveYX& d, int first, int last) {
    int nPts = last - first + 1;
    std::vector<double> u(nPts, 0); /*  Parameterization		*/

    for (int i = 1; i < nPts; i++) u[i] = u[i - 1] + distance(d[i + first], d[i + first - 1]);

    double lastU = u[nPts - 1];
    for (int i = 1; i < nPts - 1; i++) u[i] /= lastU;
    u[nPts - 1] = 1.0;

    return u;
}

/*
 *  Reparameterize:
 *	Given set of points and their parameterization, try to find
 *   a better parameterization.
 *
 * Point2 *d : Array of digitized points
 * int first, last : Indices defining region
 * double *u : Current parameter values
 * BezierCurve bezCurve : Current fitted curve
 */
void reparameterize(std::vector<double>& u, const BezierCurve& bezCurve, const CurveYX& d, int first, int last) {
    int nPts = (int)u.size();
    for (int i = 0; i < nPts; i++) u[i] = findNewtonRaphsonRoot(bezCurve, d[linspace_int(i, first, last, nPts)], u[i]);
}

/*
 *  NewtonRaphsonRootFind :
 *	Use Newton-Raphson iteration to find better root.
 *
 * BezierCurve Q : Current fitted curve
 * Point2 P : Digitized point
 * double u : Parameter value for "P"
 */
double findNewtonRaphsonRoot(const BezierCurve& Q, const Point& P, double u) {
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
 * BezierCurve V : Control points
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
 * BezierCurve bezCurve : Fitted Bezier curve
 * double *u : Parameterization of points
 * int *splitPoint : Point of maximum error
 */
std::tuple<double, int> computeMaxError(const CurveYX& d, int first, int last, const BezierCurve& bezCurve,
                                        const std::vector<double>& u) {
    int i;
    double maxDist; /*  Maximum error		*/
    double dist;    /*  Current error		*/
    Point P;        /*  Point on curve		*/
    Vector v;       /*  Vector from point to curve	*/

    int splitPoint = (last - first + 1) / 2;
    maxDist = 0.0;
    for (i = first + 1; i < last; i++) {
        P = BezierII(std::vector<Point>(bezCurve.begin(), bezCurve.end()), u[i - first]);
        v = P - d[i];
        dist = v.squaredNorm();
        if (dist >= maxDist) {
            maxDist = dist;
            splitPoint = i;
        }
    }
    return {maxDist, splitPoint};
}
