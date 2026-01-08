#ifndef _FITCURVE_H_
#define _FITCURVE_H_

#include "common.h"

using BezierCubic = std::array<Point, 4>;
using BSpline = std::vector<BezierCubic>;

#define MAXPOINTS 1000 /* The most points you can have */

std::tuple<BezierCubic, double, std::vector<double>, std::vector<double>> fit_bezier(
    const CurveYX& d, const PointList& tangent, double targetSqrError, std::size_t first = 0, std::size_t last = 0);

BezierCubic bezier_regression(const CurveYX& d, std::size_t first, std::size_t last, const std::vector<double>& uPrime,
                              const Vector& t0, const Vector& t1);

PointList evaluate_bezier(const BezierCubic& bezCurve, const std::vector<double>& u);
Point evaluate_bezier(const BezierCubic& bezCurve, const double& u);
PointList evaluate_bezier_tangent(const BezierCubic& bezCurve, const std::vector<double>& u);

Point infer_bezier_t0(const Point& p0, const Point& p1, const Point& t1, double smoothness = 0.5);

std::tuple<PointList, std::vector<double>> discretizeBezier(const BezierCubic& bezCurve);
void _recursiveDiscretizeBezier(const BezierCubic& curveSegment, double u_start, double u_end, PointList& points,
                                std::vector<double>& us);
std::pair<BezierCubic, BezierCubic> subdivideBezier(const BezierCubic& curve, const double& u);
std::vector<double> chordLengthParameterize(const CurveYX& d, std::size_t first, std::size_t last);
void reparameterize(std::vector<double>& u, const BezierCubic& bezCurve, const CurveYX& d, std::size_t first,
                    std::size_t last);
double findNewtonRaphsonRoot(const BezierCubic& Q, const Point& P, double u);

Point BezierII(std::vector<Point> V, double t);
double B0(double u), B1(double u), B2(double u), B3(double u);

template <unsigned long N>
Point bezierPolynomialTriangle(std::array<Point, N> V, double t) {
    for (std::size_t i = 1; i <= N - 1; i++) {
        for (std::size_t j = 0; j <= N - 1 - i; j++) {
            V[j] = V[j] * (1.0 - t) + V[j + 1] * t;
        }
    }
    return V[0];
}

std::tuple<std::vector<double>, double, std::size_t> computeMaxError(const CurveYX& d, std::size_t first,
                                                                     std::size_t last, const BezierCubic& bezCurve,
                                                                     const std::vector<double>& u);

torch::Tensor bspline_to_tensor(const BezierCubic& bspline);
torch::Tensor bspline_to_tensor(const BSpline& bspline);
std::vector<torch::Tensor> bsplines_to_tensor(const std::vector<BSpline>& bsplines);

#endif /* _FITCURVE_H_ */