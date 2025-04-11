#ifndef _FITCURVE_H_
#define _FITCURVE_H_

#include "common.h"

using BezierCurve = std::array<Point, 4>;
using BSpline = std::vector<BezierCurve>;

#define MAXPOINTS 1000 /* The most points you can have */

std::tuple<BezierCurve, double, std::vector<double>, std::vector<double>> fit_bezier(
    const CurveYX& d, const PointList& tangent, double targetSqrError, std::size_t first = 0, std::size_t last = 0);

BezierCurve bezier_regression(const CurveYX& d, std::size_t first, std::size_t last, const std::vector<double>& uPrime,
                              const Vector& t0, const Vector& t1);

PointList evaluate_bezier(const BezierCurve& bezCurve, const std::vector<double>& u);
PointList evaluate_bezier_tangent(const BezierCurve& bezCurve, const std::vector<double>& u);

std::vector<double> chordLengthParameterize(const CurveYX& d, std::size_t first, std::size_t last);
void reparameterize(std::vector<double>& u, const BezierCurve& bezCurve, const CurveYX& d, std::size_t first,
                    std::size_t last);
double findNewtonRaphsonRoot(const BezierCurve& Q, const Point& P, double u);

Point BezierII(std::vector<Point> V, double t);
double B0(double u), B1(double u), B2(double u), B3(double u);

template <unsigned long N>
Point BezierPolynomialTriangle(std::array<Point, N> V, double t) {
    for (std::size_t i = 1; i <= N - 1; i++) {
        for (std::size_t j = 0; j <= N - 1 - i; j++) {
            V[j] = V[j] * (1.0 - t) + V[j + 1] * t;
        }
    }
    return V[0];
}

std::tuple<std::vector<double>, double, std::size_t> computeMaxError(const CurveYX& d, std::size_t first,
                                                                     std::size_t last, const BezierCurve& bezCurve,
                                                                     const std::vector<double>& u);

torch::Tensor bspline_to_tensor(const BezierCurve& bspline);
torch::Tensor bspline_to_tensor(const BSpline& bspline);
std::vector<torch::Tensor> bsplines_to_tensor(const std::vector<BSpline>& bsplines);

#endif /* _FITCURVE_H_ */