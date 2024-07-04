#ifndef _FITCURVE_H_
#define _FITCURVE_H_

#include "common.h"

using BezierCurve = std::array<Point, 4>;
using BSpline = std::vector<BezierCurve>;

#define MAXPOINTS 1000 /* The most points you can have */

BSpline fit_bspline(const CurveYX& d, const std::vector<Point>& tangent, double error, std::size_t first = 0,
                    std::size_t last = 0);
std::tuple<BezierCurve, double, int> fit_bezier(const CurveYX& d, const std::vector<Point>& tangent, double error,
                                                std::size_t first = 0, std::size_t last = 0);

BezierCurve bezier_regression(const CurveYX& d, std::size_t first, std::size_t last, const std::vector<double>& uPrime,
                              const Vector& t0, const Vector& t1);

std::vector<double> chordLengthParameterize(const CurveYX& d, std::size_t first, std::size_t last);
void reparameterize(std::vector<double>& u, const BezierCurve& bezCurve, const CurveYX& d, std::size_t first,
                    std::size_t last);
double findNewtonRaphsonRoot(const BezierCurve& Q, const Point& P, double u);

Point BezierII(std::vector<Point> V, double t);
double B0(double u), B1(double u), B2(double u), B3(double u);

std::tuple<double, std::size_t> computeMaxError(const CurveYX& d, std::size_t first, std::size_t last,
                                                const BezierCurve& bezCurve, const std::vector<double>& u);

torch::Tensor bspline_to_tensor(const BSpline& bspline);
std::vector<torch::Tensor> bsplines_to_tensor(const std::vector<BSpline>& bsplines);

#endif /* _FITCURVE_H_ */