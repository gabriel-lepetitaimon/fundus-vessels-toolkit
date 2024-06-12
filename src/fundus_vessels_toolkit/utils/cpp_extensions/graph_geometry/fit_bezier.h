#ifndef _FITCURVE_H_
#define _FITCURVE_H_

#include "../common.h"

using BezierCurve = std::array<Point, 4>;
using BSpline = std::vector<BezierCurve>;

#define MAXPOINTS 1000 /* The most points you can have */

BSpline fit_bspline(const CurveYX& d, const std::vector<Point>& tangent, double error, int first = 0, int last = -1);
std::tuple<BezierCurve, double, int> fit_bezier(const CurveYX& d, const std::vector<Point>& tangent, double error,
                                                int first = 0, int last = -1);

BezierCurve bezier_regression(const CurveYX& d, int first, int last, const std::vector<double>& uPrime,
                              const Vector& t0, const Vector& t1);

std::vector<double> chordLengthParameterize(const CurveYX& d, int first, int last);
void reparameterize(std::vector<double>& u, const BezierCurve& bezCurve, const CurveYX& d, int first, int last);
double findNewtonRaphsonRoot(const BezierCurve& Q, const Point& P, double u);

Point BezierII(std::vector<Point> V, double t);
double B0(double u), B1(double u), B2(double u), B3(double u);

std::tuple<double, int> computeMaxError(const CurveYX& d, int first, int last, const BezierCurve& bezCurve,
                                        const std::vector<double>& u);

#endif /* _FITCURVE_H_ */