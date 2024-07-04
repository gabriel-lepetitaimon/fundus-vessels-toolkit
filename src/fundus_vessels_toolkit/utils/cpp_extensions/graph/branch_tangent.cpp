#include "branch.h"
#include "fit_bezier.h"

/**
 * @brief Evaluate the tangent of a curve.
 *
 * This method compute the tangent of a curve defined by a list of points.
 * The tangent is computed by averaging the vectors between the current point
 * and its neighbors weighted by a gaussian kernel.
 *
 * @param curveYX A list of points defining the curve.
 * @param weighting A function that takes the distance between two points and returns a weight. This function must be
 * decreasing and positive.
 * @param evaluateAtID A list of indexes where the tangent should be evaluated.
 * If empty, the tangent is evaluated at each point.
 */
Point curve_tangent(const CurveYX& curveYX, std::size_t i, const std::function<float(float)>& weighting,
                    std::size_t stride, const bool forward, const bool backward, const std::size_t curveStart,
                    std::size_t curveEnd) {
    if (curveEnd == 0) curveEnd = curveYX.size();
    auto const& p0 = curveYX[i];
    Point tangent, dv;
    float sumWeights = 0, w = 0;
    std::size_t j = stride;

    if (backward && i >= stride + curveStart) {  // i-j >= curveStart  (because initially j=stride)
        do {
            std::size_t idx = i - j;
            auto const& p = curveYX[idx];
            w = weighting(distance(p0, p));
            dv += p * w;
            sumWeights += w;
            j += stride;
        } while (w >= 0.1 && i >= j + curveStart);  // i-j >= curveStart
        if (sumWeights != 0) tangent = curveYX[i] - dv / sumWeights;
    }

    if (forward && i + stride < curveEnd) {  // i+j < curveEnd (because initially j=stride)
        j = stride;
        sumWeights = 0;
        dv = {0, 0};
        do {
            std::size_t idx = i + j;
            auto const& p = curveYX[idx];
            w = weighting(distance(p0, p));
            dv += p * w;
            sumWeights += w;
            j += stride;
        } while (w >= 0.1 && i + j < curveEnd);
        if (sumWeights != 0) tangent += dv / sumWeights - curveYX[i];
    }

    return tangent.normalize();
}

Point curve_tangent(const CurveYX& curveYX, std::size_t i, const float std, const bool forward, const bool backward,
                    const std::size_t curveStart, const std::size_t curveEnd) {
    return curve_tangent(
        curveYX, i, [std](float d) { return std::exp(-d * d / (2 * std * std)); }, std::max(1, (int)floor(std / 3)),
        forward, backward, curveStart, curveEnd);
}

/**
 * @brief Evaluate the tangent of a curve.
 *
 * This method compute the tangent of a curve defined by a list of points.
 * The tangent is computed by averaging the vectors between the current point
 * and its neighbors weighted by a gaussian kernel.
 *
 * @param curveYX A list of points defining the curve.
 * @param kernelStd The standard deviation of the gaussian kernel for each point.
 *  This parameter must be of the same size as the curve if evaluateAtID is empty,
 *  or of the same size as evaluateAtID otherwise.
 * @param evaluateAtID A list of indexes where the tangent should be evaluated.
 * If empty, the tangent is evaluated at each point.
 */
std::vector<Point> curve_tangent(const CurveYX& curveYX, const std::vector<float>& kernelStd,
                                 const std::vector<int>& evaluateAtID) {
    const std::size_t curveSize = curveYX.size();
    bool evaluateAll = evaluateAtID.size() == 0;
    const std::size_t tangentSize = evaluateAll ? curveSize : evaluateAtID.size();

    std::map<int, std::vector<float>> kernelMap;

    std::vector<Point> tangents(tangentSize, {0, 0});
    // Compute the tangent at each point
    for (std::size_t pointI = 0; pointI < tangentSize; pointI++) {
        const std::size_t i = evaluateAll ? pointI : evaluateAtID[pointI];
        if (i < 0 || i >= curveSize) continue;

        const float std = std::max(kernelStd[pointI], 3.0f);
        tangents[pointI] = curve_tangent(curveYX, i, std, true, true);
    }

    return tangents;
}

/**
 * @brief Evaluate the tangent of a curve.
 *
 * This method compute the tangent of a curve defined by a list of points.
 * The tangent is computed by averaging the vectors between the current point
 * and its neighbors weighted by a gaussian kernel.
 *
 * @param curveYX A list of points defining the curve.
 * @param i The index of the point where the tangent should be evaluated.
 * @param gaussStd The standard deviation of the gaussian kernel.
 * @param forward If true, the tangent is computed in the forward direction.
 * @param backward If true, the tangent is computed in the backward direction.
 * @param curveStart The index of the first point of the curve.
 * @param curveEnd The index of the last point of the curve.
 *
 * @return The tangent of the curve at the given index.
 */
Point fast_curve_tangent(const CurveYX& curveYX, std::size_t i, const std::vector<float>& GaussKernel,
                         const bool forward, const bool backward, const std::size_t curveStart, std::size_t curveEnd) {
    const std::size_t K = GaussKernel.size();
    if (curveEnd == 0) curveEnd = curveYX.size();

    Point tangent;
    Point dv;
    float sumWeights = 0;

    if (backward) {
        for (std::size_t j = 1; j < K && i >= j + curveStart; j++) {  // j < K && i-j >= curveStart
            std::size_t idx = i - j;
            const float w = GaussKernel[j];
            dv += curveYX[idx] * w;
            sumWeights += w;
        }
        if (sumWeights != 0) tangent = curveYX[i] - dv / sumWeights;
    }

    if (forward) {
        sumWeights = 0;
        dv = {0, 0};
        for (std::size_t j = 1; j < K && j + i < curveEnd; j++) {
            std::size_t idx = i + j;
            const float w = GaussKernel[j];
            dv += curveYX[idx] * w;
            sumWeights += w;
        }
        if (sumWeights != 0) tangent += dv / sumWeights - curveYX[i];
    }

    return tangent.normalize();
}

/**
 * @brief Evaluate the tangent of a curve.
 *
 * This method compute the tangent of a curve defined by a list of points.
 * The tangent is computed by averaging the vectors between the current point
 * and its neighbors weighted by a gaussian kernel.
 *
 * @param curveYX A list of points defining the curve.
 * @param gaussStd The standard deviation of the gaussian kernel.
 * @param evaluateAtID A list of indexes where the tangent should be evaluated.
 * If empty, the tangent is evaluated at each point.
 */
std::vector<Point> fast_curve_tangent(const CurveYX& curveYX, const std::vector<float>& GaussKernel,
                                      const std::vector<int>& evaluateAtID) {
    const std::size_t curveSize = curveYX.size();
    bool evaluateAll = evaluateAtID.size() == 0;
    const std::size_t tangentSize = evaluateAll ? curveSize : evaluateAtID.size();

    std::vector<Point> tangents(tangentSize, {0, 0});
    // Compute the tangent at each point
    for (std::size_t pointI = 0; pointI < tangentSize; pointI++) {
        const std::size_t i = evaluateAll ? pointI : evaluateAtID[pointI];
        if (i < 0 || i >= curveSize) continue;
        tangents[pointI] = fast_curve_tangent(curveYX, i, GaussKernel);
    }

    return tangents;
}

/**
 * @brief Compute the inflection points of a curve.
 *
 * @param tangents A list of tangents defining the curve.
 * @param dthetaSmoothHalfKernel A kernel to smooth the derivative of the curve.
 * @param angle_threshold The threshold to consider a point as an inflection point.
 *
 * @return A list of indexes where the curve has an inflection point.
 */
std::vector<std::size_t> curve_inflections_points(const std::vector<Point>& tangents,
                                                  const std::vector<float>& dthetaSmoothHalfKernel,
                                                  double angle_threshold, std::size_t start, std::size_t end) {
    const std::size_t curveSize = tangents.size();
    if (curveSize < 7) return {};
    if (end == 0) end = curveSize - 1;

    std::vector<std::size_t> inflections;

    std::vector<float> theta;
    std::size_t size = end - start;
    theta.reserve(size);
    for (std::size_t i = start; i < size + start; i++) theta.push_back((float)tangents[i].angle());

    for (auto itTheta = theta.begin(); itTheta != theta.end() - 1; itTheta++) *itTheta = *(itTheta + 1) - *itTheta;
    theta.pop_back();

    const std::vector<float>& dtheta_smooth = movingAvg(theta, dthetaSmoothHalfKernel);

    for (auto itDTheta = dtheta_smooth.begin(); itDTheta != dtheta_smooth.end(); itDTheta++) {
        const std::size_t i = itDTheta - dtheta_smooth.begin();
        if (*itDTheta > angle_threshold)
            theta[i] = 1;
        else if (*itDTheta < -angle_threshold)
            theta[i] = -1;
        else
            theta[i] = 0;
    }

    int HALF_MED_FILTER = 2;
    const std::vector<float>& dtheta_med = medianFilter(theta, HALF_MED_FILTER);

    const auto UNKOWN_START = std::size_t(-1);
    std::size_t inflectionStart = UNKOWN_START;
    for (auto med = dtheta_med.begin(); med != dtheta_med.end(); med++) {
        std::size_t i = med - dtheta_med.begin();
        if (*med == 0 && *(med + 1) != 0) {
            inflectionStart = i;
        } else if (*med != 0 && *(med + 1) == 0) {
            if (inflectionStart != UNKOWN_START) {
                inflections.push_back((inflectionStart + i) / 2 + start);
                inflectionStart = UNKOWN_START;
            }
        }
    }

    return inflections;
}

BSpline bspline_regression(const CurveYX& curve, const CurveTangents& tangents, double bspline_max_error,
                           const std::vector<float>& inflection_halfKernel, double inflection_max_angle,
                           std::size_t start, std::size_t end) {
    if (curve.size() < 2) return {};
    if (end == 0) end = curve.size();

    // const std::vector<std::size_t>& inflections =
    //    curve_inflections_points(tangents, inflection_halfKernel, inflection_max_angle, start, end);

    BSpline bspline;
    // bspline.reserve(inflections.size() + 1);

    std::size_t first = start, last = 0;
    /*for (std::size_t j = 0; j < inflections.size(); j++) {
        last = inflections[j];
        bspline.push_back(std::get<0>(fit_bezier(curve, tangents, bspline_max_error, first, last)));
        first = inflections[j];
    }*/
    last = end - 1;
    bspline.push_back(std::get<0>(fit_bezier(curve, tangents, bspline_max_error, first, last)));

    return bspline;
}