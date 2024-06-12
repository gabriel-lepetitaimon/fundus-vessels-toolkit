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
 * @param gaussStd The standard deviation of the gaussian kernel.
 * @param evaluateAtID A list of indexes where the tangent should be evaluated.
 * If empty, the tangent is evaluated at each point.
 */
Point fast_curve_tangent(const CurveYX& curveYX, std::size_t i, const std::vector<float>& GaussKernel,
                         const bool forward, const bool backward) {
    const std::size_t K = GaussKernel.size(), curveSize = curveYX.size();

    Point tangent;
    Point dv;
    float sumWeights = 0;

    if (backward) {
        for (std::size_t j = 1; j < K && i >= j; j++) {
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
        for (std::size_t j = 1; j < K && j < curveSize - i; j++) {
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

std::vector<int> curve_inflections_points(const std::vector<Point>& tangents,
                                          const std::vector<float>& dthetaSmoothHalfKernel, float angle_threshold) {
    const std::size_t curveSize = tangents.size();
    if (curveSize < 7) return {};

    std::vector<int> inflections;

    std::vector<float> theta(curveSize);
    theta.reserve(curveSize);
    for (std::size_t i = 1; i < curveSize - 1; i++) theta.push_back((float)tangents[i].angle());

    std::vector<float> dtheta(curveSize);
    dtheta.reserve(curveSize - 2);
    for (std::size_t i = 1; i < curveSize - 1; i++) dtheta.push_back((theta[i + 1] - theta[i - 1]) / 2);
    const std::vector<float>& dtheta_smooth = movingAvg(dtheta, dthetaSmoothHalfKernel);

    for (std::size_t i = 0; i < curveSize; i++) {
        float v = 0;
        if (dtheta_smooth[i] > angle_threshold)
            v = 1;
        else if (dtheta_smooth[i] < -angle_threshold)
            v = -1;
        dtheta[i] = v;
    }

    int HALF_MED_FILTER = 2;
    const std::vector<float>& dtheta_med = medianFilter(dtheta, HALF_MED_FILTER);

    int inflectionStart = -1;
    for (std::size_t i = 0; i < dtheta_med.size() - 1; i++) {
        int med0 = dtheta_med[i], med1 = dtheta_med[i + 1];
        if (med0 == 0 && med1 != 0) {
            inflectionStart = i;
        } else if (med0 != 0 && med1 == 0) {
            if (inflectionStart != -1) {
                inflections.push_back((int)round((inflectionStart + i) / 2));
                inflectionStart = -1;
            }
        }
    }

    return inflections;
}
