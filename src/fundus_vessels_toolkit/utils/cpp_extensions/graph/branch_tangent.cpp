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
std::vector<std::size_t> curve_inflections_points(const std::vector<Point>& tangents, const float dK_threshold,
                                                  const std::vector<float>& tangentSmoothKernel, std::size_t start,
                                                  std::size_t end) {
    if (end == 0) end = tangents.size() - 1;
    const std::size_t curveSize = end - start - 2;  // -2 because we need 2 points to compute the second derivative dK

    // Check if enough points to compute the inflection points (median filtering size of quantize_triband() is 5)
    if (curveSize < 5) return {};

    // --- Compute curvature K ---
    auto const& K = tangents_to_curvature(tangents, tangentSmoothKernel, start, end);

    // --- Compute derivative of curvature dK ---
    auto const& dK = diff(K);

    // --- Find local minimum of dK ---
    const float threshold = percentile(abs(dK), dK_threshold * 100);
    auto const& K_quant = quantize_triband(dK, -threshold, threshold);

    std::list<SizePair> minimum_intervals;
    // Search for the first decreasing point
    std::size_t i = 0;
    while (i < curveSize && K_quant[i] != -1) i++;
    // Search for the minimum intervals
    for (; i < curveSize - 1; i++) {
        if (K_quant[i + 1] != -1) {
            std::size_t j;
            for (j = i + 1; j < curveSize; j++) {
                if (K_quant[j] == 1)
                    break;
                else if (K_quant[j] == -1)
                    i = j;
            }
            minimum_intervals.push_back({i, j});

            // Search for the next decreasing point
            while (j < curveSize && K_quant[j] != -1) j++;
            i = j;
        }
    }
    // Take the center of the intervals
    std::vector<std::size_t> inflections;
    inflections.reserve(minimum_intervals.size());
    for (auto const& interval : minimum_intervals) inflections.push_back((interval[0] + interval[1] + 1) / 2);
    return inflections;
}

Point smooth_tangents(const CurveTangents& tangents, std::size_t i, const std::vector<float> weight,
                      std::size_t curveStart, std::size_t curveEnd) {
    if (curveEnd == 0) curveEnd = tangents.size();
    const std::size_t K = weight.size();

    Point t = tangents[i] * weight[0];
    for (std::size_t j = 1; j < K && i >= j + curveStart; j++) t += tangents[i - j] * weight[j];
    for (std::size_t j = 1; j < K && i + j < curveEnd; j++) t += tangents[i + j] * weight[j];

    return t.normalize();
}

std::vector<float> tangents_to_curvature(const CurveTangents& tangents, const std::vector<float> weight,
                                         std::size_t start, std::size_t end) {
    if (end == 0) end = tangents.size();
    const std::size_t outSize = end - start - 1;
    std::vector<float> curvatures;
    curvatures.reserve(outSize);

    Point lastT = smooth_tangents(tangents, start, weight, start, end);
    for (std::size_t i = 0; i < outSize; i++) {
        auto const& t = smooth_tangents(tangents, i + start + 1, weight, start, end);
        curvatures.push_back((t - lastT).norm());
        lastT = t;
    }

    return curvatures;
}