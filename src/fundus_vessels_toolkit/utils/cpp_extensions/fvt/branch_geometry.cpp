#include "branch.h"

/**
 * @brief Extract geometry information from a vessel graph.
 *
 * @param branch_labels The branch labels of the vessel graph.
 * @param segmentation The segmentation of the vessel graph.
 *
 * @return A tuple containing:
 *  - a list of the yx coordinates of the branches pixels,
 *  - a list of the tangents of the branches (or the bezier node and control points),
 *  - the width of the branches
 */
std::tuple<std::vector<CurveTangents>, std::vector<Scalars>, std::vector<IntPointPairs>, std::vector<Scalars>,
           std::vector<Sizes>, std::vector<BSpline>>
extract_branches_geometry(std::vector<CurveYX> branch_curves, const Tensor2DAcc<bool>& segmentation,
                          std::map<std::string, double> options, bool assume_contiguous) {
    // --- Parse Options ---
    bool adaptative_tangents = get_if_exists(options, "adaptative_tangents", 1.0) > 0;
    float bspline_targetSqrError = pow(get_if_exists(options, "bspline_target_error", 0.0), 3);
    float curv_roots_percentileThreshold = get_if_exists(options, "curvature_roots_percentile_threshold", 0.1);

    bool return_calibre = get_if_exists(options, "return_calibre", 1.0) > 0;
    bool return_boundaries = get_if_exists(options, "return_boundaries", 0.0) > 0;
    bool return_curvature = get_if_exists(options, "return_curvature", 0.0) > 0;
    bool return_curvature_roots = get_if_exists(options, "return_curvature_roots", 1.0) > 0;
    bool extract_bspline = get_if_exists(options, "extract_bspline", 1.0) > 0;
    float bspline_max_gap = get_if_exists(options, "bspline_max_gap", 2.0);

    // --- Prepare result vectors ---
    auto nCurves = branch_curves.size();
    std::vector<CurveTangents> branchesTangents(nCurves);
    std::vector<Scalars> branchesCalibre;
    if (return_calibre) branchesCalibre.resize(nCurves);
    std::vector<IntPointPairs> branchesBoundaries;
    if (return_boundaries) branchesBoundaries.resize(nCurves);

    std::vector<Scalars> branchesCurvature;
    branchesCurvature.resize(nCurves);
    std::vector<Sizes> branchesCurvatureRoots;
    branchesCurvatureRoots.resize(nCurves);

    std::vector<BSpline> branchesBSpline;
    if (extract_bspline) branchesBSpline.resize(nCurves);

// --- Extract geometry of each branches ---
#pragma omp parallel for
    for (std::size_t curveI = 0; curveI < nCurves; curveI++) {
        auto const& curve = branch_curves[curveI];

        if (curve.size() == 0) continue;
        if (curve.size() == 1) {
            branchesTangents[curveI] = {Point(0, 0)};
            if (return_curvature) branchesCurvature[curveI] = {0};
            if (return_calibre) branchesCalibre[curveI] = {INVALID_CALIBRE};
            if (return_boundaries) branchesBoundaries[curveI] = {{curve[0], curve[0]}};
            if (extract_bspline) branchesBSpline[curveI] = {BSpline({{curve[0], curve[0], curve[0], curve[0]}})};
            continue;
        }

        // Initialize the returned vectors
        auto& tangents = branchesTangents[curveI];
        auto& branchCurvature = branchesCurvature[curveI];
        auto& branchCurvatureRoots = branchesCurvatureRoots[curveI];

        tangents.resize(curve.size());
        branchCurvature.reserve(curve.size());
        branchCurvatureRoots.reserve(16);
        if (return_calibre) branchesCalibre[curveI].resize(curve.size());
        if (return_boundaries) branchesBoundaries[curveI].resize(curve.size());

        // Split the branch curve into contiguous curves
        auto const& contiguousCurvesStartEnd =
            assume_contiguous ? std::list<SizePair>({{0, curve.size()}}) : split_contiguous_curves(curve);
        for (auto const& [start, end] : contiguousCurvesStartEnd) {
            for (std::size_t i = start; i < end; i++) {
                // Compute the tangent of the curve
                auto const& tangent = fast_curve_tangent(curve, i, TANGENT_HALF_GAUSS, true, true, start, end);
                branchesTangents[curveI][i] = tangent;

                // Compute the calibre of the curve
                if (return_calibre || return_boundaries || adaptative_tangents) {
                    auto boundaries = fast_branch_boundaries(curve, i, segmentation, tangent);

                    if (return_calibre || adaptative_tangents) {
                        auto calibre = fast_branch_calibre(boundaries[0], boundaries[1], tangent);

                        if (adaptative_tangents && calibre == calibre) {
                            const auto& refinedTangent =
                                adaptative_curve_tangent(curve, i, calibre, true, true, start, end);
                            branchesTangents[curveI][i] = refinedTangent;
                            if (return_boundaries || return_calibre)
                                boundaries = fast_branch_boundaries(curve, i, segmentation, refinedTangent);
                            if (return_calibre)
                                branchesCalibre[curveI][i] =
                                    fast_branch_calibre(boundaries[0], boundaries[1], refinedTangent);
                        } else if (return_calibre)
                            branchesCalibre[curveI][i] = calibre;
                    }

                    if (return_boundaries) branchesBoundaries[curveI][i] = boundaries;
                }
            }

            // Compute the curvature of the curve
            if (return_curvature || return_curvature_roots || extract_bspline) {
                auto const& curvatures = tangents_to_curvature(tangents, true, 5, start, end);
                branchCurvature.insert(branchCurvature.end(), curvatures.begin(), curvatures.end());
                branchCurvature.push_back(0);

                auto const& inflections_roots =
                    curve_inflections_points(curvatures, curv_roots_percentileThreshold, start);
                branchCurvatureRoots.insert(branchCurvatureRoots.end(), inflections_roots.begin(),
                                            inflections_roots.end());
            }
        }

        if (extract_bspline) {
            // Merge contiguous curves if the gap is small enough
            std::list<SizePair> mergedStartEnd;
            Point previousEndP;
            for (auto const& [start, end] : contiguousCurvesStartEnd) {
                Point startP = curve[start];
                if (mergedStartEnd.empty() || distance(startP, previousEndP) > bspline_max_gap)
                    mergedStartEnd.push_back({start, end});
                else
                    mergedStartEnd.back()[1] = end;
                previousEndP = curve[end - 1];
            }

            // Compute the BSpline over the merged contiguous curves using the already computed inflections roots
            for (auto const& [start, end] : mergedStartEnd) {
                auto const& bspline_curve =
                    bspline_regression(curve, tangents, branchCurvatureRoots, bspline_targetSqrError, start, end);
                branchesBSpline[curveI].insert(branchesBSpline[curveI].end(), bspline_curve.begin(),
                                               bspline_curve.end());
            }
        }
    }

    if (!return_curvature) branchesCurvature.clear();
    if (!return_curvature_roots) branchesCurvatureRoots.clear();

    return std::make_tuple(branchesTangents, branchesCalibre, branchesBoundaries, branchesCurvature,
                           branchesCurvatureRoots, branchesBSpline);
}

BSpline bspline_regression(const CurveYX& curve, const CurveTangents& tangents, double targetSqrError,
                           const float KpercentileThreshold, std::size_t start, std::size_t end) {
    if (curve.size() < 2) return {};
    if (end == 0) end = curve.size();

    auto const& curvatures = tangents_to_curvature(tangents, true, 5, start, end);
    return bspline_regression(curve, tangents, curvatures, targetSqrError, KpercentileThreshold, start, end);
}

BSpline bspline_regression(const CurveYX& curve, const CurveTangents& tangents, const Scalars& curvatures,
                           double targetSqrError, const float KpercentileThreshold, std::size_t start,
                           std::size_t end) {
    if (curve.size() < 2) return {};
    if (end == 0) end = curve.size();

    auto const& [bezier, maxError, sqrErrors, u] = fit_bezier(curve, tangents, targetSqrError, start, end - 1);
    if (KpercentileThreshold > 0 && maxError > targetSqrError) {
        const std::vector<std::size_t>& splitCandidate =
            curve_inflections_points(curvatures, KpercentileThreshold, start);
        return iterative_fit_bspline(curve, tangents, bezier, u, sqrErrors, splitCandidate, targetSqrError, start,
                                     end - 1);
    } else
        return {bezier};
}

BSpline bspline_regression(const CurveYX& curve, const CurveTangents& tangents,
                           const std::vector<std::size_t>& splitCandidate, double targetSqrError, std::size_t start,
                           std::size_t end) {
    if (curve.size() < 2) return {};
    if (end == 0) end = curve.size();

    auto const& [bezier, maxError, sqrErrors, u] = fit_bezier(curve, tangents, targetSqrError, start, end - 1);
    if (maxError > targetSqrError) {
        return iterative_fit_bspline(curve, tangents, bezier, u, sqrErrors, splitCandidate, targetSqrError, start,
                                     end - 1);
    } else
        return {bezier};
}

/*
 * Fit a bezier curve to a curve using the least square method.
 * The curve is split at the split candidates which has the maximum tangent angle error.
 * The process is repeated recursively until the maximum euclidean error between each point of the curve and its bezier
 * approximation is less than the given error.
 *
 * @param d The coordinates of the curve to fit.
 * @param tangents The normalized tangents of for each point of the curve d.
 * @param bezier A bezier curve previously fitted to the segment [first, last] of the curve but who didn't reach the
 * desired error.
 * @param u The best parameterization of this bezier curve.
 * @param splitCandidates The list of the split candidates of the curve (the index are relative to d not to u).
 * @param error The targeted maximum euclidean error between each point of the curve and its bezier approximation.
 * @param first The first index of the segment of the curve to fit.
 * @param last The last index of the segment of the curve to fit.
 *
 * @return The Bezier-Spline which approximate the segment [first, last] of the curve d with a maximum error less than
 * the given error (provided that enough split candidates are available).
 */
BSpline iterative_fit_bspline(const CurveYX& d, const std::vector<Point>& tangents, const BezierCurve& bezier,
                              const std::vector<double>& u, const std::vector<double>& sqrErrors,
                              const std::vector<std::size_t>& splitCandidates, double error, std::size_t first,
                              std::size_t last) {
    std::vector<std::size_t> validSplits;
    validSplits.reserve(splitCandidates.size());
    for (auto const& split : splitCandidates) {
        if (split > first && split < last) validSplits.push_back(split);
    }
    if (validSplits.size() == 0) return {bezier};
    const std::size_t nSplit = validSplits.size();
    const std::size_t curveLength = last - first;

    // Find the best split position: the closest candidate to the tangent error barycenter
    std::size_t split = 0;
    {
        // Initialize temporary storage of the cumulated area of tangent error
        std::vector<double> tanErrors;
        tanErrors.reserve(curveLength);
        std::vector<double> cumulatedTanErrors;
        cumulatedTanErrors.reserve(curveLength);
        double totalTanError = 0;

        std::vector<double> cumulatedSqrErrors;
        cumulatedSqrErrors.reserve(curveLength);
        double totalSqrError = 0;

        // Compute the tangent to the bezier curve at each sampling point
        auto const& bezierTangents = evaluate_bezier_tangent(bezier, u);

        // Compute the cumulated square euclidian error and cumulated tangent cosine error at each sampling point
        for (std::size_t i = 0; i < curveLength; i++) {
            auto const& Tcurve = tangents[i + first];
            auto const& Tbezier = bezierTangents[i].normalize();
            const float tangentCosineError = 1 - Tcurve.dot(Tbezier);
            tanErrors.push_back(tangentCosineError);

            totalTanError += tangentCosineError;
            cumulatedTanErrors.push_back(totalTanError);

            auto const& sqrError = sqrErrors[i];
            totalSqrError += sqrError;
            cumulatedSqrErrors.push_back(totalSqrError);
        }

        // Find the split closest to the optimal split
        auto distToOptimal = [&](std::size_t i) {
            return std::abs((2 * cumulatedSqrErrors[i] + sqrErrors[i]) / totalSqrError - 1) +
                   std::abs((2 * cumulatedTanErrors[i] + tanErrors[i]) / totalTanError - 1);
        };
        std::size_t iSplit = 0;
        float minDist = distToOptimal(validSplits.front() - first);
        for (; iSplit < nSplit - 1; iSplit++) {
            const float dist = distToOptimal(validSplits[iSplit + 1] - first);
            if (dist > minDist) break;
            minDist = dist;
        }
        split = validSplits[iSplit];
    }

    // Split the curve at the maximum error and recursively fit the two parts
    BSpline bspline;
    for (auto const& [split_first, split_last] : std::array<SizePair, 2>{{{first, split}, {split, last}}}) {
        auto const& [bezier, maxError, sqrErrors, u] = fit_bezier(d, tangents, error, split_first, split_last);
        if (maxError > error) {
            auto const& bspline1 = iterative_fit_bspline(d, tangents, bezier, u, sqrErrors, splitCandidates, error,
                                                         split_first, split_last);
            bspline.insert(bspline.end(), bspline1.begin(), bspline1.end());
        } else
            bspline.push_back(bezier);
    }
    return bspline;
}