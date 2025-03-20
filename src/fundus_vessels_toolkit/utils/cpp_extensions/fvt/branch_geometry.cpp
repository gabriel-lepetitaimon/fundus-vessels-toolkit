#include "branch.h"

std::tuple<std::vector<CurveYX>, std::vector<Sizes>, std::vector<CurveTangents>, std::vector<Scalars>,
           std::vector<IntPointPairs>, std::vector<Scalars>, std::vector<Sizes>, std::vector<BSpline>>
extract_branches_geometry(std::vector<CurveYX>& branch_curves, const Tensor2DAcc<bool>& segmentation,
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
    float split_on_gaps = get_if_exists(options, "split_on_gaps", 2.0);

    // --- Prepare result vectors ---
    auto nCurves = branch_curves.size();
    std::vector<CurveYX> branchesCurves(nCurves);
    std::vector<Sizes> branchesCurvesSplits(nCurves);
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
        auto const& skelCurve = branch_curves[curveI];
        int skelCurveSize = (int)skelCurve.size();

        if (skelCurveSize == 0) continue;
        if (skelCurveSize == 1) {
            branchesTangents[curveI] = {Point(0, 0)};
            if (return_curvature) branchesCurvature[curveI] = {0};
            if (return_calibre) branchesCalibre[curveI] = {INVALID_CALIBRE};
            if (return_boundaries) branchesBoundaries[curveI] = {{skelCurve[0], skelCurve[0]}};
            if (extract_bspline)
                branchesBSpline[curveI] = {BSpline({{skelCurve[0], skelCurve[0], skelCurve[0], skelCurve[0]}})};
            continue;
        }

        // Initialize temporary vectors
        CurveYX curve;
        std::set<std::size_t> curveSplitsSet;
        std::vector<Point> tangents;
        std::vector<float> curvatures;
        std::vector<std::size_t> curvatureRoots;
        std::vector<float> calibres;
        std::vector<IntPointPair> boundaries;

        curve.reserve(skelCurveSize);
        tangents.reserve(skelCurveSize);
        curvatures.reserve(skelCurveSize);
        curvatureRoots.reserve(16);
        if (return_calibre) calibres.reserve(skelCurveSize);
        if (return_boundaries) boundaries.reserve(skelCurveSize);

        // === Split the branch curve into contiguous curves ===
        auto const& contiguousCurvesStartEnd =
            assume_contiguous ? std::list<SizePair>({{0, skelCurve.size()}}) : split_contiguous_curves(skelCurve);

        // === Compute primary properties of the branch: tangents, boundaries, calibre ===
        for (auto const& [start, end] : contiguousCurvesStartEnd) {
            for (std::size_t i = start; i < end; i++) {
                // Compute the tangent of the curve
                auto tangent = fast_curve_tangent(skelCurve, i, TANGENT_HALF_GAUSS, true, true, start, end);

                // Compute the calibre of the curve
                if (return_calibre || return_boundaries || adaptative_tangents) {
                    auto bound = fast_branch_boundaries(skelCurve, i, segmentation, tangent);
                    float calibre;

                    if (return_calibre || adaptative_tangents) {
                        calibre = fast_branch_calibre(bound[0], bound[1], tangent);

                        if (adaptative_tangents && calibre == calibre) {
                            tangent = adaptative_curve_tangent(skelCurve, i, calibre, true, true, start, end);
                            bound = fast_branch_boundaries(skelCurve, i, segmentation, tangent);
                            calibre = fast_branch_calibre(bound[0], bound[1], tangent);
                        }
                    }

                    // Check the validity of the branch pixel: ...
                    if (!is_valid_boundaries(skelCurve[i], bound)) {
                        // ... if invalid skip the pixel, ...
                        curveSplitsSet.insert(curve.size());
                        continue;
                    }
                    // ... otherwise, store the tangent calibre, boundaries and pixel position
                    tangents.push_back(tangent);
                    if (return_calibre) calibres.push_back(calibre);
                    if (return_boundaries) boundaries.push_back(bound);
                }
                curve.push_back(skelCurve[i]);
            }
            curveSplitsSet.insert(curve.size());
        }
        // erase the first split if it is 0
        if (curveSplitsSet.size() != 0 and *curveSplitsSet.begin() == 0) curveSplitsSet.erase(curveSplitsSet.begin());

        // === Remove very small contiguous region of the valid curve and save all primary properties ===
        auto& finalCurve = branchesCurves[curveI];
        auto& finalTangents = branchesTangents[curveI];
        auto& finalCalibre = branchesCalibre[curveI];
        auto& finalBoundaries = branchesBoundaries[curveI];
        std::set<std::size_t> refinedCurveSplits;

        finalCurve.reserve(curve.size());
        finalTangents.reserve(tangents.size());
        if (return_calibre) finalCalibre.reserve(calibres.size());
        if (return_boundaries) finalBoundaries.reserve(boundaries.size());

        std::size_t start = 0;
        for (auto it = curveSplitsSet.begin(); it != curveSplitsSet.end(); it++) {
            auto end = *it;
            if (end - start > 2) {
                finalCurve.insert(finalCurve.end(), curve.begin() + start, curve.begin() + end);
                finalTangents.insert(finalTangents.end(), tangents.begin() + start, tangents.begin() + end);
                if (return_calibre)
                    finalCalibre.insert(finalCalibre.end(), calibres.begin() + start, calibres.begin() + end);
                if (return_boundaries)
                    finalBoundaries.insert(finalBoundaries.end(), boundaries.begin() + start, boundaries.begin() + end);
                refinedCurveSplits.insert(finalCurve.size());
            }
            start = end;
        }
        if (finalCurve.size() == 0) continue;

        // erase the last split (i.e. refinedCurve.size())
        refinedCurveSplits.erase(--refinedCurveSplits.end());

        // === Search for gaps in the branch curve and ===
        std::set<std::size_t> bsplineNodeCandidates;
        auto& curveSplits = branchesCurvesSplits[curveI];
        for (const auto& split : refinedCurveSplits) {
            if (distance(finalCurve[split - 1], finalCurve[split]) > split_on_gaps) {
                curveSplits.push_back(split);
            } else {
                // If the gap is small, ignore it but consider it as a node candidate for the bspline
                // I'm not sure its a good idea as calibre errors (and their induced gap) are not consistent
                bsplineNodeCandidates.insert(split - 1);
                bsplineNodeCandidates.insert(split);
            }
        }

        // === Compute secondary properties of the branch: curvatures, bsplines ===
        if (return_curvature || return_curvature_roots || extract_bspline) {
            auto& curvature = branchesCurvature[curveI];
            auto& curvatureRoots = branchesCurvatureRoots[curveI];
            curvature.reserve(finalCurve.size());
            curvatureRoots.reserve(16);

            std::size_t start = 0;
            if (curveSplits.size() == 0 || curveSplits.back() != finalCurve.size())
                curveSplits.push_back(finalCurve.size());
            for (auto splitIt = curveSplits.begin(); splitIt != curveSplits.end(); splitIt++) {
                auto end = *splitIt;

                // Compute the curvature of the curve
                auto const& curvatures = tangents_to_curvature(tangents, true, 5, start, end);
                curvature.insert(curvature.end(), curvatures.begin(), curvatures.end());
                curvature.push_back(0);

                auto const& inflections_roots =
                    curve_inflections_points(curvatures, curv_roots_percentileThreshold, start);
                curvatureRoots.insert(curvatureRoots.end(), inflections_roots.begin(), inflections_roots.end());
                bsplineNodeCandidates.insert(inflections_roots.begin(), inflections_roots.end());

                start = end;
            }

            if (extract_bspline) {
                // Compute the BSpline over the merged contiguous curves using the already computed inflections roots
                std::vector<std::size_t> nodeCandidates(bsplineNodeCandidates.begin(), bsplineNodeCandidates.end());
                start = 0;
                for (auto splitIt = curveSplits.begin(); splitIt != curveSplits.end(); splitIt++) {
                    auto end = *splitIt;

                    auto const& bspline_curve = bspline_regression(finalCurve, finalTangents, nodeCandidates,
                                                                   bspline_targetSqrError, start, end);
                    branchesBSpline[curveI].insert(branchesBSpline[curveI].end(), bspline_curve.begin(),
                                                   bspline_curve.end());

                    start = end;
                }
            }
            curveSplits.pop_back();
        }
    }

    if (!return_curvature) branchesCurvature.clear();
    if (!return_curvature_roots) branchesCurvatureRoots.clear();

    return std::make_tuple(branchesCurves, branchesCurvesSplits, branchesTangents, branchesCalibre, branchesBoundaries,
                           branchesCurvature, branchesCurvatureRoots, branchesBSpline);
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
 * The process is repeated recursively until the maximum euclidean error between each point of the curve and its
 * bezier approximation is less than the given error.
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
 * @return The Bezier-Spline which approximate the segment [first, last] of the curve d with a maximum error less
 * than the given error (provided that enough split candidates are available).
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
            return std::abs(cumulatedSqrErrors[i] / totalSqrError - .5) +
                   std::abs(cumulatedTanErrors[i] / totalTanError - .5);
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