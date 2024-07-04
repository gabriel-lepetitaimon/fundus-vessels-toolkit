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
std::tuple<std::vector<CurveTangents>, std::vector<std::vector<float>>, std::vector<BSpline>> extract_branches_geometry(
    std::vector<CurveYX> branch_curves, const Tensor2DAccessor<bool>& segmentation,
    std::map<std::string, double> options, bool assume_contiguous) {
    // --- Parse Options ---
    bool compute_calibre = options.count("compute_calibre") ? options["compute_calibre"] : true;
    bool adaptative_tangent = options.count("adaptative_tangent") ? options["adaptative_tangent"] : true;
    int bspline_max_error = options.count("bspline_max_error") ? options["bspline_max_error"] : 0;
    float bspline_dK_threshold = options.count("bspline_dK_threshold") ? options["bspline_dK_threshold"] : 0.1;
    // double dtheta_kernel_std = options.count("dtheta_kernel_std") ? options["dtheta_kernel_std"] : 3;
    // double inflection_max_angle = options.count("inflection_max_angle") ? options["inflection_max_angle"] : 1.5;

    // --- Prepare result vectors ---
    auto nCurves = branch_curves.size();
    std::vector<CurveTangents> branchesTangents(nCurves);
    std::vector<std::vector<float>> branchesCalibre(nCurves);
    std::vector<BSpline> branchesBSpline(nCurves);

// --- Extract geometry of each branches ---
#pragma omp parallel for
    for (std::size_t curveI = 0; curveI < nCurves; curveI++) {
        auto const& curve = branch_curves[curveI];

        // Get references to the results vectors...
        CurveTangents& tangents = branchesTangents[curveI];
        std::vector<float>& calibres = branchesCalibre[curveI];
        BSpline& bspline = branchesBSpline[curveI];
        // ... and initialize them
        tangents.resize(curve.size());
        if (compute_calibre) calibres.resize(curve.size());

        // Split the branch curve into contiguous curves
        auto const& contiguousCurvesStartEnd =
            assume_contiguous ? std::list<SizePair>({{0, curve.size()}}) : splitInContiguousCurves(curve);
        for (auto const& [start, end] : contiguousCurvesStartEnd) {
            for (std::size_t i = start; i < end; i++) {
                // Compute the tangent of the curve
                auto const& tangent = fast_curve_tangent(curve, i, TANGENT_HALF_GAUSS, true, true, start, end);
                tangents[i] = tangent;

                // Compute the calibre of the curve
                if (compute_calibre) {
                    auto calibre = fast_branch_calibre(curve, i, segmentation, tangent);
                    calibres[i] = calibre;
                    if (adaptative_tangent) {
                        const auto& refinedTangent =
                            curve_tangent(curve, i, std::max(calibre, 1.5f), true, true, start, end);
                        tangents[i] = refinedTangent;
                        calibres[i] = fast_branch_calibre(curve, i, segmentation, refinedTangent);
                    } else
                        calibres[i] = calibre;
                }
            }
            // Compute the BSpline of the curve
            if (bspline_max_error > 0) {
                auto const& bspline_curve =
                    bspline_regression(curve, tangents, bspline_max_error, bspline_dK_threshold, start, end);
                bspline.insert(bspline.end(), bspline_curve.begin(), bspline_curve.end());
            }
        }
    }

    return std::make_tuple(branchesTangents, branchesCalibre, branchesBSpline);
}

BSpline bspline_regression(const CurveYX& curve, const CurveTangents& tangents, double bspline_max_error,
                           const float dK_threshold, std::size_t start, std::size_t end) {
    if (curve.size() < 2) return {};
    if (end == 0) end = curve.size();

    auto const& [bezier, maxError, u] = fit_bezier(curve, tangents, bspline_max_error, start, end - 1);
    if (dK_threshold > 0 && maxError > bspline_max_error) {
        const std::vector<std::size_t>& splitCandidate =
            curve_inflections_points(tangents, dK_threshold, TANGENT_SMOOTHING_KERNEL, start, end);
        return iterative_fit_bspline(curve, tangents, bezier, u, splitCandidate, bspline_max_error, start, end - 1);
    } else
        return {bezier};
}

/*
 *  FitCubic :
 *  	Fit a Bezier curve to a (sub)set of digitized points
 *
 * CurveYX d : Array of digitized points
 * int first, last : Indices of first and last (included!) pts in region
 * Vector t0, t1 : Unit tangent vectors at endpoints
 * double error : User-defined error squared
 */
BSpline iterative_fit_bspline(const CurveYX& d, const std::vector<Point>& tangents, const BezierCurve& bezier,
                              const std::vector<double>& u, const std::vector<std::size_t>& splitCandidates,
                              double error, std::size_t first, std::size_t last) {
    std::vector<std::size_t> validSplits;
    validSplits.reserve(splitCandidates.size());
    for (auto const& split : splitCandidates) {
        if (split > first && split < last) validSplits.push_back(split);
    }
    if (validSplits.size() == 0) return {bezier};
    const std::size_t nSplit = validSplits.size();

    // Evaluate the tangent cosine errors between the bezier and the curve
    std::vector<double> tangentErrors;
    tangentErrors.reserve(nSplit);
    {
        std::vector<double> splitU;
        splitU.reserve(nSplit);
        for (auto const& split : validSplits) splitU.push_back(u[split]);
        auto const& bezierTangent = evaluate_bezier_tangent(bezier, splitU);

        for (std::size_t i = 0; i < nSplit; i++) {
            auto const& T = tangents[validSplits[i]];
            auto const& Tbezier = bezierTangent[i].normalize();
            tangentErrors.push_back(1 - T.dot(Tbezier));
        }
    }

    // Find the split with maximum error
    std::size_t maxErrorSplit = 0;
    double maxTangentError = tangentErrors[0];
    for (std::size_t i = 1; i < nSplit; i++) {
        if (tangentErrors[i] > maxTangentError) {
            maxTangentError = tangentErrors[i];
            maxErrorSplit = i;
        }
    }

    // Split the curve at the maximum error and recursively fit the two parts
    const std::size_t split = validSplits[maxErrorSplit];
    BSpline bspline;
    for (auto const& [split_first, split_last] : std::array<SizePair, 2>{{{first, split}, {split, last}}}) {
        auto const& [bezier, maxError, u] = fit_bezier(d, tangents, error, split_first, split_last);
        if (maxError > error) {
            auto const& bspline1 =
                iterative_fit_bspline(d, tangents, bezier, u, splitCandidates, error, split_first, split_last);
            bspline.insert(bspline.end(), bspline1.begin(), bspline1.end());
        } else
            bspline.push_back(bezier);
    }
    return bspline;
}