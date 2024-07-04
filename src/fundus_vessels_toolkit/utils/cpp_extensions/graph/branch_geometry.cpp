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
    std::vector<CurveYX> branch_curves, const Tensor2DAccessor<bool> &segmentation,
    std::map<std::string, double> options, bool assume_contiguous) {
    // --- Parse Options ---
    bool compute_calibre = options.count("compute_calibre") ? options["compute_calibre"] : true;
    bool adaptative_tangent = options.count("adaptative_tangent") ? options["adaptative_tangent"] : true;
    int bspline_max_error = options.count("bspline_max_error") ? options["bspline_max_error"] : 0;
    double dtheta_kernel_std = options.count("dtheta_kernel_std") ? options["dtheta_kernel_std"] : 3;
    double inflection_max_angle = options.count("inflection_max_angle") ? options["inflection_max_angle"] : 1.5;

    // --- Prepare result vectors ---
    auto nCurves = branch_curves.size();
    std::vector<CurveTangents> branchesTangents(nCurves);
    std::vector<std::vector<float>> branchesCalibre(nCurves);
    std::vector<BSpline> branchesBSpline(nCurves);

// --- Extract geometry of each branches ---
#pragma omp parallel for
    for (std::size_t curveI = 0; curveI < nCurves; curveI++) {
        auto const &curve = branch_curves[curveI];

        // Get references to the results vectors...
        CurveTangents &tangents = branchesTangents[curveI];
        std::vector<float> &calibres = branchesCalibre[curveI];
        BSpline &bspline = branchesBSpline[curveI];
        // ... and initialize them
        tangents.resize(curve.size());
        if (compute_calibre) calibres.resize(curve.size());

        // Split the branch curve into contiguous curves
        auto const &contiguousCurvesStartEnd =
            assume_contiguous ? std::list<SizePair>({{0, curve.size()}}) : splitInContiguousCurves(curve);
        for (auto const &[start, end] : contiguousCurvesStartEnd) {
            for (std::size_t i = start; i < end; i++) {
                // Compute the tangent of the curve
                auto const &tangent = fast_curve_tangent(curve, i, TANGENT_HALF_GAUSS, true, true, start, end);
                tangents[i] = tangent;

                // Compute the calibre of the curve
                if (compute_calibre) {
                    auto calibre = fast_branch_calibre(curve, i, segmentation, tangent);
                    calibres[i] = calibre;
                    if (adaptative_tangent) {
                        const auto &refinedTangent =
                            curve_tangent(curve, i, std::max(calibre, 1.5f), true, true, start, end);
                        tangents[i] = refinedTangent;
                        calibres[i] = fast_branch_calibre(curve, i, segmentation, refinedTangent);
                    } else
                        calibres[i] = calibre;
                }
            }
            // Compute the BSpline of the curve
            if (bspline_max_error > 0) {
                auto const &bspline_curve =
                    bspline_regression(curve, tangents, bspline_max_error, gaussianHalfKernel1D(dtheta_kernel_std),
                                       inflection_max_angle, start, end);
                bspline.insert(bspline.end(), bspline_curve.begin(), bspline_curve.end());
            }
        }
    }

    return std::make_tuple(branchesTangents, branchesCalibre, branchesBSpline);
}