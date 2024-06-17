#include <pybind11/stl.h>

#include "branch.h"
#include "fit_bezier.h"

/**
 * @brief Extract geometry information from a vessel graph.
 *
 * @param branch_labels The branch labels of the vessel graph.
 * @param node_yx The coordinates of the nodes of the vessel graph.
 * @param branch_list The list of branches of the vessel graph.
 * @param segmentation The segmentation of the vessel graph.
 *
 * @return A tuple containing:
 *  - a list of the yx coordinates of the branches pixels,
 *  - a list of the tangents of the branches (or the bezier node and control points),
 *  - the width of the branches
 */
std::vector<std::vector<torch::Tensor>> extract_branches_geometry(torch::Tensor branch_labels,
                                                                  const torch::Tensor &node_yx,
                                                                  const torch::Tensor &branch_list,
                                                                  const torch::Tensor &segmentation,
                                                                  std::map<std::string, double> options = {}) {
    // Point seg_shape = {(int)segmentation.size(0), (int)segmentation.size(1)};
    auto seg_acc = segmentation.accessor<bool, 2>();
    auto labels_acc = branch_labels.accessor<int, 2>();

    // --- Track branches ---
    auto branchesYX = track_branches(branch_labels, node_yx, branch_list);

    // --- Clean the branches terminations ---
    int clean_terminations = options.count("clean_terminations") ? options["clean_terminations"] : 0;
    if (clean_terminations > 0) {
        auto const &adj_list = edge_list_to_adjlist(tensor_to_vectorIntPair(branch_list), node_yx.size(0));
        std::vector<IntPair> branches_terminations(branchesYX.size(), {0, 0});

#pragma omp parallel for
        for (int nodeID = 0; nodeID < (int)adj_list.size(); nodeID++) {
            // For each node, find the first valid skeleton pixel of its incident branches...
            auto const &node_adjacency = adj_list[nodeID];
            auto const &node_terminations =
                clean_branch_skeleton_around_node(branchesYX, nodeID, node_adjacency, seg_acc, clean_terminations);

            // ... and store the termination indexes, in order to clean them later.
            int i = 0;
            for (auto const &edge : node_adjacency) {
                const int termination = (int)node_terminations[i];
                if (nodeID == edge.start) branches_terminations[edge.id][0] = termination;
                if (nodeID == edge.end) branches_terminations[edge.id][1] = termination + 1;
                i++;
            }
        }

#pragma omp parallel for
        for (std::size_t i = 0; i < branchesYX.size(); i++) {
            // For each branch, remove the invalid pixels
            auto &branchYX = branchesYX[i];
            auto [startI, endI] = branches_terminations[i];

            if (startI < endI - 3) {
                // Remove the invalid pixels from the branch labels, the branchYX and the tangents
                for (auto p = branchYX.begin() + endI; p != branchYX.end(); p++) labels_acc[p->y][p->x] = 0;
                branchYX.erase(branchYX.begin() + endI, branchYX.end());

                for (auto p = branchYX.begin(); p != branchYX.begin() + startI; p++) labels_acc[p->y][p->x] = 0;
                branchYX.erase(branchYX.begin(), branchYX.begin() + startI);
            } else {
                // If the valid section of the branch is 3 px or less, remove the branch.
                for (auto p : branchYX) labels_acc[p.y][p.x] = 0;
                branchYX.clear();
            }
        }
    }

    // --- Compute the tangents ---
    std::vector<std::vector<Point>> branchesTangents(branchesYX.size());
#pragma omp parallel for
    for (int i = 0; i < (int)branchesYX.size(); i++)
        branchesTangents[i] = fast_curve_tangent(branchesYX[i], TANGENT_HALF_GAUSS, {});

    // --- Compute the calibre ---
    std::vector<torch::Tensor> branchesCalibre;
    if (options.count("compute_calibre") ? options["compute_calibre"] : true) {
        branchesCalibre.resize(branchesYX.size());
        bool refine_tangent = options.count("adaptative_tangent") ? options["adaptative_tangent"] : true;

#pragma omp parallel for
        for (int i = 0; i < (int)branchesYX.size(); i++) {
            const auto &branch = branchesYX[i];
            const auto &calibre = fast_branch_calibre(branch, seg_acc, branchesTangents[i]);
            if (refine_tangent) {
                const auto &refinedTangent = curve_tangent(branch, calibre);
                branchesTangents[i] = refinedTangent;
                branchesCalibre[i] = vector_to_tensor(fast_branch_calibre(branch, seg_acc, refinedTangent));
            } else
                branchesCalibre[i] = vector_to_tensor(calibre);
        }
    }

    // --- BSpline Regression ---
    int bspline_max_error = options.count("bspline_max_error") ? options["bspline_max_error"] : 0;
    std::vector<BSpline> branchesBSpline;
    if (bspline_max_error > 0) {
        branchesBSpline.resize(branchesYX.size());
        double dtheta_kernel_std = options.count("dtheta_kernel_std") ? options["dtheta_kernel_std"] : 3;
        double inflection_max_angle = options.count("inflection_max_angle") ? options["inflection_max_angle"] : 1.5;
        const std::vector<float> &halfKernel = gaussianHalfKernel1D(dtheta_kernel_std);

#pragma omp parallel for
        for (int i = 0; i < (int)branchesYX.size(); i++) {
            const CurveYX &branch = branchesYX[i];
            if (branch.size() < 2) continue;

            const std::vector<Point> &tangents = branchesTangents[i];
            const std::vector<int> &inflections = curve_inflections_points(tangents, halfKernel, inflection_max_angle);

            BSpline bspline;
            bspline.reserve(inflections.size() + 1);

            int first = 0, last = 0;
            for (int j = 0; j < (int)inflections.size(); j++) {
                last = inflections[j];
                bspline.push_back(std::get<0>(fit_bezier(branch, tangents, bspline_max_error, first, last)));
                first = inflections[j];
            }
            last = (int)branch.size() - 1;
            bspline.push_back(std::get<0>(fit_bezier(branch, tangents, bspline_max_error, first, last)));

            branchesBSpline[i] = bspline;
        }
    }

    // --- Convert to tensor ---
    std::vector<std::vector<torch::Tensor>> out;

    std::vector<torch::Tensor> branchesYX_t;
    branchesYX_t.reserve(branchesYX.size());
    for (const CurveYX &branch : branchesYX) branchesYX_t.push_back(vector_to_tensor(branch));
    out.push_back(branchesYX_t);

    std::vector<torch::Tensor> branchesTangents_t;
    branchesTangents_t.reserve(branchesTangents.size());
    for (const std::vector<Point> &branch : branchesTangents) branchesTangents_t.push_back(vector_to_tensor(branch));
    out.push_back(branchesTangents_t);

    if (branchesCalibre.size() > 0) out.push_back(branchesCalibre);

    if (bspline_max_error > 0) {
        std::vector<torch::Tensor> branchesBSpline_t;
        branchesBSpline_t.reserve(branchesBSpline.size());
        for (const BSpline &spline : branchesBSpline) {
            torch::Tensor spline_t = torch::zeros({(long)spline.size(), 4, 2}, torch::kFloat);
            auto spline_acc = spline_t.accessor<float, 3>();
            for (int i = 0; i < (int)spline.size(); i++) {
                for (int j = 0; j < 4; j++) {
                    spline_acc[i][j][0] = spline[i][j].y;
                    spline_acc[i][j][1] = spline[i][j].x;
                }
            }
            branchesBSpline_t.push_back(spline_t);
        }
        out.push_back(branchesBSpline_t);
    }

    return out;
}

std::vector<torch::Tensor> track_branches_to_torch(const torch::Tensor &branch_labels, const torch::Tensor &node_yx,
                                                   const torch::Tensor &branch_list) {
    auto const &branches_pixels = track_branches(branch_labels, node_yx, branch_list);

    std::vector<torch::Tensor> branches_pixels_torch;
    branches_pixels_torch.reserve(branches_pixels.size());

    for (const CurveYX &branch : branches_pixels) {
        torch::Tensor branch_tensor = torch::zeros({(long)branch.size(), 2}, torch::kInt32);
        auto branch_acc = branch_tensor.accessor<int, 2>();
        for (int i = 0; i < (int)branch.size(); i++) {
            branch_acc[i][0] = branch[i].y;
            branch_acc[i][1] = branch[i].x;
        }
        branches_pixels_torch.push_back(branch_tensor);
    }
    return branches_pixels_torch;
}

torch::Tensor fast_curve_tangent_torch(const torch::Tensor &curveYX, float gaussianStd = 2,
                                       const torch::Tensor &evaluateAtID = {}) {
    const CurveYX &curveYX_vec = tensor_to_curveYX(curveYX);
    std::vector<float> gaussKernel = gaussianHalfKernel1D(gaussianStd);

    std::vector<int> evaluateAtID_vec;
    evaluateAtID_vec.reserve(evaluateAtID.size(0));
    auto evaluateAtID_acc = evaluateAtID.accessor<int, 1>();
    for (int i = 0; i < (int)evaluateAtID.size(0); i++) evaluateAtID_vec.push_back(evaluateAtID_acc[i]);

    auto const &tangents = fast_curve_tangent(curveYX_vec, gaussKernel, evaluateAtID_vec);
    return vector_to_tensor(tangents);
}

torch::Tensor fast_branch_boundaries_torch(const torch::Tensor &curveYX, const torch::Tensor &segmentation,
                                           const torch::Tensor &evaluateAtID = {}) {
    const CurveYX &curveYX_vec = tensor_to_curveYX(curveYX);

    std::vector<int> evaluateAtID_vec;
    evaluateAtID_vec.reserve(evaluateAtID.size(0));
    auto evaluateAtID_acc = evaluateAtID.accessor<int, 1>();
    for (int i = 0; i < (int)evaluateAtID.size(0); i++) evaluateAtID_vec.push_back(evaluateAtID_acc[i]);

    auto const &tangents = fast_curve_tangent(curveYX_vec, TANGENT_HALF_GAUSS, evaluateAtID_vec);

    auto const &boundaries = fast_branch_boundaries(curveYX_vec, segmentation, tangents, evaluateAtID_vec);
    return vector_to_tensor(boundaries);
}

torch::Tensor fast_branch_calibre_torch(const torch::Tensor &curveYX, const torch::Tensor &segmentation,
                                        const torch::Tensor &evaluateAtID = {}) {
    const CurveYX &curveYX_vec = tensor_to_curveYX(curveYX);

    std::vector<int> evaluateAtID_vec;
    evaluateAtID_vec.reserve(evaluateAtID.size(0));
    auto evaluateAtID_acc = evaluateAtID.accessor<int, 1>();
    for (int i = 0; i < (int)evaluateAtID.size(0); i++) evaluateAtID_vec.push_back(evaluateAtID_acc[i]);

    auto const &tangents = fast_curve_tangent(curveYX_vec, TANGENT_HALF_GAUSS, evaluateAtID_vec);

    auto &&boundaries = fast_branch_calibre(curveYX_vec, segmentation, tangents, evaluateAtID_vec);
    torch::Tensor widths_tensor = torch::from_blob(boundaries.data(), {(long)boundaries.size()}, torch::kFloat);
    return widths_tensor.clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("extract_branches_geometry", &extract_branches_geometry, "Extract geometry information from a vessel graph.");
    m.def("find_branch_endpoints", &find_branch_endpoints, "Find the first and last endpoint of each branch.");
    m.def("track_branches", &track_branches_to_torch, "Track the orientation of branches in a vessel graph.");
    m.def("fast_curve_tangent", &fast_curve_tangent_torch, "Evaluate the tangent of a curve.");
    m.def("fast_branch_boundaries", &fast_branch_boundaries_torch, "Evaluate the width of a branch.");
}
