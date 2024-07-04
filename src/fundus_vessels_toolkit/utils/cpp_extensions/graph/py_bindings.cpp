#include <pybind11/stl.h>

#include "branch.h"
#include "graph.h"
#include "skeleton.h"

/*********************************************************************************************
 *             === SKELETON PARSING ===
 *********************************************************************************************/

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> parse_skeleton(torch::Tensor labelMap) {
    // --- Parse the skeleton ---
    auto [edge_list, branches_curves, node_yx] = parse_skeleton_to_graph(labelMap);
    return {edge_list_to_tensor(edge_list), vector_to_tensor(node_yx), vectors_to_tensors(branches_curves)};
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> parse_skeleton_with_cleanup(
    torch::Tensor labelMap, torch::Tensor segmentation, std::map<std::string, double> options = {}) {
    // --- Parse the skeleton ---
    auto [edge_list, branches_curves, node_yx] = parse_skeleton_to_graph(labelMap);
    auto const &adj_list = edge_list_to_adjlist(edge_list, node_yx.size());
    auto labels_acc = labelMap.accessor<int, 2>();
    auto seg_acc = segmentation.accessor<bool, 2>();

    // --- Clean the branches terminations ---
    double clean_terminations = options.count("clean_terminations") ? options["clean_terminations"] : 0;
    if (clean_terminations > 0)
        clean_branches_skeleton(branches_curves, labels_acc, seg_acc, adj_list, clean_terminations);

    // --- Remove spurs ---
    double max_spurs_length = options.count("max_spurs_length") ? options["max_spurs_length"] : 0;
    double remove_spurs_ratio = options.count("max_spurs_calibre_ratio") ? options["max_spurs_calibre_ratio"] : 0;
    if (max_spurs_length > 0 || remove_spurs_ratio > 0) {
        auto const &spurs =
            find_small_spurs(branches_curves, labels_acc, adj_list, seg_acc, max_spurs_length, remove_spurs_ratio);
        remove_branches(spurs, branches_curves, labels_acc, edge_list);
        remove_singleton_nodes(edge_list, node_yx, labels_acc);
        // !! The adj_list is not updated, but it is not used anymore !!
    }

    return {edge_list_to_tensor(edge_list), vector_to_tensor(node_yx), vectors_to_tensors(branches_curves)};
}

/*********************************************************************************************
 *             === BRANCH GEOMETRY ===
 *********************************************************************************************/

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
std::vector<std::vector<torch::Tensor>> extract_branches_geometry_from_curves(
    std::vector<torch::Tensor> branch_curves, const torch::Tensor &segmentation,
    std::map<std::string, double> options = {}) {
    auto const &seg_acc = segmentation.accessor<bool, 2>();
    auto const &curves = tensors_to_curves(branch_curves);
    auto const &[tangents, calibres, bsplines] = extract_branches_geometry(curves, seg_acc, options);

    // --- Convert to tensor ---
    std::vector<std::vector<torch::Tensor>> out;
    out.push_back(vectors_to_tensors(tangents));
    if (calibres.size() > 0) out.push_back(vectors_to_tensors(calibres));
    if (bsplines.size() > 0) out.push_back(bsplines_to_tensor(bsplines));

    return out;
}

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
std::vector<std::vector<torch::Tensor>> extract_branches_geometry_from_skeleton(
    torch::Tensor branch_labels, const torch::Tensor &node_yx, const torch::Tensor &branch_list,
    const torch::Tensor &segmentation, std::map<std::string, double> options = {}) {
    auto const &seg_acc = segmentation.accessor<bool, 2>();
    auto labels_acc = branch_labels.accessor<int, 2>();

    // --- Track branches ---
    auto curves = track_branches(branch_labels, node_yx, branch_list);

    // --- Clean the branches terminations ---
    int clean_terminations = options.count("clean_terminations") ? options["clean_terminations"] : 0;
    if (clean_terminations > 0) {
        auto const &adj_list = edge_list_to_adjlist(tensor_to_vectorIntPair(branch_list), node_yx.size(0));
        clean_branches_skeleton(curves, labels_acc, seg_acc, adj_list, clean_terminations);
    }

    // --- Extract branches geometry ---
    auto const &[tangents, calibres, bsplines] = extract_branches_geometry(curves, seg_acc, options, true);

    // --- Convert to tensor ---
    std::vector<std::vector<torch::Tensor>> out;
    out.push_back(vectors_to_tensors(curves));
    out.push_back(vectors_to_tensors(tangents));
    if (calibres.size() > 0) out.push_back(vectors_to_tensors(calibres));
    if (bsplines.size() > 0) out.push_back(bsplines_to_tensor(bsplines));

    return out;
}

std::vector<torch::Tensor> track_branches_to_torch(const torch::Tensor &branch_labels, const torch::Tensor &node_yx,
                                                   const torch::Tensor &branch_list) {
    auto const &branches_pixels = track_branches(branch_labels, node_yx, branch_list);
    return vectors_to_tensors(branches_pixels);
}

torch::Tensor fast_curve_tangent_torch(const torch::Tensor &curveYX, float gaussianStd = 2,
                                       const torch::Tensor &evaluateAtID = {}) {
    const CurveYX &curveYX_vec = tensor_to_curve(curveYX);
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
    const CurveYX &curveYX_vec = tensor_to_curve(curveYX);
    auto const &seg_acc = segmentation.accessor<bool, 2>();

    std::vector<int> evaluateAtID_vec;
    evaluateAtID_vec.reserve(evaluateAtID.size(0));
    auto evaluateAtID_acc = evaluateAtID.accessor<int, 1>();
    for (int i = 0; i < (int)evaluateAtID.size(0); i++) evaluateAtID_vec.push_back(evaluateAtID_acc[i]);

    auto const &tangents = fast_curve_tangent(curveYX_vec, TANGENT_HALF_GAUSS, evaluateAtID_vec);

    auto const &boundaries = fast_branch_boundaries(curveYX_vec, seg_acc, tangents, evaluateAtID_vec);
    return vector_to_tensor(boundaries);
}

torch::Tensor fast_branch_calibre_torch(const torch::Tensor &curveYX, const torch::Tensor &segmentation,
                                        const torch::Tensor &evaluateAtID = {}) {
    const CurveYX &curveYX_vec = tensor_to_curve(curveYX);
    auto const &seg_acc = segmentation.accessor<bool, 2>();

    std::vector<int> evaluateAtID_vec;
    evaluateAtID_vec.reserve(evaluateAtID.size(0));
    auto evaluateAtID_acc = evaluateAtID.accessor<int, 1>();
    for (int i = 0; i < (int)evaluateAtID.size(0); i++) evaluateAtID_vec.push_back(evaluateAtID_acc[i]);

    auto const &tangents = fast_curve_tangent(curveYX_vec, TANGENT_HALF_GAUSS, evaluateAtID_vec);

    auto &&boundaries = fast_branch_calibre(curveYX_vec, seg_acc, tangents, evaluateAtID_vec);
    torch::Tensor widths_tensor = torch::from_blob(boundaries.data(), {(long)boundaries.size()}, torch::kFloat);
    return widths_tensor.clone();
}

/**************************************************************************************
 *             === PYBIND11 BINDINGS ===
 **************************************************************************************/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parse_skeleton", &parse_skeleton, "Parse a skeleton to a graph.");
    m.def("parse_skeleton_with_cleanup", &parse_skeleton_with_cleanup, "Parse a skeleton to a graph with cleanup.");
    m.def("extract_branches_geometry", &extract_branches_geometry_from_curves,
          "Extract geometry information from a vessel graph.");
    m.def("extract_branches_geometry_from_skeleton", &extract_branches_geometry_from_skeleton,
          "Extract geometry information from a vessel graph.");
    m.def("track_branches", &track_branches_to_torch, "Track the orientation of branches in a vessel graph.");
    m.def("fast_curve_tangent", &fast_curve_tangent_torch, "Evaluate the tangent of a curve.");
    m.def("fast_branch_boundaries", &fast_branch_boundaries_torch, "Evaluate the width of a branch.");

    // === Skeleton.h ===
    m.def("detect_skeleton_nodes", &detect_skeleton_nodes, "Detect junctions and endpoints in a skeleton.");

    // === Branch.h ===
    m.def("find_branch_endpoints", &find_branch_endpoints, "Find the first and last endpoint of each branch.");
}
