#include <pybind11/stl.h>

#include "branch.h"
#include "disjoint_set.h"
#include "edit_distance.h"
#include "graph.h"
#include "ray_iterators.h"
#include "skeleton.h"

/*********************************************************************************************
 *             === SKELETON PARSING ===
 *********************************************************************************************/

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>> parse_skeleton(torch::Tensor labelMap) {
    // --- Parse the skeleton ---
    auto [edge_list, branches_curves, node_yx] = parse_skeleton_to_graph(labelMap);
    return {edge_list_to_tensor(edge_list), vector_to_tensor(node_yx), vectors_to_tensors(branches_curves)};
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<torch::Tensor>, torch::Tensor> parse_skeleton_with_cleanup(
    torch::Tensor labelMap, torch::Tensor segmentation, std::map<std::string, double> options = {}) {
    // --- Parse the skeleton ---
    auto [edge_list, branches_curves, node_yx] = parse_skeleton_to_graph(labelMap);
    auto labels_acc = labelMap.accessor<int, 2>();
    auto seg_acc = segmentation.accessor<bool, 2>();

    double min_spurs_length = get_if_exists(options, "min_spurs_length", 0.);
    if (min_spurs_length > 0) remove_small_spurs(min_spurs_length + 1, edge_list, branches_curves, node_yx, labels_acc);

    // --- Clean the branches tips ---
    double clean_branches_tips = get_if_exists(options, "clean_branches_tips", 0.);
    double clean_terminal_branches_tips = get_if_exists(options, "clean_terminal_branches_tips", 0.);
    bool adaptativeTangent = get_if_exists(options, "adaptative_tangent", 1.) > 0;
    auto tangents_calibres_tensor = torch::empty({0}, torch::kFloat);
    if (clean_branches_tips > 0) {
        auto const &adj_list = edge_list_to_adjlist(edge_list, node_yx.size());
        auto const tangents_calibres =
            clean_branches_skeleton(branches_curves, labels_acc, seg_acc, adj_list, clean_branches_tips,
                                    clean_terminal_branches_tips, adaptativeTangent);
        const long n_branches = (long)tangents_calibres.size();
        // Convert tip geometry to tensor
        tangents_calibres_tensor = torch::empty({n_branches, 2, 7}, torch::kFloat);
        auto accessor = tangents_calibres_tensor.accessor<float, 3>();
        for (int i = 0; i < n_branches; i++) {
            auto const &tc = tangents_calibres[i];
            for (int j = 0; j < 2; j++) {
                accessor[i][j][0] = std::get<0>(tc[j]).y;  // tangent.y
                accessor[i][j][1] = std::get<0>(tc[j]).x;  // tangent.x
                accessor[i][j][2] = std::get<1>(tc[j]);    // calibre
                accessor[i][j][3] = std::get<2>(tc[j]).y;  // leftBoundary.y
                accessor[i][j][4] = std::get<2>(tc[j]).x;  // leftBoundary.x
                accessor[i][j][5] = std::get<3>(tc[j]).y;  // rightBoundary.y
                accessor[i][j][6] = std::get<3>(tc[j]).x;  // rightBoundary.x
            }
        }
    }

    // --- Remove spurs ---
    double spurs_calibre_factor = get_if_exists(options, "spurs_calibre_factor", 0.);
    if (min_spurs_length > 0 || spurs_calibre_factor > 0) {
        double max_spurs_length = get_if_exists(options, "max_spurs_length", std::numeric_limits<double>::max());
        auto const &spurs =
            find_spurs(branches_curves, edge_list, seg_acc, min_spurs_length, spurs_calibre_factor, max_spurs_length);
        if (spurs.size() > 0) {
            remove_branches(spurs, branches_curves, labels_acc, edge_list);
            remove_singleton_nodes(edge_list, node_yx, labels_acc);
            if (tangents_calibres_tensor.size(0) > 0) {
                std::vector<int> spurs_ids;
                spurs_ids.reserve(spurs.size());
                for (auto const &spur : spurs) spurs_ids.push_back(spur.id);
                tangents_calibres_tensor = remove_rows(tangents_calibres_tensor, spurs_ids);
            }
            // !! The adj_list is not updated, but it is not used anymore !!
        }
    }

    return {edge_list_to_tensor(edge_list), vector_to_tensor(node_yx), vectors_to_tensors(branches_curves),
            tangents_calibres_tensor};
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
    auto const &[tangents, calibres, boundaries, curvatures, curv_roots, bsplines] =
        extract_branches_geometry(curves, seg_acc, options);

    // --- Convert to tensor ---
    std::vector<std::vector<torch::Tensor>> out;
    out.push_back(vectors_to_tensors(tangents));
    if (calibres.size() > 0) out.push_back(vectors_to_tensors(calibres));
    if (boundaries.size() > 0) out.push_back(vectors_to_tensors(boundaries));
    if (curvatures.size() > 0) out.push_back(vectors_to_tensors(curvatures));
    if (curv_roots.size() > 0) out.push_back(vectors_to_tensors(curv_roots));
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

    // --- Clean the branches tips ---
    int clean_branches_tips = get_if_exists(options, "clean_branches_tips", 0.);
    bool adaptativeTangent = get_if_exists(options, "adaptative_tangent", 1.) > 0;
    if (clean_branches_tips > 0) {
        auto const &adj_list = edge_list_to_adjlist(tensor_to_vectorIntPair(branch_list), node_yx.size(0));
        clean_branches_skeleton(curves, labels_acc, seg_acc, adj_list, clean_branches_tips, adaptativeTangent);
    }

    // --- Extract branches geometry ---
    auto const &[tangents, calibres, boundaries, curvatures, curv_roots, bsplines] =
        extract_branches_geometry(curves, seg_acc, options, true);

    // --- Convert to tensor ---
    std::vector<std::vector<torch::Tensor>> out;
    out.push_back(vectors_to_tensors(curves));
    out.push_back(vectors_to_tensors(tangents));
    if (calibres.size() > 0) out.push_back(vectors_to_tensors(calibres));
    if (boundaries.size() > 0) out.push_back(vectors_to_tensors(boundaries));
    if (curvatures.size() > 0) out.push_back(vectors_to_tensors(curvatures));
    if (curv_roots.size() > 0) out.push_back(vectors_to_tensors(curv_roots));
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

torch::Tensor compute_curvature(const torch::Tensor &curveYX, const torch::Tensor &tangents) {
    const CurveYX &curve = tensor_to_curve(curveYX);
    const PointList &tangents_vec = tensor_to_pointList(tangents);
    auto const &contiguousCurvesStartEnd = split_contiguous_curves(curve);

    torch::Tensor curvatures_tensor = torch::empty({(long)curve.size()}, torch::kFloat);

    for (auto const &[start, end] : contiguousCurvesStartEnd) {
        auto const &curvatures = tangents_to_curvature(tangents_vec, 5, start, end);

        for (std::size_t i = start; i < end; i++) curvatures_tensor[i] = curvatures[i - start];
    }

    return curvatures_tensor;
}

torch::Tensor find_inflections_points(const torch::Tensor &curvatures, float K_threshold = 0.05) {
    auto const &curvatures_vec = tensor_to_scalars(curvatures);
    auto const &inflections = curve_inflections_points(curvatures_vec, K_threshold);
    return vector_to_tensor(inflections);
}

std::tuple<torch::Tensor, double, torch::Tensor> fit_bezier_torch(const torch::Tensor &curveYX,
                                                                  const torch::Tensor &tangents,
                                                                  double bspline_max_error, std::size_t start,
                                                                  std::size_t end) {
    const CurveYX &curve = tensor_to_curve(curveYX);
    const PointList &tangents_vec = tensor_to_pointList(tangents);
    auto const &curvatures = tangents_to_curvature(tangents_vec, true, 5, start, end);
    auto const &[bezier, maxError, sqrError, u] = fit_bezier(curve, tangents_vec, bspline_max_error, start, end);
    return {bspline_to_tensor(bezier), maxError, vector_to_tensor(u)};
}

torch::Tensor drawLine(std::array<int, 2> tip, std::array<float, 2> direction, int length) {
    auto scene = torch::zeros({512, 512}, torch::kInt);
    auto sceneAcc = scene.accessor<int, 2>();
    auto iter = RayIterator(IntPoint(tip[0], tip[1]), Point(direction[0], direction[1]));
    int i = 0;
    while (++i < length) {
        auto const &p = ++iter;
        if (!p.is_inside(512, 512)) break;
        sceneAcc[p.y][p.x] += 1;
    }
    return scene;
}

torch::Tensor drawCone(std::array<int, 2> tip, std::array<float, 2> direction, float angle, int length) {
    auto scene = torch::zeros({512, 512}, torch::kInt);
    auto sceneAcc = scene.accessor<int, 2>();
    auto coneIter = ConeIterator(IntPoint(tip[0], tip[1]), Point(direction[0], direction[1]), angle);
    while (true) {
        auto const &p = ++coneIter;
        if (!p.is_inside(512, 512) || coneIter.height() >= length) break;
        sceneAcc[p.y][p.x] += 1;
    }
    return scene;
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
    m.def("fast_branch_boundaries", &fast_branch_boundaries_torch, "Evaluate the boundaries of a branch.");
    m.def("fast_branch_calibre", &fast_branch_calibre_torch, "Evaluate the width of a branch.");
    m.def("curve_curvature", &compute_curvature, "Evaluate the curvature of a curve.");
    m.def("find_inflections_points", &find_inflections_points, "Find the inflection points of a curve.");
    m.def("fit_bezier", &fit_bezier_torch, "Fit a cubic bezier curve to a set of points.");
    m.def("drawCone", &drawCone, "Draw a cone in a 2D image.");
    m.def("drawLine", &drawLine, "Draw a line in a 2D image.");

    // === Skeleton.h ===
    m.def("detect_skeleton_nodes", &detect_skeleton_nodes, "Detect junctions and endpoints in a skeleton.");
    m.def("detect_skeleton_nodes_debug", &detect_skeleton_nodes_debug, "Detect junctions and endpoints in a skeleton.");

    // === Branch.h ===
    m.def("find_branch_endpoints", &find_branch_endpoints, "Find the first and last endpoint of each branch.");
    m.def("find_closest_branches", &find_closest_branches, "Find the closest branches to a set of points.");
    m.def("draw_branches_labels", &draw_branches_labels, "Draw the branches on a tensor.");

    // === EditDistance.h ===
    m.def("shortest_secondary_path", &shortest_secondary_path, "Compute the shortest path between two sets of nodes.");
    m.def("nodes_similarity", &nodes_similarity, "Compute the similarity between two sets of nodes.");

    // === graph.h ===
    m.def("maximum_weighted_independent_set", &maximum_weighted_independent_set,
          "Compute the maximum weighted independent set.");

    // === disjoint_set.h ===
    m.def("has_cycle", &has_cycle, "Find cycles in a list of parent.");
    m.def("find_cycles", &find_cycles, "Find all cycles in a list of parent.");
}
