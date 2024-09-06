#include "branch.h"
#include "graph.h"

/**
 * @brief Clean the branches terminations on a whole skeleton.
 *
 * This method removes the invalid pixels at the terminations of the branches from the branchCurves and the
 * branchesLabelMap.
 *
 * @param branchCurves A list of branches to clean defined by a vector of points.
 * @param branchesLabelMap An accessor to a 2D tensor of shape (H, W) containing the branch labels to clean.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the vessels binary segmentation.
 * @param adjacency The adjacency list of the vessels graph.
 * @param maxRemovedLength The maximum number of pixels to remove from the branch.
 *
 * @return A list of pairs of vectors and floats representing the tangent and the calibre of the branch at the
 * terminations.
 */
std::vector<std::array<std::tuple<Vector, float, IntPoint, IntPoint>, 2>> clean_branches_skeleton(
    std::vector<CurveYX> &branchCurves, Tensor2DAcc<int> &branchesLabelMap, const Tensor2DAcc<bool> &segmentation,
    const GraphAdjList &adjacency, const int maxRemovedLength) {
    std::vector<IntPair> branches_terminations(branchCurves.size(), {0, 0});
    std::vector<std::array<std::tuple<Vector, float, IntPoint, IntPoint>, 2>> out(branchCurves.size());

#pragma omp parallel for
    for (int nodeID = 0; nodeID < (int)adjacency.size(); nodeID++) {
        // For each node, find the first valid skeleton pixel of its incident branches...
        auto const &node_adjacency = adjacency[nodeID];
        auto const &node_terminations =
            clean_branch_skeleton_around_node(branchCurves, nodeID, node_adjacency, segmentation, maxRemovedLength);

        // ... and store the termination indexes, in order to clean them later.
        int i = 0;
        for (auto const &edge : node_adjacency) {
            auto const &[termination, tangent, calibre, boundL, boundR] = node_terminations[i];
            if (nodeID == edge.start) branches_terminations[edge.id][0] = termination;
            if (nodeID == edge.end) branches_terminations[edge.id][1] = termination + 1;
            out[edge.id][nodeID == edge.start ? 0 : 1] = {tangent, calibre, boundL, boundR};
            i++;
        }
    }

#pragma omp parallel for
    for (std::size_t i = 0; i < branchCurves.size(); i++) {
        // For each branch, remove the invalid pixels
        auto &branchYX = branchCurves[i];
        auto [startI, endI] = branches_terminations[i];

        if (startI < endI - 1) {
            // Remove the invalid pixels from the branch labels, the branchYX and the tangents
            for (auto p = branchYX.begin() + endI; p != branchYX.end(); p++) branchesLabelMap[p->y][p->x] = 0;
            branchYX.erase(branchYX.begin() + endI, branchYX.end());

            for (auto p = branchYX.begin(); p != branchYX.begin() + startI; p++) branchesLabelMap[p->y][p->x] = 0;
            branchYX.erase(branchYX.begin(), branchYX.begin() + startI);
        } else {
            // If the valid section of the branch is 1 px or less, remove the branch.
            for (auto p : branchYX) branchesLabelMap[p.y][p.x] = 0;
            branchYX.clear();
        }
    }

    return out;
}

/**
 * @brief Clean the skeleton of a branch around a node.
 *
 * This method removes the pixels of the branches around a node that are not
 * part of the branch. The branch start is defined as the first pixel of the
 * skeleton which satistifies the following conditions:
 *  - The pixel is roughly at the center of the branch.
 *  - The branch width is roughly the same for this pixel and the next one in the skeleton.
 *  - The corresponding boundary pixels are closer to the current branch than to any other.
 *
 *
 * @param branchCurves A list of branches defined by a vector of points.
 * @param nodeID The id of the node around which the branches should be cleaned.
 * @param node_adjacency A set of edges representing the adjacency of the node.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the binary segmentation.
 * @param maxRemovedLength The maximum number of pixels to remove from the branch.
 *
 * @return A list of tuples containing the index of the first valid pixel, the tangent and the calibre of
 * the branch at this pixel.
 *
 */
std::vector<std::tuple<int, Vector, float, IntPoint, IntPoint>> clean_branch_skeleton_around_node(
    const std::vector<CurveYX> &branchCurves, const int nodeID, const std::set<Edge> &node_adjacency,
    const Tensor2DAcc<bool> &segmentation, const int maxRemovedLength) {
    // === Preparation ==
    // Read branches id and side
    std::vector<std::tuple<int, bool>> branchesInfos;
    branchesInfos.reserve(node_adjacency.size());
    for (const auto &edge : node_adjacency) {
        if (edge.start == nodeID) branchesInfos.push_back({edge.id, true});
        if (edge.end == nodeID) branchesInfos.push_back({edge.id, false});
    }
    // Prepare a lambda function to get the start and end (if the maximum length was to be removed) of a branch
    auto get_branch_start_end = [&](int branchID, bool forward) {
        const CurveYX &curveYX = branchCurves[branchID];
        const int c0 = node_adjacency.size() > 1 ? 1 : 0;
        const int start = forward ? c0 : curveYX.size() - 1 - c0;
        const int end = forward ? std::min(maxRemovedLength, (int)curveYX.size() - 2)
                                : std::max((int)curveYX.size() - 1 - maxRemovedLength, 1);
        return std::make_tuple(start, end);
    };

    // === For each branch: search first valid pixel (Lambda Function) ===
    auto find_first_valid_branch_pixel = [&](int branchI) {
        auto [branchID, forward] = branchesInfos[branchI];
        const CurveYX &curveYX = branchCurves[branchID];

        const int inc = forward ? 1 : -1;
        auto const [start, end] = get_branch_start_end(branchID, forward);
        // === If the branch is too short, keep it ===
        if (curveYX.size() <= 3)
            return std::make_tuple(start, Point(curveYX[end] - curveYX[start]).normalize(), 0.0f, curveYX[start],
                                   curveYX[start]);

        // === Check if pixel is valid (Lambda Function) ===
        auto is_branch_pixel_valid = [&](const IntPoint &p, const std::array<IntPoint, 2> &boundaries,
                                         const std::array<IntPoint, 2> &nextBoundaries) {
            auto const &boundL = boundaries[0], boundR = boundaries[1];
            if (!boundL.is_valid() || !boundR.is_valid()) return false;

            // Check if the skeleton is at the center of the boundaries
            const float dL = distance(p, boundL), dR = distance(p, boundR);
            if (abs(dL - dR) > 1.42) return false;

            // Check if the branch width is constant for this pixel and the next
            const float d0 = distance(boundL, boundR), d1 = distance(nextBoundaries[0], nextBoundaries[1]);
            if (abs(d0 - d1) > 1.42) return false;

            // Check if the skeleton closest to the boundaries belong to the current branch
            auto isClosestToCurrentBranch = [&](const IntPoint &bound) {
                float distToCurrentBranch = distance(p, bound);
                for (auto [id, forward] : branchesInfos) {
                    if (id == branchID) continue;
                    const CurveYX &otherCurve = branchCurves[id];
                    auto const [otherStart, otherEnd] = get_branch_start_end(id, forward);
                    const float dist = std::get<1>(findClosestPixel(otherCurve, p, otherStart, otherEnd, true));
                    if (dist < distToCurrentBranch) return false;
                }
                return true;
            };
            if (!isClosestToCurrentBranch(boundL) || !isClosestToCurrentBranch(boundR)) return false;

            return true;
        };

        Point tangent = fast_curve_tangent(curveYX, start, TANGENT_HALF_GAUSS, forward, !forward);
        auto boundaries = fast_branch_boundaries(curveYX, start, segmentation, tangent);
        Point nextTangent;
        std::array<IntPoint, 2> nextBoundaries;

        for (int i = start; i != end; i += inc) {
            nextTangent = fast_curve_tangent(curveYX, i + inc, TANGENT_HALF_GAUSS, forward, !forward);
            nextBoundaries = fast_branch_boundaries(curveYX, i + inc, segmentation, nextTangent);
            if (is_branch_pixel_valid(curveYX[i], boundaries, nextBoundaries)) {
                float calibre = fast_branch_calibre(boundaries[0], boundaries[1], tangent);
                return std::make_tuple(i, tangent, calibre, boundaries[0], boundaries[1]);
            }

            tangent = nextTangent;
            boundaries = nextBoundaries;
        }

        return std::make_tuple(end, tangent, fast_branch_calibre(boundaries[0], boundaries[1], tangent), boundaries[0],
                               boundaries[1]);
    };

    std::vector<std::tuple<int, Vector, float, IntPoint, IntPoint>> out(branchesInfos.size());
    for (std::size_t i = 0; i < branchesInfos.size(); i++) out[i] = find_first_valid_branch_pixel(i);
    return out;
}

std::vector<Edge> find_small_spurs(const std::vector<CurveYX> &branchCurves, const Tensor2DAcc<int> &branchesLabelMap,
                                   const GraphAdjList &adjacency, const Tensor2DAcc<bool> &segmentation,
                                   float max_spurs_length, const float max_calibre_ratio) {
    max_spurs_length = std::max(max_spurs_length, 1.0f);

    // Search for the terminal edges
    auto terminal_edge = terminal_edges(adjacency);

    // Instantiate the list of branches to remove
    std::vector<Edge> branchToRemove;
    branchToRemove.reserve(terminal_edge.size());

#pragma omp parallel for reduction(merge : branchToRemove)
    for (auto const &edge : terminal_edge) {
        auto const curveSize = branchCurves[edge.id].size();
        // Filter the terminal edges: only select the small spurs for removal
        if (curveSize <= max_spurs_length) {
            branchToRemove.push_back(edge);
            continue;
        }

        // If the calibres are provided, compute the maximum spur length based on the calibres
        if (max_calibre_ratio > 0) {
            float max_spur_length_calibre =
                largest_near_calibre(edge, adjacency, branchCurves, segmentation) * max_calibre_ratio;
            if (curveSize <= max_spur_length_calibre) branchToRemove.push_back(edge);
        }
    }

    return branchToRemove;
}

float largest_near_calibre(const Edge &edge, const GraphAdjList &adjacency, const std::vector<CurveYX> &branchesCurves,
                           const Tensor2DAcc<bool> &segmentation) {
    float maxCalibre = 0;
    for (auto const &nearEdge : adjacency[edge.start]) {
        auto const &curve = branchesCurves[nearEdge.id];
        if (curve.empty()) continue;

        int evaluateAtI = edge.start == nearEdge.start ? 0 : (int)curve.size() - 1;
        auto const &tangent = fast_curve_tangent(curve, evaluateAtI);
        const float calibre = fast_branch_calibre(curve, evaluateAtI, segmentation, tangent);
        maxCalibre = std::max(maxCalibre, calibre);
    }
    return maxCalibre;
}