#include "branch.h"
#include "graph.h"

/**
 * @brief Clean the branches tips on a whole skeleton.
 *
 * This method removes the invalid pixels at the tips of the branches from the branchCurves and the
 * branchesLabelMap.
 *
 * @param branchCurves A list of branches to clean defined by a vector of points.
 * @param branchesLabelMap An accessor to a 2D tensor of shape (H, W) containing the branch labels to clean.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the vessels binary segmentation.
 * @param adjacency The adjacency list of the vessels graph.
 * @param maxRemovedLength The maximum number of pixels to remove from the branch.
 * @param maxRemovedLengthEnd The maximum number of pixels to remove from the branch tips connected to endpoints.
 * @param adaptativeTangent A boolean indicating if the tangent smoothing should be scaled by the branch calibre.
 *
 * @return A list of pairs of vectors and floats representing the tangent and the calibre of the branches tips.
 */
std::vector<std::array<std::tuple<Vector, float, IntPoint, IntPoint>, 2>> clean_branches_skeleton(
    std::vector<CurveYX> &branchCurves, Tensor2DAcc<int> &branchesLabelMap, const Tensor2DAcc<bool> &segmentation,
    const GraphAdjList &adjacency, int maxRemovedLength, int maxRemovedLengthEnd, bool adaptativeTangent) {
    std::vector<IntPair> branches_tips(branchCurves.size(), {0, 0});
    std::vector<std::array<std::tuple<Vector, float, IntPoint, IntPoint>, 2>> out(branchCurves.size());

    if (maxRemovedLengthEnd < 0) maxRemovedLengthEnd = maxRemovedLength;

#pragma omp parallel for
    for (int nodeID = 0; nodeID < (int)adjacency.size(); nodeID++) {
        // For each node, find the first valid skeleton pixel of...
        auto const &node_adjacency = adjacency[nodeID];

        if (node_adjacency.size() == 1) {
            // ... its **single** branch ...
            const auto &edge = *node_adjacency.begin();
            const bool startSide = nodeID == edge.start;
            auto const &[tipIdx, tangent, calibre, boundL, boundR] = clean_branch_skeleton_tip(
                branchCurves, edge.id, startSide, segmentation, maxRemovedLengthEnd, adaptativeTangent);

            // ... and store the tip indexes, in order to clean them later.
            branches_tips[edge.id][startSide ? 0 : 1] = tipIdx;
            out[edge.id][startSide ? 0 : 1] = {startSide ? tangent : -tangent, calibre, boundL, boundR};
        } else {
            //  ... **all** its incident branches...
            auto const &tips_around_node = clean_branch_skeleton_around_node(
                branchCurves, nodeID, node_adjacency, segmentation, maxRemovedLength, adaptativeTangent);

            // ... and store the tips indexes, in order to clean them later.
            int i = 0;
            for (auto const &edge : node_adjacency) {
                auto const &[tipIdx, tangent, calibre, boundL, boundR] = tips_around_node[i];
                if (nodeID == edge.start) {
                    branches_tips[edge.id][0] = tipIdx;
                    out[edge.id][0] = {tangent, calibre, boundL, boundR};
                }
                if (nodeID == edge.end) {
                    branches_tips[edge.id][1] = tipIdx;  // TODO:  check why there was a +1 here...
                    out[edge.id][1] = {-tangent, calibre, boundL, boundR};
                }
                i++;
            }
        }
    }

#pragma omp parallel for
    for (std::size_t i = 0; i < branchCurves.size(); i++) {
        // For each branch, remove the invalid pixels
        auto &branchYX = branchCurves[i];
        auto [firstI, lastI] = branches_tips[i];

        if (firstI <= lastI - 1) {
            // Remove the invalid pixels from the branch labels, the branchYX and the tangents
            for (auto p = branchYX.begin() + lastI; p != branchYX.end(); p++) branchesLabelMap[p->y][p->x] = 0;
            branchYX.erase(branchYX.begin() + lastI + 1, branchYX.end());

            for (auto p = branchYX.begin(); p != branchYX.begin() + firstI; p++) branchesLabelMap[p->y][p->x] = 0;
            branchYX.erase(branchYX.begin(), branchYX.begin() + firstI);
        } else {
            // If the valid section of the branch is 1 px or less, remove the branch.
            for (auto p : branchYX) branchesLabelMap[p.y][p.x] = 0;
            branchYX.clear();
        }
    }

    return out;
}

bool is_valid_boundaries(const IntPoint &p, const std::array<IntPoint, 2> &boundaries) {
    // 1. Check if the boundaries are valid
    auto const &boundL = boundaries[0], boundR = boundaries[1];
    if (!boundL.is_valid() || !boundR.is_valid()) return false;

    // 2. Check if the skeleton is at the center of the boundaries
    const float dL = distance(p, boundL), dR = distance(p, boundR);
    if (abs(dL - dR) >= 1 + 1.42) return false;

    return true;
}

/**
 * @brief Clean the skeleton of a branch around a node.
 *
 * This methods find the part of the skeleton of the branches around a node which should be removed.
 * For each branch it identifies the first valid pixel of the skeleton which satisfies the following conditions:
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
 * @param adaptativeTangent A boolean indicating if the tangent smoothing should be scaled by the branch calibre.
 *
 * @return A list of tuples containing for each branch of the specified node:
 * - The index of the first valid pixel.
 * - The tangent of the branch at this pixel.
 * - The calibre of the branch at this pixel.
 * - The left boundary of the branch at this pixel.
 * - The right boundary of the branch at this pixel.
 *
 */
std::vector<std::tuple<int, Vector, float, IntPoint, IntPoint>> clean_branch_skeleton_around_node(
    const std::vector<CurveYX> &branchCurves, int nodeID, const std::set<Edge> &node_adjacency,
    const Tensor2DAcc<bool> &segmentation, int maxRemovedLength, bool adaptativeTangent = false) {
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
        const int curveEnd = (int)curveYX.size() - 1;  // Last valid index of the curve
        // If the node is a junction, it should be removed from the branch skeleton
        const int c0 = node_adjacency.size() > 1 ? 1 : 0;

        // The branch exploration should start by the last index if the node is at the end of the branch
        const int start = forward ? c0 : curveEnd - c0;
        const int end = forward ? std::min(start + maxRemovedLength, curveEnd) : std::max(start - maxRemovedLength, 0);
        return std::make_tuple(start, end);
    };

    // === For each branch: search first valid pixel (Lambda Function) ===
    auto find_first_valid_branch_pixel = [&](int branchI) {
        auto [branchID, forward] = branchesInfos[branchI];
        const CurveYX &curveYX = branchCurves[branchID];

        auto const [start, end] = get_branch_start_end(branchID, forward);
        const int curveSize = curveYX.size();

        // === If the branch is too short, remove it ===
        if (curveSize <= 3)
            return std::make_tuple(end, Point(0, 0), INVALID_CALIBRE, IntPoint::Invalid(), IntPoint::Invalid());

        // === Check if pixel is valid (Lambda Function) ===
        auto is_branch_pixel_valid = [&](const IntPoint &p, const std::array<IntPoint, 2> &boundaries,
                                         const std::array<IntPoint, 2> &nextBoundaries, const Point &tangent,
                                         const Point &nextTangent, float calibre, float nextCalibre) {
            // 1. Check if the boundaries are valid
            auto const &boundL = boundaries[0], boundR = boundaries[1];
            if (!boundL.is_valid() || !boundR.is_valid()) return false;

            // 2. Check if the skeleton is at the center of the boundaries
            const float dL = distance(p, boundL), dR = distance(p, boundR);
            if (abs(dL - dR) >= 1.42) return false;

            // 3. Check if the branch width is constant for this pixel and the next
            if (abs(calibre - nextCalibre) > 2.1213) return false;

            // If the node is an endpoint (it should not), the previous conditions are enough to check its validity
            if (node_adjacency.size() == 1) return true;

            // 4. Check if the skeleton closest to the boundaries belong to the current branch
            auto isClosestToCurrentBranch = [&](const IntPoint &bound) {
                float distToCurrentBranch = distance(p, bound);
                for (auto [id, forward] : branchesInfos) {
                    if (id == branchID) continue;
                    const CurveYX &otherCurve = branchCurves[id];
                    auto [otherStart, otherEnd] = get_branch_start_end(id, forward);
                    auto const [closest_p, dist] = find_closest_pixel(otherCurve, bound, otherStart, otherEnd, true);
                    if (dist < distToCurrentBranch) return false;
                }
                return true;
            };
            if (!isClosestToCurrentBranch(boundaries[0]) || !isClosestToCurrentBranch(boundaries[1])) return false;

            return true;
        };

        Point tangent = fast_curve_tangent(curveYX, start, TANGENT_HALF_GAUSS, forward, !forward, 0, curveSize);
        auto boundaries = fast_branch_boundaries(curveYX, start, segmentation, tangent);
        float calibre = fast_branch_calibre(boundaries[0], boundaries[1], tangent);
        if (adaptativeTangent) {
            tangent = adaptative_curve_tangent(curveYX, start, calibre, forward, !forward, 0, curveSize);
            boundaries = fast_branch_boundaries(curveYX, start, segmentation, tangent);
            calibre = fast_branch_calibre(boundaries[0], boundaries[1], tangent);
        }

        const int inc = forward ? 1 : -1;
        int nextI = start + inc;

        Point nextTangent = fast_curve_tangent(curveYX, nextI, TANGENT_HALF_GAUSS, forward, !forward, 0, curveSize);
        std::array<IntPoint, 2> nextBoundaries = fast_branch_boundaries(curveYX, nextI, segmentation, nextTangent);
        float nextCalibre = fast_branch_calibre(nextBoundaries[0], nextBoundaries[1], nextTangent);
        if (adaptativeTangent && nextCalibre == nextCalibre) {
            nextTangent = adaptative_curve_tangent(curveYX, nextI, nextCalibre, forward, !forward, 0, curveSize);
            nextBoundaries = fast_branch_boundaries(curveYX, nextI, segmentation, nextTangent);
            nextCalibre = fast_branch_calibre(nextBoundaries[0], nextBoundaries[1], nextTangent);
        }

        Point nextTangent2;
        float nextCalibre2;
        std::array<IntPoint, 2> nextBoundaries2;

        const int endI = end - inc;
        while (nextI != endI) {
            const int nextI2 = nextI + inc;
            nextTangent2 = fast_curve_tangent(curveYX, nextI2, TANGENT_HALF_GAUSS, forward, !forward, 0, curveSize);
            nextBoundaries2 = fast_branch_boundaries(curveYX, nextI2, segmentation, nextTangent2);
            nextCalibre2 = fast_branch_calibre(nextBoundaries2[0], nextBoundaries2[1], nextTangent2);
            if (adaptativeTangent && nextCalibre2 == nextCalibre2) {
                nextTangent2 = adaptative_curve_tangent(curveYX, nextI2, nextCalibre2, forward, !forward, 0, curveSize);
                nextBoundaries2 = fast_branch_boundaries(curveYX, nextI2, segmentation, nextTangent2);
                nextCalibre2 = fast_branch_calibre(nextBoundaries2[0], nextBoundaries2[1], nextTangent2);
            }

            if (is_branch_pixel_valid(curveYX[nextI - inc], boundaries, nextBoundaries2, tangent, nextTangent2, calibre,
                                      nextCalibre2))
                return std::make_tuple(nextI - inc, tangent, calibre, boundaries[0], boundaries[1]);

            tangent = nextTangent;
            boundaries = nextBoundaries;
            calibre = nextCalibre;

            nextTangent = nextTangent2;
            nextBoundaries = nextBoundaries2;
            nextCalibre = nextCalibre2;
            nextI = nextI2;
        }

        return std::make_tuple(end, tangent, fast_branch_calibre(boundaries[0], boundaries[1], tangent), boundaries[0],
                               boundaries[1]);
    };

    std::vector<std::tuple<int, Vector, float, IntPoint, IntPoint>> out(branchesInfos.size());
    for (std::size_t i = 0; i < branchesInfos.size(); i++) out[i] = find_first_valid_branch_pixel(i);
    return out;
}

/**
 * @brief Clean the skeleton of a branch tip.
 *
 * This methods find the part of the skeleton of a branch tip which should be removed.
 * It identifies the first valid pixel of the skeleton which satisfies the following conditions:
 *  - The pixel is roughly at the center of the branch.
 *  - The branch width is roughly the same for this pixel and the next one in the skeleton.
 *
 *
 * @param branchCurves A list of branches defined by a vector of points.
 * @param branchID The id of the branch to clean.
 * @param startTip A boolean indicating if the tip to clean is at the start or at the end of the branch.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the binary segmentation.
 * @param maxRemovedLength The maximum number of pixels to remove from the branch.
 * @param adaptativeTangent A boolean indicating if the tangent smoothing should be scaled by the branch calibre.
 *
 * @return A tuple containing:
 * - The index of the first valid pixel.
 * - The tangent of the branch at this pixel.
 * - The calibre of the branch at this pixel.
 * - The left boundary of the branch at this pixel.
 * - The right boundary of the branch at this pixel.
 *
 */
std::tuple<int, Vector, float, IntPoint, IntPoint> clean_branch_skeleton_tip(const std::vector<CurveYX> &branchCurves,
                                                                             int branchID, bool startTip,
                                                                             const Tensor2DAcc<bool> &segmentation,
                                                                             int maxRemovedLength,
                                                                             bool adaptativeTangent = false) {
    const CurveYX &curveYX = branchCurves[branchID];
    int start = startTip ? 0 : curveYX.size() - 1;
    const int end = startTip ? std::min(maxRemovedLength, (int)curveYX.size() - 2)
                             : std::max((int)curveYX.size() - 1 - maxRemovedLength, 1);
    const int inc = startTip ? 1 : -1;

    // === If the branch is too short, keep it ===
    if (std::abs(start - end) <= 3)
        return std::make_tuple(start, Point(curveYX[end] - curveYX[start]).normalize(), INVALID_CALIBRE, curveYX[start],
                               curveYX[start]);

    if (end != (startTip ? (int)curveYX.size() - 2 : 1)) {
        const Point endTangent = fast_curve_tangent(curveYX, end, TANGENT_HALF_GAUSS);
        float refCalibre = fast_branch_calibre(curveYX, end, segmentation, endTangent);
        if (refCalibre == refCalibre) start += std::min((int)std::floor(refCalibre / 2), std::abs(start - end)) * inc;
    }

    const int curveStart = startTip ? start : end;
    const int curveEnd = startTip ? end : start;

    // === Check if pixel is valid (Lambda Function) ===
    auto is_branch_pixel_valid = [&](const IntPoint &p, const std::array<IntPoint, 2> &boundaries,
                                     const std::array<IntPoint, 2> &nextBoundaries, const Point &tangent,
                                     const Point &nextTangent, float calibre, float nextCalibre) {
        // 1. Check if the boundaries are valid
        auto const &boundL = boundaries[0], boundR = boundaries[1];
        if (!boundL.is_valid() || !boundR.is_valid()) return false;

        // 2. Check if the skeleton is at the center of the boundaries
        const float dL = distance(p, boundL), dR = distance(p, boundR);
        if (abs(dL - dR) >= 1.42) return false;

        // 3. Check if the branch width is constant for this pixel and the next
        if (abs(calibre - nextCalibre) > 1.42) return false;

        // 4. Check if the tangent is within 20 degrees of the next tangent
        if (tangent.dot(nextTangent) < .94) return false;

        return true;
    };

    Point tangent = fast_curve_tangent(curveYX, start, TANGENT_HALF_GAUSS, startTip, !startTip, curveStart, curveEnd);
    auto boundaries = fast_branch_boundaries(curveYX, start, segmentation, tangent);
    float calibre = fast_branch_calibre(boundaries[0], boundaries[1], tangent);
    if (adaptativeTangent) {
        tangent = adaptative_curve_tangent(curveYX, start, calibre, startTip, !startTip, curveStart, curveEnd);
        boundaries = fast_branch_boundaries(curveYX, start, segmentation, tangent);
        calibre = fast_branch_calibre(boundaries[0], boundaries[1], tangent);
    }

    Point nextTangent;
    float nextCalibre;
    std::array<IntPoint, 2> nextBoundaries;

    for (int i = start; i != end; i += inc) {
        const int nextI = i + inc;
        nextTangent = fast_curve_tangent(curveYX, nextI, TANGENT_HALF_GAUSS, startTip, !startTip, curveStart, curveEnd);
        nextBoundaries = fast_branch_boundaries(curveYX, nextI, segmentation, nextTangent);
        nextCalibre = fast_branch_calibre(nextBoundaries[0], nextBoundaries[1], nextTangent);
        if (adaptativeTangent) {
            nextTangent =
                adaptative_curve_tangent(curveYX, nextI, nextCalibre, startTip, !startTip, curveStart, curveEnd);
            nextBoundaries = fast_branch_boundaries(curveYX, nextI, segmentation, nextTangent);
            nextCalibre = fast_branch_calibre(nextBoundaries[0], nextBoundaries[1], nextTangent);
        }

        if (is_branch_pixel_valid(curveYX[i], boundaries, nextBoundaries, tangent, nextTangent, calibre, nextCalibre))
            return std::make_tuple(i, tangent, calibre, boundaries[0], boundaries[1]);

        tangent = nextTangent;
        boundaries = nextBoundaries;
        calibre = nextCalibre;
    }

    return std::make_tuple(end, tangent, calibre, boundaries[0], boundaries[1]);
}

void remove_small_spurs(float min_length, EdgeList &edgeList, std::vector<CurveYX> &branchCurves,
                        std::vector<IntPoint> &nodeCoords, Tensor2DAcc<int> &labelMap) {
    // Search for the terminal edges
    std::vector<std::pair<Edge, int>> terminalEdges;
    terminalEdges.reserve(edgeList.size());

    auto const &nodesRank = nodes_rank(edgeList);
    for (const auto &edge : edgeList) {
        if (nodesRank[edge.start] == 1)
            terminalEdges.push_back({edge, edge.start});
        else if (nodesRank[edge.end] == 1)
            terminalEdges.push_back({edge, edge.end});
    }

    // Instantiate the list of branches and nodes to remove
    std::vector<Edge> branchToRemove;
    branchToRemove.reserve(terminalEdges.size());
    std::vector<std::size_t> nodeToRemove;
    nodeToRemove.reserve(terminalEdges.size());

    for (auto const &[edge, nodeId] : terminalEdges) {
        // Filter the terminal edges: only select the small spurs for removal
        if (branchCurves[edge.id].size() <= min_length) {
            branchToRemove.push_back(edge);
            nodeToRemove.push_back(nodeId);
        }
    }

    // Remove the branches
    remove_branches(branchToRemove, branchCurves, labelMap, edgeList);
    remove_nodes(nodeToRemove, edgeList, nodeCoords, labelMap);
}

/**
 * @brief Find spurs (small terminal edges) of a graph.
 *
 * @param branchCurves A list of branches skeleton defined by a vector of points.
 * @param adjacency The adjacency list of the vessels graph.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the vessels binary segmentation.
 * @param min_length The minimum length under which a terminal edges is necessarily considered as a spur.
 * @param calibre_factor If positive, a spur is removed if its length is less than the calibre ratio times the
 * maximum calibre of adjacent branches.
 * @param max_length The maximum length of a spur: a spur longer than this is never removed.
 *
 * @return A list of edges representing the terminal edges of the graph.
 */
std::vector<Edge> find_spurs(const std::vector<CurveYX> &branchCurves, const EdgeList &edgeList,
                             const Tensor2DAcc<bool> &segmentation, float min_length, const float calibre_factor,
                             const float max_length) {
    min_length = std::max(min_length, 1.0f);

    // Search for the terminal edges
    auto terminal_edge = terminal_edges(edgeList);

    // Instantiate the list of branches to remove
    std::vector<Edge> branchToRemove;
    branchToRemove.reserve(terminal_edge.size());

    GraphAdjList adjacency;
    if (calibre_factor > 0) adjacency = edge_list_to_adjlist(edgeList);

#pragma omp parallel for reduction(merge : branchToRemove)
    for (auto const &edge : terminal_edge) {
        auto const curveSize = branchCurves[edge.id].size();
        // Filter the terminal edges: only select the small spurs for removal
        if (curveSize <= min_length) {
            branchToRemove.push_back(edge);
        } else if (calibre_factor > 0 && curveSize < max_length) {
            // If the calibres are provided, compute the maximum spur length based on the calibres
            float maxCalibre = std::max(largest_node_calibre(edge.start, adjacency, branchCurves, segmentation),
                                        largest_node_calibre(edge.end, adjacency, branchCurves, segmentation));

            if (curveSize <= maxCalibre * calibre_factor) branchToRemove.push_back(edge);
        }
    }

    return branchToRemove;
}

float largest_node_calibre(const int nodeId, const GraphAdjList &adjacency, const std::vector<CurveYX> &branchesCurves,
                           const Tensor2DAcc<bool> &segmentation) {
    float maxCalibre = -1;
    for (auto const &nearEdge : adjacency[nodeId]) {
        auto const &curve = branchesCurves[nearEdge.id];
        if (curve.empty()) continue;

        int evaluateAtI = nodeId == nearEdge.start ? 0 : (int)curve.size() - 1;
        auto const &tangent = fast_curve_tangent(curve, evaluateAtI);
        const float calibre = fast_branch_calibre(curve, evaluateAtI, segmentation, tangent);
        if (is_valid_calibre(calibre)) maxCalibre = std::max(maxCalibre, calibre);
    }
    return maxCalibre;
}