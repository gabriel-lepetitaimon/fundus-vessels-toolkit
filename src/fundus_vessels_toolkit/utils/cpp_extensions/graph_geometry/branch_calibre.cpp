#include "branch.h"

std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAccessor<bool> &segmentation, const Point &tangent) {
    const IntPoint &p = curveYX[i];
    const float tY = tangent.y, tX = tangent.x;
    return {track_nearest_border(p, {-tX, tY}, segmentation), track_nearest_border(p, {tX, -tY}, segmentation)};
}

std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAccessor<bool> &segmentation) {
    return fast_branch_boundaries(curveYX, i, segmentation, fast_curve_tangent(curveYX, i));
}

/**
 * @brief Find the nearest pixel which doesn't belong to the segmentation following a normal to a tangent.
 *
 *
 * @param curveYX A list of points defining the curve.
 * @param segmentation A 2D tensor of shape (H, W) containing the binary segmentation.
 * @param tangents A list of tangents at each point to be evaluated.
 * @param evaluateAtID A list of indexes where the width should be evaluated. If empty, the width is evaluated at each
 * point.
 *
 * @return A list of pairs of floats representing the left and right width at each point.
 */
std::vector<std::array<IntPoint, 2>> fast_branch_boundaries(const CurveYX &curveYX,
                                                            const Tensor2DAccessor<bool> &segmentation,
                                                            const std::vector<Point> &tangents,
                                                            const std::vector<int> &evaluateAtID) {
    const std::size_t curveSize = curveYX.size();
    bool evaluateAll = evaluateAtID.size() == 0;
    const std::size_t outSize = evaluateAll ? curveSize : evaluateAtID.size();
    std::vector<std::array<IntPoint, 2>> boundaries(outSize, {0, 0});

    assert((tangents.size() == outSize) &&
           "The number of tangents should be equal to the number of points to evaluate.");

    // Compute the width at each point
    for (std::size_t pointI = 0; pointI < outSize; pointI++) {
        const std::size_t i = evaluateAll ? pointI : evaluateAtID[pointI];
        if (i < 0 || i >= curveSize) continue;
        const Point &tangent = tangents[pointI];
        boundaries[pointI] = fast_branch_boundaries(curveYX, i, segmentation, tangent);
    }

    return boundaries;
}

/**
 * @brief Evaluate the width of a branch.
 *
 * This method compute the width of a branch by tracking the nearest pixels which doesn't belong to the segmentation
 * following the normals to the tangents provided. It returns the left and right width at each point.
 *
 * @param curveYX A list of points defining the curve.
 * @param segmentation A 2D tensor of shape (H, W) containing the binary segmentation.
 * @param tangents A list of tangents at each point to be evaluated.
 * @param evaluateAtID A list of indexes where the width should be evaluated. If empty, the width is evaluated at each
 * point.
 *
 * @return A list of floats representing the width at each point.
 */
std::vector<float> fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAccessor<bool> &segmentation,
                                       const std::vector<Point> &tangents, const std::vector<int> &evaluateAtID) {
    auto const &boundaries = fast_branch_boundaries(curveYX, segmentation, tangents, evaluateAtID);

    const std::size_t outSize = boundaries.size();
    std::vector<float> widths(outSize, 0.0);

    IntPoint segShape(segmentation.sizes()[0], segmentation.sizes()[1]);

    const float DIAGONAL_COMPENSATION = SQRT2 / 6;
    for (std::size_t pointI = 0; pointI < outSize; pointI++) {
        IntPoint boundL(boundaries[pointI][0]);
        IntPoint boundR(boundaries[pointI][1]);
        if (!boundL.is_valid() || !boundR.is_valid()) widths[pointI] = std::numeric_limits<float>::quiet_NaN();
        float dist = distance(boundL, boundR);

        // If the skeleton is diagonal, we take into account the neighbors pixels to compensate for the discretization

        float tx = tangents[pointI].x, ty = tangents[pointI].y;
        if (tx != 0 && abs(ty / tx) < 2 && abs(ty / tx) > 0.5) {
            IntPoint l1(boundL.y - sign(tx), boundL.x), l2(boundL.y, boundL.x + sign(ty));
            IntPoint r1(boundR.y + sign(tx), boundR.x), r2(boundR.y, boundR.x - sign(ty));
            if (l1.is_inside(segShape) && segmentation[l1.y][l1.x]) dist += DIAGONAL_COMPENSATION;
            if (l2.is_inside(segShape) && segmentation[l2.y][l2.x]) dist += DIAGONAL_COMPENSATION;
            if (r1.is_inside(segShape) && segmentation[r1.y][r1.x]) dist += DIAGONAL_COMPENSATION;
            if (r2.is_inside(segShape) && segmentation[r2.y][r2.x]) dist += DIAGONAL_COMPENSATION;
        }

        widths[pointI] = dist;
    }

    return widths;
}

std::vector<int> clean_branch_skeleton_around_node(const std::vector<CurveYX> &branchCurves, const int nodeID,
                                                   const std::set<Edge> &node_adjacency,
                                                   const Tensor2DAccessor<bool> &segmentation,
                                                   const int maxRemovedLength) {
    // === Preparation ==
    // Read branches id and side
    std::vector<std::tuple<int, bool>> branchesInfos;
    branchesInfos.reserve(node_adjacency.size());
    for (const auto &edge : node_adjacency) {
        if (edge.start == nodeID) branchesInfos.push_back({edge.id, true});
        if (edge.end == nodeID) branchesInfos.push_back({edge.id, false});
    }

    auto get_branch_start_end = [&](int branchID, bool forward) {
        const CurveYX &curveYX = branchCurves[branchID];
        const int start = forward ? 0 : curveYX.size() - 1;
        const int end = forward ? std::min(maxRemovedLength, (int)curveYX.size() - 1)
                                : std::max((int)curveYX.size() - 1 - maxRemovedLength, 0);
        return std::make_tuple(start, end);
    };

    // === For each branch: search first valid pixel (Lambda Function) ===
    auto find_first_valid_branch_pixel = [&](int branchI) {
        auto [branchID, forward] = branchesInfos[branchI];
        const CurveYX &curveYX = branchCurves[branchID];

        const int inc = forward ? 1 : -1;
        auto const [start, end] = get_branch_start_end(branchID, forward);

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
            if (is_branch_pixel_valid(curveYX[i], boundaries, nextBoundaries)) return i;

            tangent = nextTangent;
            boundaries = nextBoundaries;
        }

        return end;
    };

    std::vector<int> out(branchesInfos.size());
    for (std::size_t i = 0; i < branchesInfos.size(); i++) out[i] = find_first_valid_branch_pixel(i);
    return out;
}