#include "branch.h"

std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAcc<bool> &segmentation, const Point &tangent) {
    const IntPoint &p = curveYX[i];
    const float tY = tangent.y, tX = tangent.x;
    return {track_nearest_border(p, {-tX, tY}, segmentation), track_nearest_border(p, {tX, -tY}, segmentation)};
}

std::array<IntPoint, 2> fast_branch_boundaries(const CurveYX &curveYX, const std::size_t i,
                                               const Tensor2DAcc<bool> &segmentation) {
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
                                                            const Tensor2DAcc<bool> &segmentation,
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
 * @brief Evaluate the width of a branch at a single point.
 *
 * This method compute the width of a branch given the left and right boundaries and the tangent at the point.
 *
 * @param boundL The left boundary of the branch.
 * @param boundR The right boundary of the branch.
 * @param tangents A list of tangents at each point to be evaluated.
 *
 * @return A float representing the width at the point.
 */
float fast_branch_calibre(const Point &boundL, const Point &boundR, const Point &tangent) {
    float dist = distance(boundL, boundR);

    // If the skeleton is diagonal, we take into account the neighbors pixels to compensate for the discretization
    /*
    const float DIAGONAL_COMPENSATION = SQRT2 / 6;
    float tx = tangents[pointI].x, ty = tangents[pointI].y;
    if (tx != 0 && abs(ty / tx) < 2 && abs(ty / tx) > 0.5) {
        IntPoint l1(boundL.y - sign(tx), boundL.x), l2(boundL.y, boundL.x + sign(ty));
        IntPoint r1(boundR.y + sign(tx), boundR.x), r2(boundR.y, boundR.x - sign(ty));
        if (l1.is_inside(segShape) && segmentation[l1.y][l1.x]) dist += DIAGONAL_COMPENSATION;
        if (l2.is_inside(segShape) && segmentation[l2.y][l2.x]) dist += DIAGONAL_COMPENSATION;
        if (r1.is_inside(segShape) && segmentation[r1.y][r1.x]) dist += DIAGONAL_COMPENSATION;
        if (r2.is_inside(segShape) && segmentation[r2.y][r2.x]) dist += DIAGONAL_COMPENSATION;
    }
    */

    if (!tangent.is_null()) {
        Point pixelDiagonal = tangent.positiveCoordinates();
        pixelDiagonal = pixelDiagonal / std::max(pixelDiagonal.x, pixelDiagonal.y);
        dist += pixelDiagonal.norm();
    }
    return dist;
}

/**
 * @brief Evaluate the width of a branch at a single point.
 *
 * This method compute the width of a branch by tracking the nearest pixels which doesn't belong to the segmentation
 * following the normals to the tangents provided.
 *
 * @param curveYX A list of points defining the curve.
 * @param i The index of the point to evaluate.
 * @param segmentation A 2D tensor of shape (H, W) containing the binary segmentation.
 * @param tangents A list of tangents at each point to be evaluated.
 *
 * @return A float representing the width at the point.
 */
float fast_branch_calibre(const CurveYX &curveYX, std::size_t i, const Tensor2DAcc<bool> &segmentation,
                          const Point &tangent) {
    auto [boundL, boundR] = fast_branch_boundaries(curveYX, i, segmentation, tangent);
    if (!boundL.is_valid() || !boundR.is_valid()) return std::numeric_limits<float>::quiet_NaN();
    return fast_branch_calibre(boundL, boundR, tangent);
}

/**
 * @brief Evaluate the width of a branch.
 *
 * This method compute the width of a branch by tracking the nearest pixels which doesn't belong to the segmentation
 * following the normals to the tangents provided.
 *
 * @param curveYX A list of points defining the curve.
 * @param segmentation A 2D tensor of shape (H, W) containing the binary segmentation.
 * @param tangents A list of tangents at each point to be evaluated.
 * @param evaluateAtID A list of indexes where the width should be evaluated. If empty, the width is evaluated at each
 * point.
 *
 * @return A list of floats representing the width at each point.
 */
Scalars fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAcc<bool> &segmentation,
                            const std::vector<Point> &tangents, const std::vector<int> &evaluateAtID) {
    const std::size_t curveSize = curveYX.size();
    bool evaluateAll = evaluateAtID.size() == 0;
    const std::size_t outSize = evaluateAll ? curveSize : evaluateAtID.size();
    Scalars calibres(outSize, std::numeric_limits<float>::quiet_NaN());

    for (std::size_t pointI = 0; pointI < outSize; pointI++) {
        const std::size_t i = evaluateAll ? pointI : evaluateAtID[pointI];
        if (i < 0 || i >= curveSize) continue;
        calibres[i] = fast_branch_calibre(curveYX, i, segmentation, tangents[pointI]);
    }

    return calibres;
}

/**
 * @brief Evaluate the width of a branch.
 *
 * This method compute the width of a branch by tracking the nearest pixels which doesn't belong to the segmentation
 * following the normals to the tangents provided.
 *
 * @param curveYX A list of points defining the curve.
 * @param segmentation A 2D tensor of shape (H, W) containing the binary segmentation.
 * @param tangents A list of tangents at each point to be evaluated.
 * @param evaluateAtID A list of indexes where the width should be evaluated. If empty, the width is evaluated at each
 * point.
 *
 * @return A list of floats representing the width at each point.
 */

Scalars fast_branch_calibre(const CurveYX &curveYX, const Tensor2DAcc<bool> &segmentation,
                            const std::vector<int> &evaluateAtID) {
    return fast_branch_calibre(curveYX, segmentation, fast_curve_tangent(curveYX, TANGENT_HALF_GAUSS, evaluateAtID),
                               evaluateAtID);
}