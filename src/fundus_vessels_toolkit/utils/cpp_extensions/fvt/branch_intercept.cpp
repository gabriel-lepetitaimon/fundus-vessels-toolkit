#include "bezier.h"
#include "branch.h"

/**
 * @brief Track the not-segmented pixel on a semi-infinite line defined by a
 * start point and a direction.
 *
 * @param start The start point of the line.
 * @param direction The direction of the line.
 * @param segmentation An accessor to a 2D tensor of shape (H, W) containing the
 * binary segmentation.
 * @param max_distance The maximum distance to track.
 *
 * @return The first point of the line for which the segmentation is false. If
 * no such point is found, return IntPoint::Invalid().
 */
IntPoint track_nearest_edge(const IntPoint& start, const Point& direction, const Tensor2DAcc<bool>& segmentation,
                            int max_distance) {
    if (direction.is_null()) return IntPoint::Invalid();
    const int H = segmentation.size(0), W = segmentation.size(1);

    Point current = {(float)start.y, (float)start.x};
    IntPoint next = start;
    IntPoint last;
    int i = 0;
    do {
        last = next;
        current += direction;
        next = current.toInt();
        i++;
    } while (i < max_distance && next.is_inside(H, W) && segmentation[next.y][next.x]);
    if (i == max_distance) return IntPoint::Invalid();
    return last;
}

/**
 * @brief Find the closest pixel to a point in a curve.
 *
 * This method returns the index of the closest pixel to a point in a curve.
 * The search is performed between the start and end indices.
 *
 * @param curve A list of points defining the curve.
 * @param p The point to which the distance should be computed.
 * @param start The start index of the search.
 * @param end The end index of the search.
 * @param mode If true, the search bisects the curve until finding a local minimum. If false, every pixels of the
 * curve is checked and the global minimum is returned.
 *
 * @return A tuple containing the index of the closest pixel and the distance to
 * the point.
 */
std::tuple<int, float> find_closest_pixel(const CurveYX& curve, const Point& p, int start, int end,
                                          SearchStrategy strategy) {
    if (start == end) return {start, distance(curve[start], p)};
    if (strategy == SearchStrategy::LastLocalMinimum) std::swap(start, end);
    const int inc = (start < end) ? 1 : -1;
    std::tuple<int, float> min_point = {0, distanceSqr(curve[start], p)};
    if (strategy == SearchStrategy::Bisection) {
        int left = start, right = end;
        double dist_left = distanceSqr(curve[left], p), dist_right = distanceSqr(curve[right], p);
        while (left != right) {
            int mid = left + ((right - left) / 2);
            float dist_mid = distanceSqr(curve[mid], p);
            if (dist_mid < std::get<1>(min_point)) {
                min_point = {mid, dist_mid};
            }
            if (dist_left < dist_right) {
                right = mid;
                dist_right = dist_mid;
            } else {
                left = mid + inc;
                dist_left = dist_mid;
            }
        }
    } else {
        for (int i = start + inc; i != end; i += inc) {
            float dist = distanceSqr(curve[i], p);
            if (dist <= std::get<1>(min_point))
                min_point = {i, dist};
            else if (strategy != SearchStrategy::GlobalMinimum)
                break;
        }
    }
    return {std::get<0>(min_point), std::sqrt(std::get<1>(min_point))};
}

std::pair<torch::Tensor, torch::Tensor> find_closest_branches(const torch::Tensor& branch_labels,
                                                              const torch::Tensor& points,
                                                              const torch::Tensor& direction, float max_dist,
                                                              float angle) {
    TORCH_CHECK(points.ndimension() == 2 && points.size(1) == 2,
                "Invalid argument points: should have a shape of (N, 2) instead of", points.sizes());
    const std::size_t N = points.size(0);
    TORCH_CHECK(direction.ndimension() == 2 && (std::size_t)direction.size(0) == N && direction.size(1) == 2,
                "Invalid argument direction: should have a shape of (", N, ", 2) instead of", direction.sizes());

    auto points_acc = points.accessor<int, 2>();
    auto direction_acc = direction.accessor<float, 2>();
    auto branch_labels_acc = branch_labels.accessor<int, 2>();
    std::vector<std::pair<uint, IntPoint>> out(N);

    auto branch = torch::empty({(long)N}, torch::kInt);
    auto intercept = torch::empty({(long)N, 2}, torch::kInt);
    auto branch_acc = branch.accessor<int, 1>();
    auto intercept_acc = intercept.accessor<int, 2>();

#pragma omp parallel for
    for (std::size_t i = 0; i < N; i++) {
        auto start = IntPoint(points_acc[i][0], points_acc[i][1]);
        auto dir = Point(direction_acc[i][0], direction_acc[i][1]);
        const auto& [b, p] = track_nearest_branch(start, dir, angle, max_dist, branch_labels_acc);
        branch_acc[i] = b - 1;
        intercept_acc[i][0] = p.y;
        intercept_acc[i][1] = p.x;
    }

    return {branch, intercept};
}

/**
 * @brief Find the intersection point of a curve within a cone defined by a point, a direction and a thecosine of
 * the angle.
 * @param curve The curve to intersect with. Warning: Assume the curve is non-empty!
 * @param start The starting point of the line.
 * @param dir The direction of the cone bisector.
 * @param maxDistSqr The square of the maximum distance to consider.
 * @param startMinCosSim The minimum cosine similarity at the start point.
 * @param endMinCosSim The minimum cosine similarity at the end point.
 * @param minSnapDistSqr The square of the minimum distance under which the intersection is automatically snapped to
 * the closest curve tip regardless of the angle.
 * @param maxSnapDistSqr The square of the maximum distance under which the intersection snaps to the closest curve
 * tip.
 * @param maxSnapCosAngle The cosine of the maximum angle under which the intersection snaps.
 * @return A tuple containing the index of the closest point on the curve, the squared distance to it, and the
 * average square distance to every point on the curve.
 */
std::tuple<std::size_t, int, float> _intercept_curve(const CurveYX& curve, const IntPoint& start, const Point& dir,
                                                     float maxDistSqr, float startMinCosSim, float endMinCosSim,
                                                     float minSnapDistSqr, float maxSnapDistSqr,
                                                     float maxSnapCosAngle) {
    std::size_t closestP = curve.size();
    int closestDistSqr = maxDistSqr;
    float closestManhattanDist = std::numeric_limits<float>::max();
    float avgSqrDist = 0.0f;

    for (std::size_t i = 0; i < curve.size(); i++) {
        const IntPoint p = curve[i] - start;

        // Check if the point is closer than the previous closest point (or the initial maxDistance)
        int distSqr = p.squaredNorm();
        if (distSqr > maxDistSqr) continue;
        avgSqrDist += distSqr;

        // Check if the point is inside the cone
        const float dist = std::sqrt(distSqr), a = distSqr / maxDistSqr;
        const float cosSim = dir.dot(p) / dist;
        const float minCosSimAtDist = startMinCosSim * (1 - a) + endMinCosSim * a;
        if (cosSim < minCosSimAtDist) continue;

        // Check if the point is closer regarding the manhattan Dist (to favor points aligned with the cone
        // bisector) const float sin = sqrt(1 - cosSim * cosSim);
        const float manhattanDist = (2 - cosSim * cosSim) * dist;  // Equivalent to dist * (|sin| + |cos|)
        if (manhattanDist > closestManhattanDist) continue;

        // Record the closest point and distance
        closestP = i;
        closestDistSqr = distSqr;
        closestManhattanDist = manhattanDist;
    }

    avgSqrDist /= curve.size();

    // If no point was found, return
    if (closestP == curve.size()) return {closestP, -1, avgSqrDist};

    // Try to snap to the nearest curve tip
    if ((maxSnapDistSqr > 0 || minSnapDistSqr > 0) && closestP != 0 && closestP != curve.size() - 1) {
        bool lastTip = closestP > curve.size() - closestP;
        const auto& tipP = lastTip ? curve.back() : curve.front();
        const auto& p = curve[closestP];

        // If the snapping tip is within the allowed distance and angle, snap to it
        const float sqrNorm = (p - tipP).squaredNorm();
        if (sqrNorm <= minSnapDistSqr ||
            (sqrNorm <= maxSnapDistSqr && (tipP - start).cosSim(p - start) >= maxSnapCosAngle))
            closestP = lastTip ? curve.size() - 1 : 0;
    }
    return {closestP, closestDistSqr, avgSqrDist};
}

struct InterceptIntermediateResults {
    std::size_t curveID;
    std::size_t posInCurve;
    int distSqr;
    float avgDistSqr;
};

std::vector<std::list<InterceptPoint>> intercept_curves(const std::vector<CurveYX>& branchCurves,
                                                        const std::vector<IntPair>& branchList,
                                                        const GraphAdjList& graph, const std::vector<IntPoint>& nodesYX,
                                                        const std::vector<IntPoint>& starts, const PointList& dirs,
                                                        float maxDistSqr, float startMinCosSim, float endMinCosSim,
                                                        float minSnapDistSqr, float maxSnapDistSqr,
                                                        float maxSnapCosAngle, bool interpolateCurves) {
    // === INTERPOLATE CURVES ===
    std::vector<CurveYX> curves;
    std::vector<std::vector<int>> curvesIndices(branchCurves.size());

    if (interpolateCurves) {
        curves.resize(branchCurves.size());
        for (std::size_t i = 0; i < branchCurves.size(); i++) {  // For each branch add missing points in its curve
            const auto& curve = branchCurves[i];
            const auto& nodes = branchList[i];

            CurveYX& interCurve = curves[i];
            std::vector<int>& indices = curvesIndices[i];

            if (curve.size() == 0) {
                // IF CURVE IS EMPTY
                // Starting node -> End node
                for (const auto& p : Line(nodesYX[nodes[0]], nodesYX[nodes[1]], true)) interCurve.push_back(p);
                indices.resize(interCurve.size(), 0);  // Pad with 0
            } else {
                // OTHERWISE
                // - Starting node -> First curve point
                for (const auto& p : Line(nodesYX[nodes[0]], curve.front(), true)) interCurve.push_back(p);
                indices.resize(interCurve.size(), 0);  // Add 0 at the beginning of indices
                for (std::size_t i = 0; i < curve.size() - 1; i++) {
                    const auto &p1 = curve[i], &p2 = curve[i + 1];
                    if (p1.is_adjacent(p2)) {
                        interCurve.push_back(p1);
                        indices.push_back(i);
                    } else {
                        for (const auto& p : Line(p1, p2, false)) interCurve.push_back(p);  // Fill the gap
                        // Fill indices with...
                        float delta = (interCurve.size() - indices.size()) / 2.0;
                        indices.resize(indices.size() + ceil(delta), i);  // ... i for the first half
                        indices.resize(interCurve.size(), i + 1);         // ... i+1 for the second half
                    }
                }
                // - Last curve point -> Ending node (skipping the first pixel)
                for (const auto& p : Line(curve.back(), nodesYX[nodes[1]], false, true)) interCurve.push_back(p);
                indices.resize(interCurve.size(), curve.size() - 1);  // Pad indices with max_index
            }
        }
    } else
        curves = branchCurves;

    // === SCAN FOR INTERCEPT POINTS ===
    std::vector<std::list<InterceptPoint>> result(starts.size());

#pragma omp parallel for
    for (std::size_t startID = 0; startID < starts.size(); startID++) {
        const auto& p = starts[startID];
        const auto& dir = dirs[startID];

        std::vector<InterceptIntermediateResults> intercepts;
        intercepts.reserve(branchCurves.size());

        // Find intercept points with each curve
        for (std::size_t curveID = 0; curveID < branchCurves.size(); curveID++) {
            auto [pointID, distSqr, avgDistSqr] =
                _intercept_curve(curves[curveID], p, dir, maxDistSqr, startMinCosSim, endMinCosSim, minSnapDistSqr,
                                 maxSnapDistSqr, maxSnapCosAngle);
            intercepts.emplace_back(InterceptIntermediateResults{curveID, pointID, distSqr, avgDistSqr});
        }

        // Deduplicates intercept points
        std::size_t nodeID = 0;
        for (const auto& adjacentBranches : graph) {
            std::list<std::size_t> duplicates;
            for (const auto& branch : adjacentBranches) {
                if (intercepts[branch.id].posInCurve == (branch.is_first(nodeID) ? 0 : curves[branch.id].size() - 1))
                    duplicates.push_back(branch.id);
            }

            if (duplicates.size() > 1) {
                // Find the closest intercept point
                std::size_t closest = duplicates.front();
                for (const auto& id : duplicates) {
                    int distDiff = intercepts[closest].distSqr - intercepts[id].distSqr;
                    if (abs(distDiff) > maxSnapDistSqr) {
                        if (distDiff > 0) closest = id;
                    } else {
                        if (intercepts[closest].avgDistSqr > intercepts[id].avgDistSqr) closest = id;
                    }
                }

                // Mark the others as invalid
                for (const auto& id : duplicates) {
                    if (id != closest) intercepts[id].distSqr = -1;
                }
            }

            nodeID++;
        }

        // Populate results
        for (const auto& intercept : intercepts) {
            if (intercept.distSqr <= 0) continue;  // No intercept or duplicate
            const auto& pos = curves[intercept.curveID][intercept.posInCurve];
            if (pos == p) continue;  // Intercept is at the start point

            result[startID].emplace_back(
                InterceptPoint{intercept.curveID, curvesIndices[intercept.curveID][intercept.posInCurve], pos});
        }
    }

    return result;
}

struct InterceptCandidate {
    std::size_t i;
    float l;
    bool towardsBefore;
    bool towardsAfter;
    float score;

    std::size_t b0 = 0;
    int tip0 = 0;
    std::size_t b1 = 0;
};

struct ConnexionCandidate {
    std::size_t b0 = 0;
    int tip0 = 0;
    std::size_t b1 = 0;
    int tip1 = 0;
};

ConnexionCandidate symmetric_connexion(const ConnexionCandidate& c) {
    return ConnexionCandidate{c.b1, c.tip1, c.b0, c.tip0};
}

#pragma omp declare reduction( \
        merge : std::list<ConnexionCandidate> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction( \
        merge : std::list<InterceptCandidate> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

/**
 * @brief Find the intersection point of a curve within a cone defined by a point, a direction and a the cosine of
 * the angle.
 * @param curve The curve to intersect with. Warning: Assume the curve is non-empty!
 * @param curveTan The tangent of the curve to intersect with. Warning: Assume the have the same size!
 * @param coneApex The position of thecone apex.
 * @param coneDir The direction of the cone bisector.
 * @param coneSqrHeight The square of the cone height (distance from the apex to the base).
 * @param coneApexCos The cosine of the angle at the apex of the cone.
 * @param coneEndCos The cosine of the angle at the end of the cone (at the base).
 * @param minCosSim The minimum cosine similarity between cone direction and curve tangent at the intersection
 * point.
 * @param minHypCosSim The minimum cosine similarity between the vector from the cone apex to the intersection
 * point and the curve tangent at the intersection
 * point.
 * @return
 *  - The list of intercept candidates, each containing the index of the closest point on the curve, the proximity
 * score, and the direction of the curve tangent at the intersection point.
 *  - The total length of the curve.
 */
std::tuple<std::list<InterceptCandidate>, float> _cone_curve_intercept(
    const CurveYX& curve, const std::vector<Point>& curveTan, const IntPoint& coneApex, const Point& coneDir,
    float coneSqrHeight, float coneApexCos, float coneEndCos, float minCosSim, float minHypCosSim,
    float minSpaceBetweenSplits) {
    float minScore = std::numeric_limits<float>::max();

    std::list<InterceptCandidate> intercepts;
    /// TODO: Early stop if the curve start, end and mid points are on the opposite side of the cone bisector

    // Lambda function to compute score
    auto computeScore = [&](std::size_t i) {
        const IntPoint p = curve[i] - coneApex;

        // === Check if the point is inside the cone ===
        // 1. Check if the point is inside a circle of radius coneHeight
        int distSqr = p.squaredNorm();
        if (distSqr > coneSqrHeight) return std::numeric_limits<float>::max();

        // 2. Check the point angle with the cone bisector
        const float dist = std::sqrt(distSqr), a = distSqr / coneSqrHeight;
        // const Point pUnit = Point(p) / dist;
        const float pAngleCos = coneDir.dot(p) / dist;
        const float coneCos = coneApexCos * (1 - a) + coneEndCos * a;
        if (pAngleCos < coneCos) return std::numeric_limits<float>::max();

        // === Compute the intercept score ===
        // Check if the point is closer regarding the manhattan Dist (to favor points aligned with the cone
        // bisector) const float sin = sqrt(1 - cosSim * cosSim);
        return (2 - pAngleCos * pAngleCos) * dist;  // Equivalent to dist * (|sin| + |cos|)
    };

    // Lambda function to register intercept candidates
    auto registerIntercept = [&](std::size_t i, float l, float score) {
        float tanSim = curveTan[i].dot(coneDir);
        float hypCosSim = curveTan[i].dot((curve[i] - coneApex).normalize());
        bool towardsLastTip = tanSim >= minCosSim && hypCosSim >= minHypCosSim;
        bool towardsFirstTip = -tanSim >= minCosSim && -hypCosSim >= minHypCosSim;
        if (towardsLastTip || towardsFirstTip) {
            intercepts.emplace_back(InterceptCandidate{i, l, towardsFirstTip, towardsLastTip, score});
            if (score < minScore) minScore = score;
        }
    };

    float l = 0, prevScore = std::numeric_limits<float>::max(), score = computeScore(0);
    for (std::size_t i = 0; i < curve.size() - 1; i++) {
        float nextScore = computeScore(i + 1);
        if (prevScore > score && score < nextScore) registerIntercept(i, l, score);
        prevScore = score;
        score = nextScore;
        l += curve[i + 1].distance(curve[i]);
    }
    if (prevScore > score) registerIntercept(curve.size() - 1, l, score);

    // === FILTER INTERCEPTS ===
    // only keep local minimum equivalent to the best score
    intercepts.remove_if([minScore](const InterceptCandidate& c) { return c.score > 1.5 * minScore; });

    // Remove intercepts that are too close to each other
    if (minSpaceBetweenSplits > 0) {
        auto it = intercepts.begin(), nextIt = std::next(it);
        while (nextIt != intercepts.end()) {
            if (nextIt->l - it->l < minSpaceBetweenSplits) {
                if (nextIt->score < it->score) {
                    intercepts.erase(it);
                    it = nextIt;
                    nextIt++;
                } else
                    nextIt = intercepts.erase(nextIt);
            } else {
                it = nextIt;
                nextIt++;
            }
        }
    }

    return {intercepts, l};
}

/**
 * @brief Find possible connexions between branches.
 * @param branchCurves The curves of the branches.
 * @param branchTangents The tangents of the branches.(Must have the same size as branchCurves)
 * @param branchList The list of branches, each containing the indices of its starting and ending nodes.
 * @param nodesYX The coordinates of the nodes.
 * @param maxDist The maximum distance to consider for a connexion.
 * @param nearConeAngle The angle in degrees of the cone near the emitting branch tip.
 * @param farConeAngle The angle in degrees of the cone at maxDist from the emitting branch tip.
 * @param maxTanAngle The maximum angle in degrees between the tangents of the emitting and receiving branch tips.
 * @param snapDist The distance under which the intersection snaps to the closest curve tip.
 * @param minSpaceBetweenSplits The minimum space between two splits on the same branch.
 * @param mergeNodeDist The distance under which two not-connected nodes are considered to be the same.
 * @return
 * splits:
 *      A tuple containing the list of splits as a tuple containing the branch to split and the position of the splits;
 * connexion_candidates:
 *      A integer tensor of shape (C, 4) where each row is in the form (b0, tip0, b1, tip1) where:
 *      - b0 is the index of the emitting branch
 *      - tip0 is the index of the emitting branch tip (0 for the first tip, 1 for the last tip)
 *      - b1 is the index of the intercepted branch
 *      - tip1 is the index of the intercepted branch tip (0 for the first tip, 1 for the last tip)
 */
std::tuple<std::vector<std::pair<int, Splits>>, torch::Tensor> branch_connexion_candidates(
    const std::vector<torch::Tensor>& branchCurves, const std::vector<torch::Tensor>& branchTangents,
    const torch::Tensor& branchListTensor, const torch::Tensor& nodesYX, const IntPair& shape, float maxDist,
    float nearConeAngle, float farConeAngle, float maxTanAngle, float maxHypAngle, float snapDist,
    float minSpaceBetweenSplits, float mergeNodeDist) {
    // === PREPROCESS INPUTS ===
    std::size_t B = branchCurves.size(), N = nodesYX.size(0);
    if (B == 0 || N == 0) return {std::vector<std::pair<int, Splits>>(), torch::empty({0, 2, 0, 2}, torch::kBool)};

    std::vector<CurveYX> curves;
    tensors_to_vectors(branchCurves, curves);

    TORCH_CHECK(branchTangents.size() == B, "branchTangents must have the same size as branchCurves");
    std::vector<PointList> tangents;
    tensors_to_vectors(branchTangents, tangents);

    TORCH_CHECK(
        branchListTensor.dim() == 2 && (std::size_t)branchListTensor.size(0) == B && branchListTensor.size(1) == 2,
        "branchList must have shape (B, 2) where B is the number of branches");
    std::vector<IntPair> branchList;
    tensor_to_vector(branchListTensor, branchList);

    TORCH_CHECK(nodesYX.dim() == 2 && nodesYX.size(1) == 2,
                "nodesYX must have shape (N, 2) where N is the number of nodes");
    std::vector<IntPoint> nodesPoint;
    tensor_to_vector(nodesYX, nodesPoint);

    GraphAdjList graph = edge_list_to_adjlist(branchList, N);

    // === INTERPOLATE CURVES ===
    std::vector<std::array<IntPoint, 2>> tipsPos(B);
    std::vector<std::array<Point, 2>> tipsTangents(B);
    std::vector<std::vector<int>> curvesInitialIndices(B);

#pragma omp parallel for
    for (std::size_t i = 0; i < B; i++) {  // For each branch add missing points in its curve
        const auto initialCurve = curves[i];
        const auto initialTangents = tangents[i];
        const auto &n0 = nodesPoint[branchList[i][0]], &n1 = nodesPoint[branchList[i][1]];

        CurveYX& curve = curves[i];
        curve.clear();
        std::vector<Point>& tangent = tangents[i];
        tangent.clear();
        std::vector<int>& indices = curvesInitialIndices[i];

        // The first and last point of the curve (on the nodes) are only kept if the branch is the "first" branch
        // connected to this node, so that only one branch take the "ownership" if the node pixel.
        bool skipFirst = graph[branchList[i][0]].begin()->id != (int)i,
             skipLast = graph[branchList[i][1]].begin()->id != (int)i;

        if (initialCurve.size() == 0) {
            // IF CURVE IS EMPTY
            // Use node as tips positions and tangents
            tipsPos[i] = {n0, n1};
            const Point& t = (n1 - n0).normalize();
            tipsTangents[i] = {-t, t};

            // Interpolate Starting node -> End node
            for (const auto& p : Line(n0, n1, skipLast, skipFirst)) curve.push_back(p);
            tangent.resize(curve.size(), t);  // Pad with t
            indices.resize(curve.size(), 0);  // Pad with 0
        } else {
            // OTHERWISE
            // Record tips positions and tangents
            tipsPos[i] = {initialCurve.front(), initialCurve.back()};
            tipsTangents[i] = {-initialTangents.front(), initialTangents.back()};

            // Interpolate the curve:
            // - Starting node -> First curve point
            std::size_t i = 0;
            if (n0 != initialCurve.front()) {
                BezierIterator it(n0, {0, 0}, initialCurve.front(), -initialTangents.front());
                if (skipFirst) it.next();
                while (it.next()) {
                    curve.push_back(it.p());
                    tangent.push_back(it.t());
                }
                curve.pop_back(), tangent.pop_back();  // Remove the last point to avoid duplicates
                indices.resize(curve.size(), 0);       // Pad indices with 0
            } else if (skipFirst)
                i = 1;
            for (; i < initialCurve.size() - 1; i++) {
                const auto &p1 = initialCurve[i], &p2 = initialCurve[i + 1];
                if (p1.is_adjacent(p2)) {
                    curve.push_back(p1);
                    tangent.push_back(initialTangents[i]);
                    indices.push_back(i);
                } else {
                    // Interpolate p1 -> p2 with a bezier curve
                    BezierIterator it(p1, initialTangents[i], p2, -initialTangents[i + 1]);
                    while (it.next()) {
                        curve.push_back(it.p());
                        tangent.push_back(it.t());
                    }
                    curve.pop_back(), tangent.pop_back();  // Remove the last points
                    // Fill indices with...
                    float half = ceil((curve.size() - indices.size()) / 2.0);
                    indices.resize(indices.size() + half, i);  // ... i for the first half
                    indices.resize(curve.size(), i + 1);       // ... i+1 for the second half
                }
            }
            if (initialCurve.back() != n1) {
                // Interpolate last curve point -> Ending node
                BezierIterator it(initialCurve.back(), initialTangents.back(), n1, {0, 0});
                while (it.next()) {
                    curve.push_back(it.p());
                    tangent.push_back(it.t());
                }
                if (skipLast) curve.pop_back(), tangent.pop_back();     // Remove the last points
                indices.resize(curve.size(), initialCurve.size() - 1);  // Pad indices with max_index
            } else if (!skipLast) {
                curve.push_back(initialCurve.back());
                tangent.push_back(initialTangents.back());
                indices.push_back(initialCurve.size() - 1);
            }
        }
    }

    // === Utility function to split a branch ===
    std::vector<std::pair<int, Splits>> splits;
    auto split_branch = [&](std::size_t b, std::vector<std::size_t> splitsAt, std::vector<std::size_t> splitsNode,
                            bool updateCurves = true) -> std::vector<std::size_t> {
        // Create placeholder for the new branch
        std::size_t s = splitsAt.size();
        std::size_t B = branchList.size();
        branchList.resize(B + s);
        tipsPos.resize(B + s);
        tipsTangents.resize(B + s);
        if (updateCurves) {
            curves.resize(B + s);
            tangents.resize(B + s);
            curvesInitialIndices.resize(B + s);
        }
        std::vector<std::size_t> branchIds(s + 1);
        branchIds[0] = b;

        // Create aliases to the branch data
        const auto& curve = curves[b];
        const auto& tangent = tangents[b];
        const auto& indices = curvesInitialIndices[b];

        auto& branchSplits = splits.emplace_back((int)b, Splits{}).second;
        branchSplits.resize(splitsAt.size());
        auto& endNode = branchList[b][1];

        // Create the new nodes if not provided
        if (splitsNode.size() == 0) {
            // Create the new node
            for (const auto& splitAt : splitsAt) {
                splitsNode.push_back(nodesPoint.size());
                nodesPoint.push_back(curve[splitAt]);  // Add the new node at the split position
            }
        }

        // Iterate split in reverse order to prevent unecessary copy
        std::sort(splitsAt.begin(), splitsAt.end(),
                  [](const std::size_t& a, const std::size_t& b) { return a > b; });  // Sort decreasingly
        for (const auto& splitAt : splitsAt) {
            s--;  // Index of the crossing in the list of crossings for this branch
            const auto& splitNode = splitsNode[s];

            // Register the split
            branchSplits[s] = {indices[splitAt], nodesPoint[splitNode].toIntPair()};

            // Create the new branch
            std::size_t b_new = B + s;  // Index of the new branch after the split
            branchIds[s + 1] = b_new;
            branchList[b_new] = {(int)splitNode, endNode};
            endNode = splitNode;

            // Update tips position and tangents
            tipsPos[b_new] = {curve[splitAt + 1], tipsPos[b][1]};
            tipsTangents[b_new] = {-tangent[splitAt + 1], tipsTangents[b][1]};
            tipsPos[b][1] = curve[splitAt];
            tipsTangents[b][1] = tangent[splitAt];

            // Update curves
            if (updateCurves) {
                // Create the new branch
                curves[b_new] = CurveYX(curve.begin() + splitAt, curve.end());
                tangents[b_new] = PointList(tangent.begin() + splitAt, tangent.end());
                auto& indices_b_new = curvesInitialIndices[b_new];
                indices_b_new = std::vector<int>(indices.begin() + splitAt, indices.end());
                const auto id0 = indices_b_new.front();  // Rebase the indices of the new branch to start at 0
                for (auto& id : indices_b_new) id -= id0;

                // Update the original branch
                curves[b].resize(splitAt + 1);
                tangents[b].resize(splitAt + 1);
                curvesInitialIndices[b].resize(splitAt + 1);
            }
        }
        return branchIds;
    };

    // === SEARCH CURVES CROSSINGS ===
    struct CrossingCandidate {
        int b;                           // Branch index
        std::size_t i = 0;               // Index in the curve
        float l = -1;                    // Length along the curve
        std::size_t crossingNodeID = 0;  // Index of the crossing
    };
    std::map<IntPoint, std::list<CrossingCandidate>> crossingsCandidates;
    auto mapTensor = torch::zeros({shape[0], shape[1]}, torch::kInt);
    auto map = mapTensor.accessor<int, 2>();
    std::vector<float> curvesL(B);

    // --- Draw the curves on a map and record the crossings ---
    for (std::size_t b = 0; b < B; b++) {
        const auto& curve = curves[b];
        float l = 0;
        for (std::size_t i = 0; i < curve.size(); i++) {
            const auto& p = curve[i];
            if (i > 0) l += p.distance(curve[i - 1]);
            if (!p.is_inside(shape[0], shape[1])) continue;

            auto& m = map[p.y][p.x];
            if (m > 0 && m != b + 1) {
                // If the point is already on a curve, record the crossing
                auto [it, inserted] = crossingsCandidates.try_emplace(p, std::list<CrossingCandidate>{{m - 1}});
                it->second.emplace_back(CrossingCandidate{(int)b, i, l});
            } else
                m = b + 1;
        }
        curvesL[b] = l;
    }
    // For each crossing, the inital curve indices and distance is not stored and must be re-computed
    for (auto& [pos, crossings] : crossingsCandidates) {
        auto& crossing0 = crossings.front();
        const auto& curve0 = curves[crossing0.b];
        crossing0.l = 0;
        for (std::size_t i = 0; i < curve0.size(); i++) {
            const auto& p = curve0[i];
            if (i > 0) crossing0.l += p.distance(curve0[i - 1]);
            if (p == pos) {
                crossing0.i = i;
                break;
            }
        }
    }

    // --- Reshape the crossings into a more convenient structure ---
    struct CrossingLocation {
        std::size_t b;  // Branch index
        std::size_t i;  // Index in the curve
        float l;        // Length along the curve
    };
    struct Crossing {
        IntPoint pos;                            // Position of the crossing
        std::vector<CrossingLocation> branches;  // List of branches crossing at this position
        int crossingNodeID = -1;                 // Index of the crossing
    };
    std::map<IntPoint, Crossing> crossingsByPos;
    for (const auto& [pos, crossings] : crossingsCandidates) {
        if (crossings.size() < 2) continue;  // Only keep positions with at least 2 crossings
        std::vector<CrossingLocation> branches;
        for (const auto& crossing : crossings)
            branches.push_back(CrossingLocation{(std::size_t)crossing.b, crossing.i, crossing.l});
        std::sort(branches.begin(), branches.end(),
                  [](const CrossingLocation& a, const CrossingLocation& b) { return a.b < b.b; });
        crossingsByPos.emplace(pos, Crossing{pos, std::move(branches)});
    }

    // --- Cluster equivalent crossing close to each other ---
    {
        std::map<std::vector<std::size_t>, std::vector<Crossing>> equivalentCrossings;
        for (auto& [pos, crossing] : crossingsByPos) {
            std::vector<std::size_t> branchIDs;
            for (const auto& info : crossing.branches) branchIDs.push_back(info.b);
            std::sort(branchIDs.begin(), branchIDs.end());
            auto [it, inserted] = equivalentCrossings.try_emplace(branchIDs);
            it->second.push_back(crossing);
        }
        crossingsByPos.clear();

        for (auto& [branchIDs, crossings] : equivalentCrossings) {
            if (crossings.size() < 2) {
                if (crossings.size() == 1) crossingsByPos.emplace(crossings.front().pos, crossings.front());
                continue;
            }

            std::vector<Crossing> validCrossings;

            // Sort crossing by their position along the first branch
            std::sort(crossings.begin(), crossings.end(),
                      [](const Crossing& a, const Crossing& b) { return a.branches[0].l < b.branches[0].l; });
            // Keep only one crossing among those that are too close to each other
            auto from = crossings.begin(), currentCross = crossings.begin(), nextCross = std::next(currentCross);
            auto keepOneCrossing = [&](std::vector<Crossing>::iterator to) {
                if (from == to) {
                    validCrossings.push_back(*from);
                    return;
                }
                // Keep the crossing that is closest to the middle of the cluster
                float midL = (from->branches[0].l + to->branches[0].l) / 2;
                auto best = std::make_pair(std::abs(from->branches[0].l - midL), from);

                for (auto it = std::next(from); it != to; ++it) {
                    float dist = std::abs(it->branches[0].l - midL);
                    if (dist < best.first) best = std::make_pair(dist, it);
                }
                validCrossings.push_back(*best.second);
            };
            while (nextCross != crossings.end()) {
                if (nextCross->branches[0].l - currentCross->branches[0].l > minSpaceBetweenSplits) {
                    keepOneCrossing(currentCross);
                    from = nextCross;
                }
                currentCross = nextCross;
                nextCross++;
            }
            keepOneCrossing(currentCross);

            for (const auto& crossing : validCrossings) crossingsByPos.emplace(crossing.pos, crossing);
        }
    }

    // --- Group crossings by branch and filter out those too close to the tips ---
    std::size_t nCrossings = 0;
    struct CrossingSplit {
        std::size_t b;       // Branch index
        std::size_t i;       // Index in the curve
        std::size_t nodeID;  // Index of the crossing node
    };

    std::map<std::size_t, std::vector<CrossingSplit>> crossingByBranch;
    for (const auto& [pos, crossing] : crossingsByPos) {
        bool crossingKept = false;
        for (const auto& branch : crossing.branches) {
            if (std::min(branch.l, curvesL[branch.b] - branch.l) > snapDist) {
                auto [it, inserted] = crossingByBranch.try_emplace(branch.b);
                it->second.emplace_back(CrossingSplit{branch.b, branch.i, nodesPoint.size()});
                crossingKept = true;
                nCrossings++;
            }
        }
        if (crossingKept) nodesPoint.push_back(pos);  // Add the crossing node to the list of nodes
    }

    // --- Split branches at crossings ---
    if (nCrossings > 0) {
        for (auto& [b, crossings] : crossingByBranch) {
            std::vector<std::size_t> splitsAt, splitsNode;
            for (const auto& crossing : crossings) {
                splitsAt.push_back(crossing.i);
                splitsNode.push_back(crossing.nodeID);
            }
            split_branch(b, splitsAt, splitsNode, true);  // Split branch b at the given crossings
        }
    }
    B = curves.size();  // Update the number of branches after splitting

    // === SCAN FOR INTERCEPT POINTS ===
    std::list<InterceptCandidate> intercepts;
    std::list<ConnexionCandidate> connexionsCandidates;
    float maxDistSqr = maxDist * maxDist;
    float nearConeCos = std::cos(deg2rad(nearConeAngle)), farConeCos = std::cos(deg2rad(farConeAngle));
    float minTanCos = std::cos(deg2rad(maxTanAngle));
    float minHypCos = std::cos(deg2rad(maxHypAngle));

#pragma omp parallel for collapse(2) reduction(merge : intercepts, connexionsCandidates)
    for (std::size_t b0 = 0; b0 < tipsPos.size(); b0++) {
        for (std::size_t tip0 = 0; tip0 < 2; tip0++) {
            const auto& p = tipsPos[b0][tip0];
            const auto& dir = tipsTangents[b0][tip0];

            // Enumerate intercepts of the current tip with every other branch
            for (std::size_t b1 = 0; b1 < curves.size(); b1++) {
                if (b1 == b0) continue;  // Skip self-interception
                const auto& b1Curve = curves[b1];
                // Find intercepts points
                auto [_intercepts, totalL] =
                    _cone_curve_intercept(b1Curve, tangents[b1], p, dir, maxDistSqr, nearConeCos, farConeCos, minTanCos,
                                          minHypCos, minSpaceBetweenSplits);
                if (_intercepts.empty()) continue;

                // Snap to the start tip if within the snapping distance
                float snapDist_ = std::min(snapDist, totalL / 2.0f);
                bool snap = false;
                while (!_intercepts.empty() && _intercepts.front().l <= snapDist_) {
                    snap |= _intercepts.front().towardsAfter;
                    _intercepts.pop_front();  // Discard the intercept point to avoid splitting the branch ...
                }
                // ... but save the connexion candidate to the start tip of the branch
                if (snap) connexionsCandidates.emplace_back(ConnexionCandidate{b0, (int)tip0, b1, 0});

                // Snap to the end tip if within the snapping distance
                snap = false;
                while (!_intercepts.empty() && totalL - _intercepts.back().l <= snapDist_) {
                    snap |= _intercepts.front().towardsBefore;
                    _intercepts.pop_back();
                }
                // Save the connexion candidate to the end tip of the branch
                if (snap) connexionsCandidates.emplace_back(ConnexionCandidate{b0, (int)tip0, b1, 1});

                // Save the remaining intercepts
                for (auto& intercept : _intercepts) {
                    intercept.b0 = b0;
                    intercept.tip0 = tip0;
                    intercept.b1 = b1;
                    intercepts.push_back(intercept);
                }
            }
        }
    }

    // === PROCESS INTERCEPT CANDIDATES ===
    std::vector<std::list<InterceptCandidate>> interceptsByBranch(B);
    std::vector<std::size_t> branchIdAtTip1;
    branchIdAtTip1.reserve(B);
    for (std::size_t b = 0; b < B; b++) branchIdAtTip1.push_back(b);
    for (const auto& intercept : intercepts) interceptsByBranch[intercept.b1].push_back(intercept);

    std::list<ConnexionCandidate> splitIncidentConnexions;

    for (std::size_t b1 = 0; b1 < interceptsByBranch.size(); b1++) {
        auto& intercepts = interceptsByBranch[b1];
        if (intercepts.empty()) continue;
        if (intercepts.size() == 1) {
            const auto& intercept = intercepts.front();
            auto b0 = intercept.b0;
            auto tip0 = intercept.tip0;

            auto newB = split_branch(b1, {intercept.i}, {}, false)[1];  // Split branch b1 at the intercept
            branchIdAtTip1[b1] = newB;  // Update the branch ID at tip 1 to the new branch ID

            if (intercept.towardsBefore) splitIncidentConnexions.emplace_back(ConnexionCandidate{b0, tip0, b1, 1});
            if (intercept.towardsAfter) splitIncidentConnexions.emplace_back(ConnexionCandidate{b0, tip0, newB, 0});
        } else {
            // TRY TO CLUSTER INTERCEPTS CANDIDATES
            // Sort the intercepts (Normally they are already sorted but just in case...)
            intercepts.sort([](const auto& a, const auto& b) { return a.i < b.i; });

            // Iteratively merge the intercepts so they are separated by at least minSpaceBetweenSplits
            struct Cluster {
                std::list<InterceptCandidate> intercepts;
                float avgL, weight;
            };
            std::list<Cluster> clusters;
            // List intercepts and weight them by 1/score to favor the closest intercepts (lowest score)
            for (auto it = intercepts.begin(); it != intercepts.end(); it++)
                clusters.emplace_back(Cluster{{*it}, it->l, 1.0f / (it->score + 1e-2f)});

            while (clusters.size() > 1) {
                float minSpace = std::numeric_limits<float>::max();
                auto c0 = clusters.end();
                for (auto it = clusters.begin(); it != std::prev(clusters.end()); it++) {
                    float space = std::next(it)->avgL - it->avgL;
                    if (space < minSpace) {
                        minSpace = space;
                        c0 = it;
                        if (space == 0) break;
                    }
                }
                if (minSpace < minSpaceBetweenSplits) {
                    auto c1 = std::next(c0);
                    float w0 = c0->weight, w1 = c1->weight;
                    c0->intercepts.splice(c0->intercepts.end(), c1->intercepts);
                    c0->avgL = (c0->avgL * w0 + c1->avgL * w1) / (w0 + w1);
                    c0->weight = w0 + w1;
                    clusters.erase(c1);
                } else
                    break;
            }

            // Compute clusters center (the index of the split)
            std::vector<std::size_t> splitsAt;
            splitsAt.reserve(clusters.size());
            for (const auto& cluster : clusters) {
                float weightedI = 0, weight = 0;
                for (const auto& intercept : cluster.intercepts) {
                    float w = 1.0f / (intercept.score + 1e-2f);
                    weightedI += intercept.i * w;
                    weight += w;
                }
                splitsAt.push_back((std::size_t)std::round(weightedI / weight));
            }

            // Split the branch
            const auto& newBranches = split_branch(b1, splitsAt, {}, true);

            // Register connexions candidates between the intercepting branches and the new branches
            auto b_it = newBranches.begin();
            for (const auto& cluster : clusters) {
                std::size_t bBefore = *b_it, bAfter = *(++b_it);
                for (const auto& intercept : cluster.intercepts) {
                    auto b0 = intercept.b0;
                    auto tip0 = intercept.tip0;
                    if (intercept.towardsBefore)
                        splitIncidentConnexions.emplace_back(ConnexionCandidate{b0, tip0, bBefore, 1});
                    if (intercept.towardsAfter)
                        splitIncidentConnexions.emplace_back(ConnexionCandidate{b0, tip0, bAfter, 0});
                }
            }
            branchIdAtTip1[b1] = newBranches.back();  // Update the branch ID at tip 1 to the new branch ID
        }
    }

    // Update graph with all splits (crossing & intercepts)
    graph = edge_list_to_adjlist(branchList, nodesPoint.size());

    // Update the branch ID to account for the split in the tip connections candidates
    for (auto& c : connexionsCandidates) {
        if (c.tip0 == 1 && c.b0 < B) c.b0 = branchIdAtTip1[c.b0];
        if (c.tip1 == 1 && c.b1 < B) c.b1 = branchIdAtTip1[c.b1];
    }
    for (auto& c : splitIncidentConnexions) {
        if (c.tip0 == 1 && c.b0 < B) c.b0 = branchIdAtTip1[c.b0];
    }
    connexionsCandidates.splice(connexionsCandidates.end(), splitIncidentConnexions);

    // === CREATE CONNEXION TENSOR ===

    // Create a list of all the branch tips adjacent to each node and add the connexions between them
    for (int node = 0; node < (int)graph.size(); node++) {
        const auto& edges = graph[node];
        std::vector<SizePair> adjBranchTips;
        for (const auto& edge : edges) {
            std::size_t b0 = edge.id, t0 = edge.is_first(node) ? 0 : 1;
            for (const auto& [b1, t1] : adjBranchTips)
                connexionsCandidates.emplace_back(ConnexionCandidate{b0, (int)t0, b1, (int)t1});
            adjBranchTips.emplace_back(SizePair{b0, t0});
        }
    }

    // Propagate the connexions of small "through" branches to their adjacent branches
    std::list<ConnexionCandidate> propagatedConnexions;
    for (const auto& connexion : connexionsCandidates) {
        for (const auto& c : {connexion, symmetric_connexion(connexion)}) {
            if (c.b0 >= curvesInitialIndices.size() ||
                (curvesInitialIndices[c.b0].size() > 0 && curvesInitialIndices[c.b0].back() >= 5))
                continue;

            const std::size_t& oppositeNodeId = branchList[c.b0][1 - c.tip0];
            for (const auto& edge : graph[oppositeNodeId]) {
                std::size_t b = edge.id;
                if (b == c.b0) continue;
                int t = branchList[b][0] == oppositeNodeId ? 0 : 1;
                propagatedConnexions.emplace_back(ConnexionCandidate{b, t, c.b1, c.tip1});
            }
        }
    }
    connexionsCandidates.splice(connexionsCandidates.end(), propagatedConnexions);

    // Search for mergeable nodes and add the connexions between their adjacent branch tips
    if (mergeNodeDist > 0) {
        for (int n0 = 0; n0 < (int)nodesPoint.size(); n0++) {
            for (int n1 = n0 + 1; n1 < (int)nodesPoint.size(); n1++) {
                if (nodesPoint[n0].distance(nodesPoint[n1]) <= mergeNodeDist) {
                    bool alreadyConnected = false;
                    for (const auto& edge : graph[n0]) {
                        if ((edge.start == n0 && edge.end == n1) || edge.start == n1) {
                            alreadyConnected = true;
                            break;
                        }
                    }
                    if (alreadyConnected) continue;

                    // If not connected, add the connexions between their adjacent branch tips
                    std::vector<SizePair> n0BranchTips;
                    for (const auto& e0 : graph[n0])
                        n0BranchTips.emplace_back(SizePair{(std::size_t)e0.id, (std::size_t)(e0.is_first(n0) ? 0 : 1)});
                    std::vector<SizePair> n1BranchTips;
                    for (const auto& e1 : graph[n1])
                        n1BranchTips.emplace_back(SizePair{(std::size_t)e1.id, (std::size_t)(e1.is_first(n1) ? 0 : 1)});
                    for (const auto& [b0, t0] : n0BranchTips) {
                        for (const auto& [b1, t1] : n1BranchTips) {
                            if (tipsTangents[b0][t0].dot(tipsTangents[b1][t1]) >= minTanCos)
                                connexionsCandidates.emplace_back(ConnexionCandidate{b0, (int)t0, b1, (int)t1});
                        }
                    }
                }
            }
        }
    }

    // Ensure all connexions are symmetric
    std::list<ConnexionCandidate> symmetricConnexions;
    for (const auto& c : connexionsCandidates) symmetricConnexions.emplace_back(symmetric_connexion(c));
    connexionsCandidates.splice(connexionsCandidates.end(), symmetricConnexions);

    // Remove duplicate connexions
    connexionsCandidates.sort([](const auto& a, const auto& b) {
        if (a.b0 != b.b0) return a.b0 < b.b0;
        if (a.tip0 != b.tip0) return a.tip0 < b.tip0;
        if (a.b1 != b.b1) return a.b1 < b.b1;
        return a.tip1 < b.tip1;
    });
    connexionsCandidates.unique([](const auto& a, const auto& b) {
        return a.b0 == b.b0 && a.tip0 == b.tip0 && a.b1 == b.b1 && a.tip1 == b.tip1;
    });

    torch::Tensor connexions = torch::zeros({(long)connexionsCandidates.size(), 4}, torch::kLong);
    auto connexions_acc = connexions.accessor<long, 2>();
    std::size_t i = 0;
    for (const auto& c : connexionsCandidates) {
        connexions_acc[i][0] = (long)c.b0;
        connexions_acc[i][1] = (long)c.tip0;
        connexions_acc[i][2] = (long)c.b1;
        connexions_acc[i][3] = (long)c.tip1;
        i++;
    }

    return {splits, connexions};
}
