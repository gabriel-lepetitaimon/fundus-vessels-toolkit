#include "rasterize_topo.h"

#include "bezier.h"
#include "branch.h"
#include "ray_iterators.h"

void rasterize_topology(const torch::Tensor& branch_list, const torch::Tensor& root_nodes,
                        const std::vector<torch::Tensor>& curves, const std::vector<torch::Tensor>& boundaries,
                        int N_nodes, float bspline_interpolate, bool fill_junctions, torch::Tensor& branchLabelsMap,
                        torch::Tensor& topoMap) {
    // Ensure the branchLabelsMap and topoMap are initialized correctly
    TORCH_CHECK(branchLabelsMap.dim() == 2 && topoMap.dim() == 2, "branchLabelsMap and topoMap must be 2D tensors.");
    TORCH_CHECK(branchLabelsMap.size(0) == topoMap.size(0) && branchLabelsMap.size(1) == topoMap.size(1),
                "branchLabelsMap and topoMap must have the same shape.");
    TORCH_CHECK(branchLabelsMap.scalar_type() == torch::kInt32 && topoMap.scalar_type() == torch::kFloat32,
                "branchLabelsMap and topoMap must be of type Int32.");

    auto branchLabelsMapAcc = branchLabelsMap.accessor<int, 2>();
    auto topoMapAcc = topoMap.accessor<float, 2>();
    IntPoint maxShape = {(int)branchLabelsMap.size(0), (int)branchLabelsMap.size(1)};

    // Initialize the adjacency list for the topology
    const auto& branchListAcc = branch_list.accessor<int, 2>();
    std::size_t N_branches = branchListAcc.size(0);
    GraphAdjList adjList = edge_list_to_adjlist(branchListAcc, N_nodes);

    // Traverse the graph starting from root branches
    std::vector<int> branchRanks(N_branches);
    std::vector<int> branchDirs(N_branches, 0);
    auto rootNodesAcc = root_nodes.accessor<int, 1>();
    std::stack<int> q;
    for (std::size_t i = 0; i < (std::size_t)rootNodesAcc.size(0); i++) {
        const int& rootNode = rootNodesAcc[i];
        for (const auto& branch : adjList[rootNode]) {
            if (branch.id == -1) continue;
            q.push(branch.id);
            branchRanks[branch.id] = 1;                                 // Initialize rank for the branch
            branchDirs[branch.id] = branch.start == rootNode ? 1 : -1;  // Determine direction based on edge start
        }
    }

    while (!q.empty()) {
        // Read branch info
        int branchID = q.top();
        q.pop();
        int rank = branchRanks[branchID];
        bool reversed = branchDirs[branchID] == -1;
        int headNode = branchListAcc[branchID][reversed ? 0 : 1];

        std::list<std::tuple<int, bool>> nextBranches;

        // Iterate over the neighbors of the current branch
        for (const auto& nextBranch : adjList[headNode]) {
            // Skip the current branch or already visited branches
            if (nextBranch.id == branchID || branchDirs[nextBranch.id] != 0) continue;

            // Assign the neighbor branch rank and direction, and enqueue it
            branchRanks[nextBranch.id] = rank + 1;
            bool nextBranchReversed = nextBranch.start != headNode;
            branchDirs[nextBranch.id] = nextBranchReversed ? -1 : 1;
            q.push(nextBranch.id);
            nextBranches.push_back({nextBranch.id, nextBranchReversed});
        }

        // Get the corresponding curve
        const auto& curve = curves[branchID];
        if (curve.size(0) == 0) continue;

        // Get the corresponding boundaries
        const auto& boundary = boundaries[branchID].accessor<int, 3>();
        std::array<IntPoint, 2> headBounds;
        if (!reversed) {
            const auto last = boundary.size(0) - 1;
            headBounds = {IntPoint(boundary[last][0]), IntPoint(boundary[last][1])};
        } else {
            headBounds = {IntPoint(boundary[0][1]), IntPoint(boundary[0][0])};
        }

        // Rasterize the branch
        _rasterize_branch_topo(curve, boundary, branchID + 1, rank, branchLabelsMapAcc, topoMapAcc, bspline_interpolate,
                               reversed);

        // Fill junctions
        if (!fill_junctions) continue;
        for (const auto& [nextBranchID, nextBranchReversed] : nextBranches) {
            auto nextBoundariesAcc = boundaries[nextBranchID].accessor<int, 3>();
            if (nextBoundariesAcc.size(0) == 0) continue;  // If the boundaries are empty, skip this filling

            std::array<IntPoint, 2> nextBounds;
            if (!nextBranchReversed) {
                nextBounds = {IntPoint(nextBoundariesAcc[0][0]), IntPoint(nextBoundariesAcc[0][1])};
            } else {
                const auto last = nextBoundariesAcc.size(0) - 1;
                nextBounds = {IntPoint(nextBoundariesAcc[last][1]), IntPoint(nextBoundariesAcc[last][0])};
            }
            continue;

            // Draw the quad for the junction
            auto it = QuadIterator(headBounds[0], headBounds[1], nextBounds[1], nextBounds[0], maxShape);
            it.precomputeInvDiffNorms();
            while (it.iter()) {
                IntPoint p = it.point();
                branchLabelsMapAcc[p.y][p.x] = branchID + 1;  // Use branchID + 1 to avoid zero
                float topoValue = rank + 0.9 + 0.1 * it.fromP12toP34();
                if (topoMapAcc[p.y][p.x] < topoValue) topoMapAcc[p.y][p.x] = topoValue;
            }
        }
    }
}

void rasterize_branch_topo(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                           torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bspline_interpolate) {
    // Ensure the branchLabelsMap and topoMap are initialized correctly
    TORCH_CHECK(branchLabelsMap.dim() == 2 && topoMap.dim() == 2, "branchLabelsMap and topoMap must be 2D tensors.");
    TORCH_CHECK(branchLabelsMap.size(0) == topoMap.size(0) && branchLabelsMap.size(1) == topoMap.size(1),
                "branchLabelsMap and topoMap must have the same shape.");
    TORCH_CHECK(branchLabelsMap.scalar_type() == torch::kInt32 && topoMap.scalar_type() == torch::kFloat32,
                "branchLabelsMap must be of type Int32 and topoMap must be of type Float32.");

    // Ensure the curve and boundaries tensors are of the correct shape
    TORCH_CHECK(curve.dim() == 2 && boundaries.dim() == 3,
                "curve must be a 2D tensor and boundaries must be a 3D tensor.");
    TORCH_CHECK(curve.size(1) == 2 && boundaries.size(1) == 2 && boundaries.size(2) == 2,
                "curve must have shape [N, 2] and boundaries must have shape [N, 2, 2].");
    TORCH_CHECK(curve.size(0) == boundaries.size(0), "curve and boundaries must have the same first dimension size.");

    return _rasterize_branch_topo(curve, boundaries.accessor<int, 3>(), branchID, branchRank,
                                  branchLabelsMap.accessor<int, 2>(), topoMap.accessor<float, 2>(),
                                  bspline_interpolate);
}

void _rasterize_bezier(const IntPoint& p0, const IntPoint& p1, const IntPointPair& b0, const IntPointPair& b1,
                       const Point& t0, const Point& t1, const IntPoint& maxShape, float bezier_smoothness,
                       std::function<void(IntPoint, float)> updater) {
    const float w0 = distance(b0[0], b0[1]), w1 = distance(b1[0], b1[1]);

    // == Discretize Bezier ==
    bezier_smoothness *= distance(Point(p0), Point(p1));

    const BezierCubic bezier = {Point(p0), Point(p0) + t0 * bezier_smoothness, Point(p1) - t1 * bezier_smoothness,
                                Point(p1)};
    auto [interpPoints, us] = discretizeBezier(bezier);
    auto tangents = evaluate_bezier_tangent(bezier, us);
    auto N = interpPoints.size();

    double u = 0.0f, nextU;
    IntPoint p = interpPoints[0].toInt(), nextP;
    Point t = t0, nextT;
    float w = w0, nextW;
    IntPointPair b = b0, nextB;

    for (std::size_t i = 0; i < N - 1; i++) {
        if (i != N - 2) {
            nextU = us[i + 1];
            nextP = interpPoints[i + 1].toInt();
            nextT = tangents[i + 1].normalize();
            nextW = lerp(w0, w1, nextU);
            auto dB = (nextT.rot90() * (nextW / 2.0)).toInt();
            nextB = {nextP + dB, nextP - dB};
        } else {
            nextP = p1;
            nextT = t1;
            nextB = b1;
        }
        updater(p, u);
        auto externalError = t.angle(nextT) * w / 2.0;
        for (int lr = 0; lr < 2; ++lr) {  // Iterate over left and right quads
            /// TODO: if t.dot(nextT)*W/2 is large, subdivide the exterior quad further
            int lr_sign = 1 - lr * 2;  // +1 for left, -1 for right
            if (externalError * lr_sign < 0 && std::ceil(std::abs(externalError)) > 1) {
                // Subdivide exterior perimeter
                int N_splits = std::ceil(std::abs(externalError));
                IntPoint prev_b = b[lr], b;
                float ds = 1.0f / N_splits;
                for (float s = ds; s < 1; s += ds) {
                    double s_u = lerp(u, nextU, s);
                    auto s_t = lerp(t, nextT, s).normalize();
                    auto s_w = lerp(w, nextW, s);
                    b = (evaluate_bezier(bezier, s_u) + s_t.rot90() * (s_w * lr_sign * 0.5)).toInt();
                    QuadIterator it(p, prev_b, b, nextP, maxShape);
                    it.precomputeInvDiffNorms();
                    while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP1toP4()));
                    prev_b = b;
                }
                QuadIterator it(p, prev_b, nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP1toP4()));
            } else {
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) updater(it.point(), lerp(u, nextU, it.fromP12toP34()));
            }
        }

        // Move to the next point
        p = nextP;
        t = nextT;
        w = nextW;
        b = nextB;
        u = nextU;
    }
}

void _rasterize_branch_topo(const torch::Tensor& curve_tensor, const Tensor3DAcc<int>& boundaries, int branchID,
                            float rank, Tensor2DAcc<int> branchLabelsMap, Tensor2DAcc<float> topoMap,
                            float bspline_interpolate, bool reverse) {
    const IntPoint maxShape = {(int)branchLabelsMap.size(0), (int)branchLabelsMap.size(1)};
    const auto curve = tensor_to_curve(curve_tensor);
    int N = (int)curve.size();

    long first = reverse ? N - 1 : 0, last = reverse ? 0 : N - 1;
    long d_it = reverse ? -1 : 1;
    IntPoint p = curve[first], nextP;
    IntPointPair b = {boundaries[first][0], boundaries[first][1]}, nextB;

    for (int i = first; i != last; i += d_it) {
        auto nextI = i + d_it;
        IntPoint nextP = curve[nextI];
        IntPointPair nextB = {boundaries[nextI][0], boundaries[nextI][1]};

        auto drawTopo = [&](IntPoint pt, float u) {
            branchLabelsMap[pt.y][pt.x] = branchID;
            float topoValue = rank + 0.9 * (u + i) / N;
            if (topoMap[pt.y][pt.x] < topoValue) topoMap[pt.y][pt.x] = topoValue;
        };

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= 9) {
            for (int lr = 0; lr < 2; ++lr) {
                // Draw the center point
                drawTopo(p, 0.0f);

                // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                it.precomputeInvDiffNorms();
                while (it.iter()) drawTopo(it.point(), it.fromP12toP34());
            }
        } else if (bspline_interpolate > 0.0f) {
            // Rasterize a bezier curve between p and nextP
            Point t = adaptative_curve_tangent(curve, i, distance(b[0], b[1]), reverse, !reverse),
                  nextT = adaptative_curve_tangent(curve, nextI, distance(nextB[0], nextB[1]), !reverse, reverse);
            if (!reverse)
                _rasterize_bezier(p, nextP, b, nextB, t, nextT, maxShape, bspline_interpolate, drawTopo);
            else
                _rasterize_bezier(p, nextP, {b[1], b[0]}, {nextB[1], nextB[0]}, -t, -nextT, maxShape,
                                  bspline_interpolate, drawTopo);
        }

        // Move to the next point
        p = nextP;
        b = nextB;
    }
}

torch::Tensor& rasterize_branch(const torch::Tensor& curveTensor, const torch::Tensor& boundariesTensor,
                                torch::Tensor& outTensor, int fill_value, float bspline_interpolate) {
    // Check input shapes
    TORCH_CHECK(curveTensor.dim() == 2 && curveTensor.size(1) == 2, "curve must have shape [N, 2]");
    TORCH_CHECK(boundariesTensor.dim() == 3 && boundariesTensor.size(1) == 2 && boundariesTensor.size(2) == 2,
                "boundaries must have shape [N, 2, 2]");
    TORCH_CHECK(curveTensor.size(0) == boundariesTensor.size(0),
                "curve and boundaries must have the same first dimension size.");

    // Check output tensor
    TORCH_CHECK(outTensor.dim() == 2, "out must be a 2D tensor");
    TORCH_CHECK(outTensor.dtype() == torch::kInt, "out must be an integer tensor");

    // Prepare accessors and constants
    const auto curve = tensor_to_curve(curveTensor);
    auto boundaries = boundariesTensor.accessor<int, 3>();
    auto out = outTensor.accessor<int, 2>();

    const IntPoint maxShape = {(int)out.size(0), (int)out.size(1)};
    int N = (int)curve.size();

    IntPoint p = curve[0], nextP;
    IntPointPair b = {boundaries[0][0], boundaries[0][1]}, nextB;

    for (int i = 0; i < N - 1; i++) {
        int nextI = i + 1;
        IntPoint nextP = curve[nextI];
        IntPointPair nextB = {boundaries[nextI][0], boundaries[nextI][1]};

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= 9) {
            out[p.y][p.x] = fill_value;       // Draw the center point
            for (int lr = 0; lr < 2; ++lr) {  // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                while (it.iter()) out[it.point().y][it.point().x] = fill_value;
            }
        } else if (bspline_interpolate > 0.0f) {
            // Rasterize a bezier curve between p and nextP
            Point t = adaptative_curve_tangent(curve, i, distance(b[0], b[1]), false, true),
                  nextT = adaptative_curve_tangent(curve, nextI, distance(nextB[0], nextB[1]), true, false);

            _rasterize_bezier(p, nextP, b, nextB, t, nextT, maxShape, bspline_interpolate,
                              [&](IntPoint pt, float u) { out[pt.y][pt.x] = fill_value; });
        }

        // Move to the next point
        p = nextP;
        b = nextB;
    }

    return outTensor;
}

/**********************************************************************************************************************
 *     === QUAD RASTERIZATION ===
 *********************************************************************************************************************/
bool isPointInQuad(const std::array<int, 4>& d) {
    // Check if the point is inside the quad defined by the cross products
    return (d[0] >= 0 && d[1] >= 0 && d[2] >= 0 && d[3] >= 0) || (d[0] <= 0 && d[1] <= 0 && d[2] <= 0 && d[3] <= 0);
}

IntPoint minBoundingPoint(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                          const IntPoint& maxPoint) {
    // Calculate the minimum bounding point of the quad defined by points p1, p2, p3, and p4
    return IntPoint(std::max({std::min({p1.y, p2.y, p3.y, p4.y, maxPoint.y - 1}), 0}),
                    std::max({std::min({p1.x, p2.x, p3.x, p4.x, maxPoint.x - 1}), 0}));
}

IntPoint maxBoundingPoint(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                          const IntPoint& maxPoint) {
    // Calculate the minimum bounding point of the quad defined by points p1, p2, p3, and p4
    return IntPoint(std::min({std::max({p1.y, p2.y, p3.y, p4.y, 0}), maxPoint.y - 1}),
                    std::min({std::max({p1.x, p2.x, p3.x, p4.x, 0}), maxPoint.x - 1}));
}

QuadIterator::QuadIterator(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                           const IntPoint& maxShape)
    : p1(p1),
      p2(p2),
      p3(p3),
      p4(p4),
      pMin(minBoundingPoint(p1, p2, p3, p4, maxShape)),
      pMax(maxBoundingPoint(p1, p2, p3, p4, maxShape)),
      pDiff{{p2 - p1, p2 - p3, p4 - p3, p4 - p1}},
      p(pMin.y, pMin.x - 1),
      _crossProd{0, 0, 0, 0},
      hourGlassQuad(pDiff[0].dot(pDiff[2]) > 0) {}

bool QuadIterator::finished() const {
    // Check if the iterator has finished iterating over the quad
    return p.y >= pMax.y && p.x >= pMax.x;
}

bool QuadIterator::iter() {
    while (true) {
        // === Move to the next point ===
        if (p.x < pMax.x) {
            p.x++;
        } else if (p.y < pMax.y) {
            p.x = pMin.x;  // Reset x to the minimum x value
            p.y++;         // Move to the next row
        } else {
            return false;  // No more points to iterate
        }

        // === Check if the point is inside the quad ===
        // Compute the cross products
        auto pp1 = p - p1, pp3 = p - p3;
        _crossProd[0] = pDiff[0].cross(pp1);                      //  (p2-p1) x (p-p1)
        _crossProd[1] = -pDiff[1].cross(pp3);                     // -(p2-p3) x (p-p3)
        _crossProd[3] = -pDiff[3].cross(pp1);                     // -(p4-p1) x (p-p1)
        if (!hourGlassQuad) _crossProd[2] = pDiff[2].cross(pp3);  //  (p4-p3) x (p-p3)

        // Check if the point is inside the quad using the cross products
        if (isPointInQuad(_crossProd)) return true;
    }
}

const IntPoint& QuadIterator::point() const { return p; }
const std::array<int, 4>& QuadIterator::crossProd() const { return _crossProd; }

double QuadIterator::fromP12toP34() const {
    double d = abs(_crossProd[0]) * _invDiffNorm[0];
    double D = abs(_crossProd[2]) * _invDiffNorm[3] + d;
    return D > 0 ? d / D : 0;
}

double QuadIterator::fromP1toP4() const {
    IntPoint pp1 = p - p1;
    return clip(pp1.normalize().dot(pDiff[3]) * _invDiffNorm[3], 0.0, 1.0);
}

void QuadIterator::precomputeInvDiffNorms() {
    for (int i = 0; i < 4; ++i) _invDiffNorm[i] = 1.0 / pDiff[i].norm();
}

torch::Tensor drawQuad(const IntPair& p1, const IntPair& p2, const IntPair& p3, const IntPair& p4,
                       const IntPair& maxShape) {
    auto scene = torch::zeros({maxShape[0], maxShape[1]}, torch::kFloat);
    auto sceneAcc = scene.accessor<float, 2>();

    QuadIterator it(p1, p2, p3, p4, maxShape);

    const float invNorm = it.pDiff[0].norm(), nextInvNorm = it.pDiff[3].norm();
    while (it.iter()) {
        IntPoint p = it.point();
        float d = abs(it.crossProd()[0]) * invNorm;
        float sumD = abs(it.crossProd()[2]) * nextInvNorm + d;
        d = sumD > 0 ? d / sumD : 0;  // Avoid division by zero
        sceneAcc[p.y][p.x] = 1 + d;
    }

    return scene;
}