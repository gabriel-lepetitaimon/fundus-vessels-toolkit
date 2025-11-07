#include "rasterize_topo.h"

#include "ray_iterators.h"

void rasterize_topology(const torch::Tensor& branch_list, const torch::Tensor& root_nodes,
                        const std::vector<torch::Tensor>& curves, const std::vector<torch::Tensor>& boundaries,
                        int N_nodes, float bridge_gap_smaller_than, bool fill_junctions, torch::Tensor& branchLabelsMap,
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
    bridge_gap_smaller_than *= bridge_gap_smaller_than;  // Use squared distance for comparison

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
        const auto& curve = curves[branchID].accessor<int, 2>();
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
        float headBoundsNorm = (headBounds[1] - headBounds[0]).norm();

        // Rasterize the branch
        _rasterize_branch_topo(curve, boundary, branchID + 1, rank, branchLabelsMapAcc, topoMapAcc,
                               bridge_gap_smaller_than, reversed);

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
            // if (std::max(distanceSqr(headBounds[1], nextBounds[1]), distanceSqr(headBounds[0], nextBounds[0])) >
            //    bridge_gap_smaller_than)
            //    continue;

            // Draw the quad for the junction
            auto it = QuadIterator(headBounds[0], headBounds[1], nextBounds[1], nextBounds[0], maxShape);

            float nextBoundsNorm = (nextBounds[1] - nextBounds[0]).norm();
            while (it.iter()) {
                IntPoint p = it.point();
                branchLabelsMapAcc[p.y][p.x] = branchID + 1;        // Use branchID + 1 to avoid zero
                float d = abs(it.crossProd()[0]) / headBoundsNorm;  // Normalize by the bounds norm
                float sumD = abs(it.crossProd()[2]) / nextBoundsNorm + d;
                d = sumD > 0 ? d / sumD : 0;  // Avoid division by zero
                float topoValue = rank + 0.9 + 0.1 * d;
                if (topoMapAcc[p.y][p.x] < topoValue) topoMapAcc[p.y][p.x] = topoValue;
            }
        }
    }
}

void rasterize_branch_topo(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                           torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bridge_gap_smaller_than) {
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

    return _rasterize_branch_topo(curve.accessor<int, 2>(), boundaries.accessor<int, 3>(), branchID, branchRank,
                                  branchLabelsMap.accessor<int, 2>(), topoMap.accessor<float, 2>(),
                                  bridge_gap_smaller_than * bridge_gap_smaller_than);
}

void _rasterize_branch_topo(const Tensor2DAcc<int>& curve, const Tensor3DAcc<int>& boundaries, int branchID, float rank,
                            Tensor2DAcc<int> branchLabelsMap, Tensor2DAcc<float> topoMap,
                            float bridge_gap_smaller_than_sqr, bool reverse) {
    const IntPoint maxShape = {(int)branchLabelsMap.size(0), (int)branchLabelsMap.size(1)};
    int N = (int)curve.size(0);

    long first = reverse ? N - 1 : 0;
    IntPoint p = curve[first], nextP;
    IntPointPair b = {boundaries[first][0], boundaries[first][1]}, nextB;

    for (int i = 0; i < N - 1; i++) {
        auto nextPos = reverse ? first - i - 1 : i + 1;
        IntPoint nextP = curve[nextPos];
        IntPointPair nextB = {boundaries[nextPos][0], boundaries[nextPos][1]};

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= bridge_gap_smaller_than_sqr) {
            for (int lr = 0; lr < 2; ++lr) {
                // Draw the center point
                branchLabelsMap[p.y][p.x] = branchID;
                topoMap[p.y][p.x] = rank + static_cast<float>(i) / N;

                // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                const float invNorm = it.pDiff[0].norm(), nextInvNorm = it.pDiff[3].norm();
                while (it.iter()) {
                    IntPoint p = it.point();
                    branchLabelsMap[p.y][p.x] = branchID;
                    float d = abs(it.crossProd()[0]) * invNorm;
                    float sumD = abs(it.crossProd()[2]) * nextInvNorm + d;
                    d = sumD > 0 ? d / sumD : 0;  // Avoid division by zero
                    float topoValue = rank + 0.9 * (d + i) / N;
                    if (topoMap[p.y][p.x] < topoValue) topoMap[p.y][p.x] = topoValue;
                }
            }
        }

        // Move to the next point
        p = nextP;
        b = nextB;
    }
}

torch::Tensor& rasterize_branch(const torch::Tensor& curveTensor, const torch::Tensor& boundariesTensor,
                                torch::Tensor& outTensor, int fill_value, float bridge_gap_smaller_than) {
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
    auto curve = curveTensor.accessor<int, 2>();
    auto boundaries = boundariesTensor.accessor<int, 3>();
    auto out = outTensor.accessor<int, 2>();

    const IntPoint maxShape = {(int)out.size(0), (int)out.size(1)};
    float bridge_gap_smaller_than_sqr = bridge_gap_smaller_than * bridge_gap_smaller_than;
    int N = (int)curve.size(0);

    IntPoint p = curve[0], nextP;
    IntPointPair b = {boundaries[0][0], boundaries[0][1]}, nextB;

    for (int i = 0; i < N - 1; i++) {
        IntPoint nextP = curve[i + 1];
        IntPointPair nextB = {boundaries[i + 1][0], boundaries[i + 1][1]};

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() <= bridge_gap_smaller_than_sqr) {
            for (int lr = 0; lr < 2; ++lr) {
                // Draw the center point
                out[p.y][p.x] = fill_value;

                // Iterate over left and right quads
                QuadIterator it(p, b[lr], nextB[lr], nextP, maxShape);
                while (it.iter()) out[it.point().y][it.point().x] = fill_value;
            }
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