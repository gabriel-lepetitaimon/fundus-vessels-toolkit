#include "rasterize_topo.h"

#include "ray_iterators.h"

void rasterize_topology(const torch::Tensor& branch_list, const torch::Tensor& root_branches,
                        const std::vector<torch::Tensor>& curves, const std::vector<torch::Tensor>& boundaries,
                        torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, int N_nodes) {
    // Ensure the branchLabelsMap and topoMap are initialized correctly
    TORCH_CHECK(branchLabelsMap.dim() == 2 && topoMap.dim() == 2, "branchLabelsMap and topoMap must be 2D tensors.");
    TORCH_CHECK(branchLabelsMap.size(0) == topoMap.size(0) && branchLabelsMap.size(1) == topoMap.size(1),
                "branchLabelsMap and topoMap must have the same shape.");
    TORCH_CHECK(branchLabelsMap.scalar_type() == torch::kInt32 && topoMap.scalar_type() == torch::kInt32,
                "branchLabelsMap and topoMap must be of type Int32.");

    // Initialize the adjacency list for the topology
    GraphAdjList adjList = edge_list_to_adjlist(branch_list.accessor<int, 2>(), N_nodes, true);
}

void rasterize_branch(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                      torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bridge_gap_smaller_than_sqr) {
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

    return _rasterize_branch(curve.accessor<int, 2>(), boundaries.accessor<int, 3>(), branchID, branchRank,
                             branchLabelsMap, topoMap, bridge_gap_smaller_than_sqr);
}

void _rasterize_branch(const Tensor2DAcc<int>& curve, const Tensor3DAcc<int>& boundaries, int branchID,
                       float branchRank, torch::Tensor& branchLabelsMap, torch::Tensor& topoMap,
                       float bridge_gap_smaller_than_sqr) {
    IntPoint p = curve[0], nextP;
    IntPointPair b = {boundaries[0][0], boundaries[0][1]}, nextB;
    auto N = curve.size(0);
    for (long i = 0; i < N - 1; i++) {
        IntPoint nextP = curve[i + 1];
        IntPointPair nextB = {boundaries[i + 1][0], boundaries[i + 1][1]};

        IntPoint diff = nextP - p;
        if (diff.squaredNorm() > bridge_gap_smaller_than_sqr) continue;

        auto topoValue = [&](const SimpleTriangleIterator& it) { return branchRank + (it.relativeHeight() + i) / N; };

        // Curve point
        branchLabelsMap[p.y][p.x] = branchID;
        topoMap[p.y][p.x] = branchRank + static_cast<float>(i) / N;

        // Left triangle
        SimpleTriangleIterator it(p, b[0], nextB[0]);
        while (it.iter()) {
            IntPoint point = *it;
            branchLabelsMap[point.y][point.x] = branchID;
            topoMap[point.y][point.x] = topoValue(it);
        }
        // Right triangle
        it = SimpleTriangleIterator(p, b[1], nextB[1]);
        while (it.iter()) {
            IntPoint point = *it;
            branchLabelsMap[point.y][point.x] = branchID;
            topoMap[point.y][point.x] = topoValue(it);
        }

        if (abs(diff.x) > 1 || abs(diff.y) > 1) {
            auto invTopoValue = [&](const SimpleTriangleIterator& it) {
                return branchRank + (1 - it.relativeHeight() + i) / N;
            };

            // Opposite Left triangle
            it = SimpleTriangleIterator(nextP, p, nextB[0]);
            while (it.iter()) {
                IntPoint point = *it;
                branchLabelsMap[point.y][point.x] = branchID;
                topoMap[point.y][point.x] = invTopoValue(it);
            }

            // Opposite Right triangle
            it = SimpleTriangleIterator(nextP, p, nextB[1]);
            while (it.iter()) {
                IntPoint point = *it;
                branchLabelsMap[point.y][point.x] = branchID;
                topoMap[point.y][point.x] = invTopoValue(it);
            }
        }

        // Move to the next point
        p = nextP;
        b = nextB;
    }
}