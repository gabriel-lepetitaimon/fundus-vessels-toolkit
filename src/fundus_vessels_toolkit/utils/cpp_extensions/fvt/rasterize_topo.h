#ifndef RASTERIZE_TOPO_H
#define RASTERIZE_TOPO_H

#include "common.h"

/**
 * @brief Rasterize the topology of the branches from the curves and boundaries.
 *
 * @param branch_list The edge list of the branches, where each edge is represented by a pair of node indices.
 * @param curves A vector of tensors representing the curves of the branches.
 * @param boundaries A vector of tensors representing the boundaries of the branches.
 * @param branchLabelsMap The tensor to store the branch labels.
 * @param topoMap The tensor to store the topology map.
 */
void rasterize_topology(const torch::Tensor& branch_list, const torch::Tensor& root_branches,
                        const std::vector<torch::Tensor>& curves, const std::vector<torch::Tensor>& boundaries,
                        int N_nodes, float bspline_interpolate, bool fill_junctions, torch::Tensor& branchLabelsMap,
                        torch::Tensor& topoMap);

torch::Tensor& rasterize_branch(const torch::Tensor& curveTensor, const torch::Tensor& boundariesTensor,
                                torch::Tensor& outTensor, int fill_value = 1, float bspline_interpolate = 0.5);

void rasterize_branch_topo(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                           torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bspline_interpolate = 0.5);

void _rasterize_branch_topo(const torch::Tensor& curve, const Tensor3DAcc<int>& boundaries, int branchID,
                            float branchRank, Tensor2DAcc<int> branchLabelsMap, Tensor2DAcc<float> topoMap,
                            float bspline_interpolate = 0.5, bool reverse = false);

torch::Tensor drawQuad(const IntPair& p1, const IntPair& p2, const IntPair& p3, const IntPair& p4,
                       const IntPair& maxShape);

class QuadIterator {
   public:
    QuadIterator(const IntPoint& p1, const IntPoint& p2, const IntPoint& p3, const IntPoint& p4,
                 const IntPoint& maxPoint);

    bool iter();
    bool finished() const;
    const IntPoint& point() const;
    const std::array<int, 4>& crossProd() const;

    double fromP12toP34() const;
    double fromP1toP4() const;
    void precomputeInvDiffNorms();

    const IntPoint p1, p2, p3, p4, pMin, pMax;
    const std::array<IntPoint, 4> pDiff;  // Differences between points for cross product calculations

   private:
    IntPoint p;
    std::array<int, 4> _crossProd;
    std::array<double, 4> _invDiffNorm;
    bool hourGlassQuad;
};

#endif  // RASTERIZE_TOPO_H