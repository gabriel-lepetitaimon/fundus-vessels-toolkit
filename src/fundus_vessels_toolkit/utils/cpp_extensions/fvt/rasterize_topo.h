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
                        int N_nodes, float bridge_gap_smaller_than, bool fill_junctions, torch::Tensor& branchLabelsMap,
                        torch::Tensor& topoMap);

void rasterize_branch(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                      torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bridge_gap_smaller_than = 2);

void _rasterize_branch(const Tensor2DAcc<int>& curve, const Tensor3DAcc<int>& boundaries, int branchID,
                       float branchRank, Tensor2DAcc<int> branchLabelsMap, Tensor2DAcc<float> topoMap,
                       float bridge_gap_smaller_than_sqr = 2, bool reverse = false);

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

    const IntPoint p1, p2, p3, p4, pMin, pMax;
    const std::array<IntPoint, 4> pDiff;  // Differences between points for cross product calculations

   private:
    IntPoint p;
    std::array<int, 4> _crossProd;
    bool hourGlassQuad;
};

#endif  // RASTERIZE_TOPO_H