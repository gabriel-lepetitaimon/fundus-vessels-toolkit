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
                        torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, int N_nodes = -1);

void rasterize_branch(const torch::Tensor& curve, const torch::Tensor& boundaries, int branchID, float branchRank,
                      torch::Tensor& branchLabelsMap, torch::Tensor& topoMap, float bridge_gap_smaller_than_sqr = 2);

void _rasterize_branch(const Tensor2DAcc<int>& curve, const Tensor3DAcc<int>& boundaries, int branchID,
                       float branchRank, torch::Tensor& branchLabelsMap, torch::Tensor& topoMap,
                       float bridge_gap_smaller_than_sqr = 2);

#endif  // RASTERIZE_TOPO_H