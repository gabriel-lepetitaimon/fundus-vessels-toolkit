#ifndef TREE_TOPOLOGY_H
#define TREE_TOPOLOGY_H

#include "common.h"

using TopoLabel = uint64_t;

uint8_t rank(const TopoLabel& label);
int32_t subtree(const TopoLabel& label);
TopoLabel parent(const TopoLabel& label);
uint64_t branching_bit_mask(const uint8_t& rank);
bool is_ancestor(const TopoLabel& ancestor, const TopoLabel& descendant, const bool& strict = false);
bool is_between(const TopoLabel& label, const TopoLabel& start, const TopoLabel& end, const bool& strict = false,
                const float& rank = -1, const float& start_rank = -1, const float& end_rank = -1,
                bool strict_rank = true);

std::array<torch::Tensor, 5> read_branches_topology(const std::vector<torch::Tensor>& branch_curves,
                                                    const IntPair& domain, const torch::Tensor& topo_idxs,
                                                    const torch::Tensor& topo_labels, const torch::Tensor& topo_ranks,
                                                    const torch::Tensor& fuzzy_skeleton, float min_rank_threshold,
                                                    float max_rank_tolerance);

std::tuple<TopoLabel, float, float, std::array<TopoLabel, 2>, std::array<float, 2>> read_branch_topology(
    const Tensor2DAcc<int32_t>& curve, const IntPair& domain, const Tensor2DAcc<uint32_t>& topo_idxs,
    const Tensor1DAcc<TopoLabel>& topo_labels, const Tensor1DAcc<at::Half>& topo_ranks,
    const Tensor1DAcc<at::Half>& fuzzy_skeleton, float min_rank_threshold, float max_rank_tolerance);

TopoLabel most_present_ancestor(const std::vector<int32_t>& curve, const Tensor1DAcc<TopoLabel>& topo_labels);

template <typename T>
T minimum(const Tensor1DAcc<T>& values, const std::vector<int32_t>& idxs) {
    if (idxs.empty()) return T(0);
    T min_value = values[idxs[0]];
    for (const auto& idx : idxs) {
        const auto& value = values[idx];
        if (value < min_value) min_value = value;
    }
    return min_value;
}

template <typename T>
T maximum(const Tensor1DAcc<T>& values, const std::vector<int32_t>& idxs) {
    if (idxs.empty()) return T(0);
    T max_value = values[idxs[0]];
    for (const auto& idx : idxs) {
        const auto& value = values[idx];
        if (value > max_value) max_value = value;
    }
    return max_value;
}

template <typename T>
T mean(const Tensor1DAcc<T>& values, const std::vector<int32_t>& idxs) {
    T mean_value = 0;
    if (idxs.empty()) return mean_value;
    for (const auto& idx : idxs) mean_value += values[idx];
    return mean_value / static_cast<T>(idxs.size());
}

#endif  // TREE_TOPOLOGY_H