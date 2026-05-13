#ifndef GRAPH_MATCHING_H
#define GRAPH_MATCHING_H

#include "common.h"

/*************************************************************************************************
   === node_matching.cpp ===
 ************************************************************************************************/
using IncidentBranchesMatch = std::vector<std::vector<torch::Tensor>>;  // [N1][N2][{b1, b2}]

struct IncidentBranchesData {
    const Tensor3DAcc<float> dot_features_1;
    const Tensor3DAcc<float> dot_features_2;
    const Tensor3DAcc<float> scalar_features_1;
    const Tensor3DAcc<float> scalar_features_2;
    const Tensor1DAcc<float> scalar_features_std;
    const Tensor3DAcc<float> branch_uvector_1;
    const Tensor3DAcc<float> branch_uvector_2;
    const bool uvectors_available;
    const int angle_dot_features;
};

std::tuple<torch::Tensor, IncidentBranchesMatch, torch::Tensor> nodes_similarity(
    torch::Tensor matchable_nodes, torch::Tensor dot_features_1, torch::Tensor dot_features_2,
    torch::Tensor scalar_features_1, torch::Tensor scalar_features_2, torch::Tensor scalar_features_std,
    torch::Tensor n_branches_1, torch::Tensor n_branches_2, torch::Tensor branch_uvector_1,
    torch::Tensor branch_uvector_2, bool rotation_invariant, int angle_dot_features = 0);

std::tuple<float, std::vector<UIntPair>, uint> node_similarity(uint n1, uint n2, uint B1, uint B2,
                                                               const IncidentBranchesData& data,
                                                               bool rotation_invariant);

std::tuple<float, std::vector<UIntPair>, uint> djikstra_optimal_branch_matching(
    std::vector<uint> B1_order, std::vector<uint> B2_order,
    std::function<float(uint, uint, const UIntPair&)> eval_match_cost, std::vector<UIntPair> initial_pairs = {{0, 0}},
    const bool are_match_cost_independent_of_ini_pairs = true);

// === Djikstra Utility structures ===
namespace BranchMatching {
struct DjikstraStep {
    uint iniPairId;  // initial_pairs_ID
    uint b1;         // branch 1
    uint b2;         // branch 2
    std::size_t inline index(const std::array<std::size_t, 3>& Ns) const {
        return (iniPairId * Ns[1] + b1) * Ns[2] + b2;
    }
    bool operator>(const DjikstraStep& other) const;
};

struct DjikstraPath {
    float cost;         // Total cost to reach this step
    DjikstraStep step;  // The current evaluated match (initial_pairs_ID, b1, b2)

    std::size_t inline stepIndex(const std::array<std::size_t, 3>& Ns) const { return step.index(Ns); }
};

class LowerPriorityPath {
    uint B1;
    uint B2;

   public:
    LowerPriorityPath(uint B1, uint B2);
    bool operator()(const DjikstraPath& p1, const DjikstraPath& p2) const;
};

static const float DROP_COST = 0.5;  // Drop cost for branch matching

float inline astar_lower_bound(const DjikstraStep& step, uint B1, uint B2) {
    // Distance to final diagonal
    return abs((int)(B1 - step.b1) - (int)(B2 - step.b2)) * DROP_COST;
};

};  // namespace BranchMatching

/*************************************************************************************************
   === mcc.cpp ===
 ************************************************************************************************/

#endif  // GRAPH_MATCHING_H