#include "disjoint_set.h"
#include "rasterize_topo.h"
#include "tree.h"

std::string label2str(const TopoLabel& label, const float& r = -1.0f) {
    std::ostringstream oss;
    oss << get_subtree(label) << ":";
    for (int8_t r = 0; r < get_rank(label); ++r) oss << (((label >> (51 - r)) & 1) ? "1" : "0");
    if (r >= 0.0f) oss << ":" << r;
    return oss.str();
}
TopoLabel parent(const TopoLabel& label, bool self_if_no_parent) {
    const auto& label_rank = get_rank(label);
    if (label_rank == 0) return self_if_no_parent ? label : 0;
    return (label - 1) & branching_bit_mask(label_rank - 1);
}
uint64_t branching_bit_mask(const uint8_t& rank) {
    return ~((static_cast<uint64_t>(1) << (52 - rank)) - static_cast<uint64_t>(1));
}
bool is_ancestor(const TopoLabel& ancestor, const TopoLabel& descendant, const bool& strict) {
    if (ancestor == descendant) return strict ? false : true;
    if (ancestor > descendant) return false;
    auto ancestor_rank_mask = branching_bit_mask(get_rank(ancestor));
    return (ancestor & ancestor_rank_mask) == (descendant & ancestor_rank_mask);
}
bool is_either_ancestor(const TopoLabel& label1, const TopoLabel& label2) {
    if (label1 == label2) return true;
    auto ancestor_rank_mask = branching_bit_mask(std::min(get_rank(label1), get_rank(label2)));
    return (label1 & ancestor_rank_mask) == (label2 & ancestor_rank_mask);
}
bool is_sibling(const TopoLabel& label1, const TopoLabel& label2) {
    if (label1 == label2) return false;
    auto rank = get_rank(label1);
    if (rank == 0 || rank != get_rank(label2)) return false;
    auto rank_mask = branching_bit_mask(rank - 1);
    return (label1 & rank_mask) == (label2 & rank_mask);
}
bool is_between(const TopoLabel& label, const TopoLabel& start, const TopoLabel& end, const float& rank,
                const float& start_rank, const float& end_rank, bool strict_start, bool strict_end) {
    if (label < start || label > end) return false;

    const uint64_t& start_rank_mask = branching_bit_mask(get_rank(start));
    if ((label & start_rank_mask) != (start & start_rank_mask)) return false;

    const uint64_t& rank_mask = branching_bit_mask(get_rank(label));
    if ((label & rank_mask) != (end & rank_mask)) return false;

    if (rank == start_rank) return strict_start ? false : true;
    if (rank == end_rank) return strict_end ? false : true;
    return start_rank < rank && rank < end_rank;
}
TopoLabel most_present_ancestor(const std::vector<int32_t>& curve, const Tensor1DAcc<TopoLabel>& topo_labels,
                                const Tensor1DAcc<float>& topo_ranks, float min_rank_threshold) {
    std::map<TopoLabel, int> label_count;
    for (const auto& idx : curve) {
        auto label = topo_labels[idx];
        // If the rank of the label is below the threshold, we consider its parent instead
        if (fmod(topo_ranks[idx], 1.0f) < min_rank_threshold) label = parent(label, true);
        label_count[topo_labels[idx]] += 1;
    }

    for (auto it = label_count.begin(); it != label_count.end(); ++it) {
        for (auto it2 = std::next(it); it2 != label_count.end(); ++it2)
            if (is_ancestor(it->first, it2->first, false)) it->second += it2->second;
    }

    const auto& maxIt = std::max_element(label_count.begin(), label_count.end(),
                                         [](const auto& a, const auto& b) { return a.second < b.second; });

    return maxIt->first;
}

std::array<torch::Tensor, 6> read_branches_topology(const std::vector<torch::Tensor>& branch_curves,
                                                    const IntPair& domain, const torch::Tensor& topo_idxs,
                                                    const torch::Tensor& topo_labels, const torch::Tensor& topo_ranks,
                                                    const torch::Tensor& fuzzy_skeleton, float min_rank_threshold,
                                                    float max_rank_tolerance, bool filter_overlap) {
    TORCH_CHECK_VALUE(topo_idxs.dim() == 2, "topo_idxs must be a 2D tensor");
    TORCH_CHECK_VALUE(topo_idxs.dtype() == torch::kUInt32, "topo_idxs must be of dtype uint32");
    TORCH_CHECK_VALUE(topo_labels.dim() == 1, "topo_labels must be a 1D tensor");
    TORCH_CHECK_VALUE(topo_labels.dtype() == torch::kUInt64, "topo_labels must be of dtype uint64");
    TORCH_CHECK_VALUE(topo_ranks.dim() == 1, "topo_ranks must be a 1D tensor");
    TORCH_CHECK_VALUE(topo_ranks.dtype() == torch::kFloat32, "topo_ranks must be of dtype float32");
    TORCH_CHECK_VALUE(fuzzy_skeleton.dim() == 1, "fuzzy_skeleton must be a 1D tensor");
    TORCH_CHECK_VALUE(fuzzy_skeleton.dtype() == torch::kFloat16, "fuzzy_skeleton must be of dtype float16");

    auto topo_idxs_acc = topo_idxs.accessor<uint32_t, 2>();
    auto topo_labels_acc = topo_labels.accessor<uint64_t, 1>();
    auto topo_ranks_acc = topo_ranks.accessor<float, 1>();
    auto fuzzy_skeleton_acc = fuzzy_skeleton.accessor<at::Half, 1>();

    std::size_t B = branch_curves.size();
    std::vector<TopoLabel> branch_labels(B);
    std::vector<float> branch_dir(B);
    std::vector<float> branch_plausibility(B);
    std::vector<float> branch_length(B);
    std::vector<std::array<TopoLabel, 2>> tips_label(B);
    std::vector<std::array<float, 2>> tips_rank(B);

    // → Read topology for each branch
#pragma omp parallel for
    for (std::size_t b = 0; b < B; ++b) {
        const auto& curveTensor = branch_curves[b];
        TORCH_CHECK_VALUE(curveTensor.dim() == 2, "Each branch curve must be a 2D tensor");
        TORCH_CHECK_VALUE(curveTensor.size(1) == 2, "Each branch curve must have 2 channels (y, x)");
        TORCH_CHECK_VALUE(curveTensor.dtype() == torch::kInt32, "Each branch curve must be of dtype int32");

        auto curve_acc = curveTensor.accessor<int32_t, 2>();
        std::tie(branch_labels[b], branch_dir[b], branch_plausibility[b], branch_length[b], tips_label[b],
                 tips_rank[b]) = read_branch_topology(curve_acc, domain, topo_idxs_acc, topo_labels_acc, topo_ranks_acc,
                                                      fuzzy_skeleton_acc, min_rank_threshold, max_rank_tolerance, b);
    }

    // → Filter branches that overlap on gt to keep only the most plausible one
    if (filter_overlap) {
        std::list<SizePair> overlapping_branch_pairs;
        for (std::size_t b0 = 0; b0 < B; ++b0) {
            if (branch_labels[b0] == 0) continue;
            auto [tail0_l, head0_l] = tips_label[b0];
            auto [tail0_r, head0_r] = tips_rank[b0];
            if (branch_dir[b0] < 0) std::swap(head0_l, tail0_l), std::swap(head0_r, tail0_r);
            for (std::size_t b1 = b0 + 1; b1 < B; ++b1) {
                auto [tail1_l, head1_l] = tips_label[b1];
                auto [tail1_r, head1_r] = tips_rank[b1];
                if (branch_dir[b1] < 0) std::swap(head1_l, tail1_l), std::swap(head1_r, tail1_r);

                if (is_between(tail0_l, tail1_l, head1_l, tail0_r, tail1_r, head1_r, false, true) ||
                    is_between(head0_l, tail1_l, head1_l, head0_r, tail1_r, head1_r, true, false) ||
                    is_between(tail1_l, tail0_l, head0_l, tail1_r, tail0_r, head0_r, false, true) ||
                    is_between(head1_l, tail0_l, head0_l, head1_r, tail0_r, head0_r, true, false))
                    overlapping_branch_pairs.emplace_back(SizePair{b0, b1});
            }
        }
        const auto& overlapping_branches = solve_clusters(overlapping_branch_pairs, B);
        for (auto cluster : overlapping_branches) {
            if (cluster.size() <= 1) continue;
            std::list<SizePair> cluster_pairs;
            for (const auto& pair : overlapping_branch_pairs) {
                for (const auto& b : cluster) {
                    if (pair[0] == b || pair[1] == b) {
                        cluster_pairs.push_back(pair);
                        break;
                    }
                }
            }

            // Iteratively remove the branch that overlaps the most with other branches in the cluster
            while (cluster_pairs.size() > 0) {
                std::map<std::size_t, float> branch_opposite_plausibility;
                for (const auto& pair : cluster_pairs) {
                    branch_opposite_plausibility[pair[0]] +=
                        branch_plausibility[pair[1]] * std::sqrt(branch_length[pair[1]]);
                    branch_opposite_plausibility[pair[1]] +=
                        branch_plausibility[pair[0]] * std::sqrt(branch_length[pair[0]]);
                }
                std::size_t b =
                    std::max_element(branch_opposite_plausibility.begin(), branch_opposite_plausibility.end(),
                                     [](const auto& a, const auto& b) { return a.second < b.second; })
                        ->first;

                branch_labels[b] = 0;
                branch_plausibility[b] = 0.0f;

                cluster_pairs.remove_if([b](const SizePair& p) { return p[0] == b || p[1] == b; });
            }
        }
    }

    return {vector_to_tensor(branch_labels), vector_to_tensor(branch_dir), vector_to_tensor(branch_plausibility),
            vector_to_tensor(branch_length), vector_to_tensor(tips_label), vector_to_tensor(tips_rank)};
}

std::tuple<TopoLabel, float, float, float, std::array<TopoLabel, 2>, std::array<float, 2>> read_branch_topology(
    const Tensor2DAcc<int32_t>& curve_acc, const IntPair& domain, const Tensor2DAcc<uint32_t>& topo_idxs,
    const Tensor1DAcc<TopoLabel>& topo_labels, const Tensor1DAcc<float>& topo_ranks,
    const Tensor1DAcc<at::Half>& fuzzy_skeleton, float min_rank_threshold, float max_rank_tolerance, int b_id) {
    std::size_t N = curve_acc.size(0), max_size = topo_labels.size(0);
    if (N == 0) return {0, 0.0f, 0.0f, 0, {0, 0}, {0.0f, 0.0f}};

    // → Check that at least half the branch is inside the gt tree topology
    std::vector<int32_t> curve;
    curve.reserve(N);
    std::vector<float> normalCosVec;
    normalCosVec.reserve(N);
    IntPoint prevP = IntPoint::Invalid();
    uint32_t prevIdx = 0;
    for (std::size_t n = 0; n < N; ++n) {
        IntPoint p = IntPoint(curve_acc[n][0], curve_acc[n][1]);
        if (!p.is_inside(0, 0, domain[0], domain[1])) {
            N--;
            continue;
        }
        const auto& idx = topo_idxs[p.y][p.x];
        if (idx >= max_size) continue;
        curve.push_back(idx);
        if (p != prevP && p.is_adjacent(prevP))
            normalCosVec.push_back((fuzzy_skeleton[idx] - fuzzy_skeleton[prevIdx]) / p.distance(prevP));

        prevP = p, prevIdx = idx;
    }

    auto curveLength = [curve, &curve_acc](const std::vector<int32_t>& finalCurve) -> float {
        if (curve.size() == 0) return 0.0f;

        // Find the curve length by looking forpoints of the original curve that are still in the final curve
        auto it = finalCurve.begin();
        std::size_t iniID = 0;  // Index of the point in the original curve
        float length = 0.0f;

        IntPoint prevP = IntPoint::Invalid();
        do {
            while (curve[iniID] != *it) {  // Iterate through the original curve until the current point
                if (++iniID >= curve.size()) return length;
            }
            IntPoint p = IntPoint(curve_acc[iniID][0], curve_acc[iniID][1]);
            if (p.is_adjacent(prevP)) length += prevP.distance(p);
            prevP = p;
        } while ((++it) != finalCurve.end());
        return length;
    };

    // → Skip branch if too short or if oriented towards a normal direction to the skeleton +/- 40 degrees
    float normalCos = 0;
    for (const auto& cos : movingAvg(normalCosVec, 4)) normalCos += std::abs(cos);
    if (normalCosVec.size() > 0) normalCos /= normalCosVec.size();
    float plausibility = std::max(mean(fuzzy_skeleton, curve) / MAX_PLAUSIBILITY, 0.0f);
    if (curve.size() < 3 || normalCos >= 0.75f) {
        return {0, 0.0f, std::max(plausibility - normalCos, 0.0f), curveLength(curve), {0, 0}, {0.0f, 0.0f}};
    }

    float known_label_ratio = static_cast<float>(curve.size()) / static_cast<float>(N);

    // → Search points descendant of the main ancestor ...
    const auto& main_ancestor = most_present_ancestor(curve, topo_labels, topo_ranks, min_rank_threshold);
    std::vector<int32_t> curveTmp;
    curveTmp.reserve(curve.size());
    for (const auto& idx : curve)
        if (is_ancestor(main_ancestor, topo_labels[idx], false)) curveTmp.push_back(idx);
    if (curveTmp.size() < 3) return {0, 0.0f, plausibility, curveLength(curveTmp), {0, 0}, {0, 0}};
    float valid_ancestor_ratio = static_cast<float>(curveTmp.size()) / static_cast<float>(curve.size());
    curve = curveTmp;
    curveTmp.clear();

    // → Sum fuzzy skeleton value as plausibility score
    plausibility = std::max(mean(fuzzy_skeleton, curve) / MAX_PLAUSIBILITY - normalCos, 0.0f);
    float curve_length = curveLength(curve);

    // → Get the direction of the branch based on the topological distance map
    float direction = 0.0f;
    for (std::size_t i = 2; i < curve.size(); ++i) {
        const auto &idx0 = curve[i - 2], &idx1 = curve[i];
        const auto& diff = topo_ranks[idx1] - topo_ranks[idx0];
        if (diff > 0) direction += 1.0f;
        if (diff < 0) direction -= 1.0f;
    }
    direction /= curve.size() - 2;

    // → Skip branch if not enough valid ancestor points or low directionality
    if (known_label_ratio < 0.33f || valid_ancestor_ratio < 0.66f || abs(direction) < 0.33f)
        return {0, 0.0f, plausibility, curve_length, {0, 0}, {0.0f, 0.0f}};

    // → Exclude starting curve points which are part of the transition between labels
    float min_rank = int(std::floor(minimum(topo_ranks, curve)));
    min_rank += float(min_rank_threshold);

    for (const auto& idx : curve)
        if (topo_ranks[idx] >= min_rank) curveTmp.push_back(idx);
    if (curveTmp.size() >= 3 && curveTmp.size() != curve.size()) curve = curveTmp;

    // → Compute maximum valid rank considering rank tolerance
    curveTmp.clear();
    for (const auto& idx : curve)
        if (std::fmod(topo_ranks[idx], 1.0f) > max_rank_tolerance) curveTmp.push_back(idx);
    float max_rank = 50;
    TopoLabel max_label = 0;
    if (curveTmp.size() > 0) {
        const auto& [max_idx, max_value] = argmax(topo_ranks, curveTmp);
        max_rank = std::ceil(max_value);
        max_label = topo_labels[max_idx];
    }
    curveTmp.clear();

    // → Assign the most occurring label to the branch
    std::vector<std::pair<TopoLabel, int>> label_count;
    for (const auto& idx : curve) {  // Count labels
        // Use max_label for points above max_rank
        TopoLabel label = topo_ranks[idx] >= max_rank ? max_label : topo_labels[idx];
        bool found = false;
        for (auto& [l, count] : label_count) {
            if (label == l) {
                count += 1;
                found = true;
                break;
            }
        }
        if (!found) label_count.emplace_back(label, 1);
    }

    TopoLabel branch_label = main_ancestor;
    int max_count = 0;
    for (const auto& [l, count] : label_count) {  // Search the most present label
        if (count > max_count) {
            max_count = count;
            branch_label = l;
        }
    }

    // → Get tip labels and ranks
    std::array<TopoLabel, 2> tip_labels;
    std::array<float, 2> tip_ranks;
    for (const auto& [tip, idx] : {std::make_pair(0, curve.front()), std::make_pair(1, curve.back())}) {
        float rank = topo_ranks[idx];
        if (rank < max_rank) {
            tip_ranks[tip] = static_cast<float>(rank);
            tip_labels[tip] = topo_labels[idx];
        } else {
            tip_ranks[tip] = max_rank;
            tip_labels[tip] = max_label;
        }
    }

    return {branch_label, direction, plausibility, curve_length, tip_labels, tip_ranks};
}

struct BranchTopo {
    TopoLabel label;      // Topological label of the branch
    float p_dir;          // Average direction of the branch based on the topological distance map
    float plausibility;   // Average plausibility of the branch based on the fuzzy skeleton
    float length;         // Length of the branch curve in pixels
    TopoLabel tailLabel;  // Label of the first tip
    TopoLabel headLabel;  // Label of the second tip
    float tailRank;       // Rank of the first tip
    float headRank;       // Rank of the second tip
};
const std::size_t UNKNOWN_ID = std::numeric_limits<std::size_t>::max();
const long UNKNOWN = -2;
const long ROOT = -1;
struct OptiBranchTopo {
    float dirLogit = 0;       // Logit direction of the branch (positive for arteries, negative for veins)
    long bestTopo = UNKNOWN;  // Which topology is the best for this branch, -1 if the branch is a false positive
    long parent = UNKNOWN;    // Index of the parent branch or the incoming line
};
class TopoLine {
   public:
    std::size_t id;  // Index of the line in the lines tensor
    std::size_t b0;  // Parent branch
    int t0;          // Tip of the parent branch (0 or 1)
    std::size_t b1;  // Child branch
    int t1;          // Tip of the child branch (0 or 1)

    bool check_t0(const float& b0_dirLogit) const {
        if (b0_dirLogit == 0) return true;
        return t0 == (b0_dirLogit > 0 ? 1 : 0);
    }
    bool check_t1(const float& b1_dirLogit) const {
        if (b1_dirLogit == 0) return true;
        return t1 == (b1_dirLogit > 0 ? 0 : 1);
    }
};

/**
 * @brief A* exploration algorithm to find the optimal combination of node states in a sequence of nodes with fixed
 * order and length.
 * @param base_cost A 2D vector of shape (N, S) containing the base cost of each node state. N is the number of nodes
 * and S is the number of possible states for each node.
 * @param marginal_cost A function that computes the marginal cost of adding a new node state to the current sequence of
 * node states. It takes the current sequence of node states and the new node state as input and returns the marginal
 * cost.
 *
 * @return A tuple containing the optimal sequence of node states and the total cost of that sequence.
 */
std::tuple<std::vector<int>, float> astar_explore(std::vector<std::map<int, float>> base_cost,
                                                  std::function<float(const std::vector<int>&, int)> marginal_cost) {
    // Compute the remaining minimum cost for each length of the sequence to use as a heuristic for A* search
    std::vector<float> remaining_min_cost(base_cost.size() + 1, 0.0f);
    // Cumulative sum from the last node to the second one,
    // TODO: sort by the minimum cost of each node state to improve the heuristic
    for (int i = (int)base_cost.size() - 1; i >= 0; --i) {
        float min_cost = std::numeric_limits<float>::max();
        for (const auto& entry : base_cost[i])
            if (entry.second < min_cost) min_cost = entry.second;
        remaining_min_cost[i] = remaining_min_cost[i + 1] + min_cost;
    }
    // Create a priority queue to store the nodes to explore, ordered by their total cost (base cost + marginal cost)
    struct Node {
        std::vector<int> states;  // Current sequence of node states
        float cost;               // Total cost of the current sequence
        float total_cost;
    };
    auto cmp = [&remaining_min_cost](const Node& a, const Node& b) {
        return a.total_cost > b.total_cost;
    };  // Min-heap based on total cost
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);

    // Initialize the priority queue with the base costs of the first node
    float cost;
    for (const auto& entry : base_cost[0]) {
        cost = entry.second + marginal_cost({}, entry.first);
        pq.push({{entry.first}, cost, cost + remaining_min_cost[1]});
    }

    // Perform A* search
    while (!pq.empty()) {
        Node current = pq.top();
        pq.pop();

        // If we have reached the last node, return the current sequence and its cost
        if (current.states.size() == base_cost.size()) return {current.states, current.cost};

        // Explore the next node states
        int next_node = current.states.size();
        for (const auto& entry : base_cost[next_node]) {
            int new_state = entry.first;
            const float &new_state_cost = entry.second,
                        &new_state_marginal_cost = marginal_cost(current.states, new_state);
            if (!std::isfinite(new_state_marginal_cost))
                continue;  // If new_state_marginal_cost is infinity, skip this state

            // Push the new sequence to the priority queue
            std::vector<int> new_states = current.states;
            new_states.push_back(new_state);
            cost = current.cost + new_state_cost + new_state_marginal_cost;
            pq.push({new_states, cost, cost + remaining_min_cost[next_node + 1]});
        }
    }

    // If we exhaust the priority queue without reaching the last node, return an empty sequence and infinite cost
    std::cerr << "Warning: A* search exhausted without finding a solution." << std::endl;
    return {std::vector<int>{}, std::numeric_limits<float>::infinity()};
}

/**
 * @brief Computes the optimal topology (optimal branch fp and av label, direction and optimal lines) of a set of
 * branches given their topological information and the lines connecting them.
 * @param branches_topology A vector of arrays containing the topological information of each branch. Each array
 * contains the following elements:
 * - branch_labels: A tensor of shape (B,) containing the topological labels of each branch.
 * - branch_dir: A tensor of shape (B,) containing the average direction of each branch based on the topological
 * distance map.
 * - branch_plausibility: A tensor of shape (B,) containing the average plausibility of each branch based on the
 * fuzzy skeleton.
 * - branch_length: A tensor of shape (B,) containing the length of each branch curve in pixels.
 * - tips_label: A tensor of shape (B, 2) containing the labels of the two tips of each branch.
 * - tips_rank: A tensor of shape (B, 2) containing the ranks of the two tips of each branch.
 * @param plausibility_threshold The threshold for determining the plausibility of each branch.
 * @param tipPos The (y,x) positions of the tips of each branch as a tensor of shape (B, 2, 2).
 * @param plausibility_threshold The threshold for determining the plausibility of each branch.
 * @param linesTensor A tensor of shape (L, 4) containing the lines connecting the branches. Each line is
 * represented by four integers: (b0, t0, b1, t1), where b0 and b1 are the indices of the parent and child branches,
 * and t0 and t1 are the indices of the tips of the parent and child branches (0 or 1).
 * @param branchListTensor A tensor of shape (B, 2) defining the connexion between branch in the graph. If provided this
 * will be use to refine the parent assignment for very short branch with unreliable topology.
 *
 * @return An array of three tensors containing the optimal topology:
 # - branch_labels: A tensor of shape (B,) containing the optimal topological labels of each branch. (-1 for false
 positive branches)
 * - branch_dir: A tensor of shape (B,) containing the logit direction of each branch (positive for arteries, negative
 * for veins).
 * - lines: A boolean tensor of shape (L,) describing whether each line is optimal (1) or not (0).
 */
std::array<torch::Tensor, 3> optimal_topology(const std::vector<std::array<torch::Tensor, 6>>& branches_topology,
                                              const torch::Tensor& linesTensor, torch::Tensor nodesPosTensor,
                                              torch::Tensor branchListTensor, float plausibility_threshold) {
    // === UNPACK BRANCHES TOPOLOGY AND LINES ===
    const std::size_t B = branches_topology[0][0].size(0);
    const std::size_t T = branches_topology.size();
    const std::size_t L = linesTensor.size(0);

    std::vector<std::vector<BranchTopo>> topologies(branches_topology.size());
    for (std::size_t t = 0; t < T; ++t) {
        auto& topo = topologies[t];
        const auto& [branch_labels, branch_dir, branch_plausibility, branch_length, tips_label, tips_rank] =
            branches_topology[t];
        TORCH_CHECK_VALUE(
            branch_labels.dim() == 1 && branch_labels.size(0) == (long)B && branch_labels.dtype() == torch::kUInt64,
            "branch_labels must be a 1D tensor of size B and of dtype uint64");
        TORCH_CHECK_VALUE(
            branch_dir.dim() == 1 && branch_dir.size(0) == (long)B && branch_dir.dtype() == torch::kFloat32,
            "branch_dir must be a 1D tensor of size B and of dtype float32");
        TORCH_CHECK_VALUE(branch_plausibility.dim() == 1 && branch_plausibility.size(0) == (long)B &&
                              branch_plausibility.dtype() == torch::kFloat32,
                          "branch_plausibility must be a 1D tensor of size B and of dtype float32");
        TORCH_CHECK_VALUE(
            branch_length.dim() == 1 && branch_length.size(0) == (long)B && branch_length.dtype() == torch::kFloat32,
            "branch_length must be a 1D tensor of size B and of dtype float32");
        TORCH_CHECK_VALUE(tips_label.dim() == 2 && tips_label.size(0) == (long)B && tips_label.size(1) == 2 &&
                              tips_label.dtype() == torch::kUInt64,
                          "tips_label must be a 2D tensor of size (B, 2) and of dtype uint64");
        TORCH_CHECK_VALUE(tips_rank.dim() == 2 && tips_rank.size(0) == (long)B && tips_rank.size(1) == 2 &&
                              tips_rank.dtype() == torch::kFloat32,
                          "tips_rank must be a 2D tensor of size (B, 2) and of dtype float32");

        auto branch_labels_acc = branch_labels.accessor<uint64_t, 1>();
        auto branch_dir_acc = branch_dir.accessor<float, 1>();
        auto branch_plausibility_acc = branch_plausibility.accessor<float, 1>();
        auto branch_length_acc = branch_length.accessor<float, 1>();
        auto tips_label_acc = tips_label.accessor<uint64_t, 2>();
        auto tips_rank_acc = tips_rank.accessor<float, 2>();
        for (std::size_t b = 0; b < B; ++b) {
            topo.push_back({branch_labels_acc[b], branch_dir_acc[b], branch_plausibility_acc[b], branch_length_acc[b],
                            tips_label_acc[b][0], tips_label_acc[b][1], tips_rank_acc[b][0], tips_rank_acc[b][1]});
            if (branch_dir_acc[b] < 0) {
                std::swap(topo.back().tailLabel, topo.back().headLabel);
                std::swap(topo.back().tailRank, topo.back().headRank);
            }
        }
    }

    TORCH_CHECK_VALUE(linesTensor.dim() == 2 && linesTensor.size(1) == 4 && linesTensor.dtype() == torch::kInt64,
                      "linesTensor must be a 2D tensor of size (L, 4) and of dtype int64");
    std::vector<TopoLine> lines;
    std::vector<std::array<std::size_t, 2>> root_line_ids(B, {UNKNOWN_ID, UNKNOWN_ID});
    lines.reserve(linesTensor.size(0));
    auto lines_acc = linesTensor.accessor<long, 2>();
    for (std::size_t l = 0; l < L; ++l) {
        if (lines_acc[l][0] >= 0)
            lines.push_back(TopoLine{l, (std::size_t)lines_acc[l][0], (int)lines_acc[l][1],
                                     (std::size_t)lines_acc[l][2], (int)lines_acc[l][3]});
        else
            root_line_ids[lines_acc[l][2]][lines_acc[l][3]] = l;
    }

    TORCH_CHECK_VALUE(
        nodesPosTensor.dim() == 2 && nodesPosTensor.size(1) == 2 && nodesPosTensor.dtype() == torch::kFloat64,
        "nodesPos must be a 2D tensor of size (N, 2) and of dtype float64");
    std::vector<Point> nodesPos;
    tensor_to_vector(nodesPosTensor, nodesPos);

    TORCH_CHECK_VALUE(branchListTensor.dim() == 2 && branchListTensor.size(1) == 2 &&
                          branchListTensor.size(0) == (long)B && branchListTensor.dtype() == torch::kInt64,
                      "branchList must be a 2D tensor of size (B, 2) and of dtype int64");
    std::vector<std::array<long, 2>> branchList;
    tensor_to_vector(branchListTensor, branchList);

    // === UTILITIES FUNCTIONS ===
    auto headPos = [&nodesPos, &topologies, &branchList](std::size_t b, std::size_t t) -> Point {
        return nodesPos[branchList[b][topologies[t][b].p_dir >= 0 ? 1 : 0]];
    };
    auto tailPos = [&nodesPos, &topologies, &branchList](std::size_t b, std::size_t t) -> Point {
        return nodesPos[branchList[b][topologies[t][b].p_dir >= 0 ? 0 : 1]];
    };

    struct Parent {
        long id = ROOT;         // Index of the parent branch or the incoming line
        long branch_id = ROOT;  // Index of the parent branch
        float headRank = 0;     // Rank of the parent tip
    };

    auto best_parent = [&topologies, &headPos, &tailPos](std::size_t b1, std::size_t t,
                                                         const std::vector<std::size_t>& admissibleB0,
                                                         const std::vector<std::size_t>& b0LinesIds = {}) -> Parent {
        const auto& b1Topo = topologies[t][b1];
        auto b1Subtree = get_subtree(b1Topo.label);
        Parent bestParent, bestSibling;

        for (std::size_t b = 0; b < admissibleB0.size(); ++b) {
            const auto& b0 = admissibleB0[b];
            if (b0 == b1) continue;
            const auto& b0Topo = topologies[t][b0];

            // Optimal parent must be ...
            if (b1Subtree != get_subtree(b0Topo.label)) continue;  //... in the same subtree,
            Parent b0AsParent = {b0LinesIds.empty() ? b0 : b0LinesIds[b], b0, b0Topo.headRank};
            if (is_ancestor(b0Topo.headLabel, b1Topo.tailLabel, false) && b0Topo.headRank <= b1Topo.tailRank) {
                // ... the closest ancestor
                if (b0Topo.headRank > bestParent.headRank) bestParent = b0AsParent;
            } else if (is_sibling(b0Topo.headLabel, b1Topo.tailLabel)) {
                // ... or the nearest uncertain sibling
                if (std::abs(b0Topo.headRank - b1Topo.tailRank) < std::abs(bestSibling.headRank - b1Topo.tailRank))
                    bestSibling = b0AsParent;
            }
        }

        if (bestParent.id <= ROOT && bestSibling.id <= ROOT)
            return {ROOT, ROOT, 0};  // No parent or sibling found, assign to ROOT
        else if (bestSibling.id <= ROOT)
            return bestParent;  // If no sibling found, assign the best parent
        else {
            const auto& b1Tail = tailPos(b1, t);
            const auto& sHead = headPos(bestSibling.branch_id, t);
            auto sHeadDist = distance(sHead, b1Tail);

            if (bestParent.id <= ROOT) {
                // If only sibling found, check that b0 tail is closer to b1 head than tail
                if (sHeadDist < distance(b1Tail, tailPos(bestSibling.branch_id, t)))
                    return bestSibling;
                else
                    return Parent{ROOT, ROOT, 0};
            }

            // If both parent and sibling found, choose the one with the closest head tip to this branch's tail tip
            const auto& pHead = headPos(bestParent.branch_id, t);
            return (distance(pHead, b1Tail) <= sHeadDist) ? bestParent : bestSibling;
        }
    };

    auto is_overlapping = [&topologies](std::size_t b0, std::size_t b1, std::size_t t) -> bool {
        const auto& topo = topologies[t];
        if (get_subtree(topo[b0].label) != get_subtree(topo[b1].label))
            return false;  // Early exit if the two branches don't belong to the same subtree

        const auto &b0TailRank = topo[b0].tailRank, b1TailRank = topo[b1].tailRank;
        if (!is_either_ancestor(b0TailRank, b1TailRank))
            return false;  // Early exit if the two branches tails are not ancestor of one another

        const auto &b0HeadRank = topo[b0].headRank, &b1HeadRank = topo[b1].headRank;
        const auto &b0TailLabel = topo[b0].tailLabel, &b1TailLabel = topo[b1].tailLabel;
        auto b0HeadLabel = topo[b0].headLabel, b1HeadLabel = topo[b1].headLabel;
        if (!is_either_ancestor(b0HeadLabel, b1HeadLabel)) {
            // Early return if the two branches heads are not ancestor of one another
            // but double check if they were not mistaken for their parent
            bool uncertainParent = false;
            if (fmod(b0HeadRank, 1.0) <= 0.1) b0HeadLabel = parent(b0HeadLabel, true), uncertainParent = true;
            if (fmod(b1HeadRank, 1.0) <= 0.1) b1HeadLabel = parent(b1HeadLabel, true), uncertainParent = true;
            if (!uncertainParent || !is_either_ancestor(b0HeadLabel, b1HeadLabel)) return false;
        }

        // If both tails and heads are ancestor of one another, then the branches overlap if either:
        return is_between(b0TailLabel, b1TailLabel, b1HeadLabel,  // b0 tail is in b1
                          b0TailRank, b1TailRank, b1HeadRank, false, true) ||
               is_between(b0HeadLabel, b1TailLabel, b1HeadLabel,  // b0 head is in b1
                          b0HeadRank, b1TailRank, b1HeadRank, true, false) ||
               is_between(b1TailLabel, b0TailLabel, b0HeadLabel,  // b1 tail is in b0
                          b1TailRank, b0TailRank, b0HeadRank, false, true) ||
               is_between(b1HeadLabel, b0TailLabel, b0HeadLabel,  // b1 head is in b0
                          b1HeadRank, b0TailRank, b0HeadRank, true, false);
    };

    // === IDENTIFY OVERLAPPING BRANCHES ===
    ConstantDisjointSet branch_disjoint_sets(B);
    std::vector<std::array<std::size_t, 3>> overlapping_branch_pairs;
    for (std::size_t t = 0; t < T; ++t) {
        const auto& topo = topologies[t];
        for (std::size_t b0 = 0; b0 < B; ++b0) {
            const auto& b0Topo = topo[b0];
            if (b0Topo.label == 0) continue;
            for (std::size_t b1 = b0 + 1; b1 < B; ++b1) {
                const auto& b1Topo = topo[b1];
                if (b1Topo.label == 0) continue;

                // Check if the branches overlap
                if (is_overlapping(b0, b1, t)) {
                    branch_disjoint_sets.merge(b0, b1);
                    overlapping_branch_pairs.push_back({b0, b1, t});
                }
            }
        }
    }
    std::list<std::vector<std::size_t>> overlaps;
    std::vector<bool> overlapping_branches(B, false);
    for (const auto& [_, set] : branch_disjoint_sets.get_sets()) {
        if (set.size() > 1) {
            overlaps.push_back(std::vector<std::size_t>(set.begin(), set.end()));
            for (const auto& b : set) overlapping_branches[b] = true;
        }
    }

    // === PROCESS NON-OVERLAPPING BRANCHES ===
    std::vector<OptiBranchTopo> optiTopo(B);
    auto setOptiTopo = [&optiTopo, &topologies](std::size_t b, int t) {
        optiTopo[b].bestTopo = t;
        if (t != -1) {
            const auto& bestTopo = topologies[t][b];
            optiTopo[b].dirLogit = bestTopo.p_dir * bestTopo.plausibility * bestTopo.length;
        }
    };
    struct TopoScore {
        std::size_t id;
        float plausibility;
    };
    std::vector<TopoScore> topoScores(T);
    for (std::size_t b = 0; b < B; ++b) {
        if (overlapping_branches[b]) continue;
        for (std::size_t t = 0; t < T; ++t) {
            const auto& topo = topologies[t][b];
            topoScores[t] = {t, topo.label != 0 ? topo.plausibility * topo.length : 0};
        }

        // Sort descending by plausibility
        std::sort(topoScores.begin(), topoScores.end(),
                  [](const TopoScore& a, const TopoScore& b) { return a.plausibility > b.plausibility; });
        if (topoScores[0].plausibility < plausibility_threshold) {
            // If plausibility is low for all topologies, consider the branch as a false positive
            optiTopo[b].bestTopo = -1;
            optiTopo[b].dirLogit = 0;
            for (std::size_t t = 0; t < T; ++t) {
                const auto& topo = topologies[t][b];
                optiTopo[b].dirLogit += topo.p_dir * topo.plausibility * topo.length;
            }
        } else if ((topoScores[0].plausibility - topoScores[1].plausibility <
                    plausibility_threshold * topoScores[0].plausibility)) {
            // If top2 plausibilities are equivalent process the branch as overlapping to decide given the surrounding
            if (!overlapping_branches[b]) {
                overlapping_branches[b] = true;
                overlaps.push_back({b});
            }
        } else
            // Otherwise assign the best topology and direction
            setOptiTopo(b, topoScores[0].id);
    }

    // Filter lines to keep only those that connect branches with the same best topology and correct tip directions
    auto check_line = [&optiTopo, &topologies](const TopoLine& line, int b1_topo = UNKNOWN) -> bool {
        const auto& b0_topo = optiTopo[line.b0].bestTopo;
        if (b0_topo == ROOT || !line.check_t0(optiTopo[line.b0].dirLogit)) return false;  // Check parent best topology
        if (b1_topo == UNKNOWN) b1_topo = optiTopo[line.b1].bestTopo;
        if (b1_topo != UNKNOWN) {
            if (b0_topo != UNKNOWN && b0_topo != b1_topo) return false;    // Check topologies
            if (!line.check_t1(optiTopo[line.b1].dirLogit)) return false;  // Check child tail tip
        }
        return true;
    };
    std::vector<TopoLine> nonOverlapLines;
    std::vector<TopoLine> allOverlapLines;
    for (const auto& line : lines) {
        if (overlapping_branches[line.b0] || overlapping_branches[line.b1]) {
            allOverlapLines.push_back(line);
        } else if (check_line(line))
            nonOverlapLines.push_back(line);
    }

    // === OPTIMIZE OVERLAPPING BRANCHES ===
    // Sort overlap clusters by lowest tail rank
    overlaps.sort([&](const auto& a, const auto& b) {
        std::vector<float> minRankA(T, std::numeric_limits<float>::max());
        std::vector<float> minRankB(T, std::numeric_limits<float>::max());
        for (const auto& b : a) {
            for (std::size_t t = 0; t < T; ++t) {
                if (topologies[t][b].label != 0 && topologies[t][b].tailRank < minRankA[t])
                    minRankA[t] = topologies[t][b].tailRank;
            }
        }
        for (const auto& b : b) {
            for (std::size_t t = 0; t < T; ++t) {
                if (topologies[t][b].label != 0 && topologies[t][b].tailRank < minRankB[t])
                    minRankB[t] = topologies[t][b].tailRank;
            }
        }
        for (std::size_t t = 0; t < T; ++t)
            if (std::isfinite(minRankA[t]) && std::isfinite(minRankB[t])) return minRankA[t] < minRankB[t];
        return std::min(*minRankA.begin(), *minRankA.end()) < std::min(*minRankB.begin(), *minRankB.end());
    });

    for (auto& overlap : overlaps) {
        // If too many overlap fall back to an approximative solution
        if (overlap.size() > 10) {
            std::vector<bool> overlapMask(B, false);
            for (const auto& b : overlap) overlapMask[b] = true;
            std::list<std::array<std::size_t, 3>> cluster_pairs;
            for (const auto& pair : overlapping_branch_pairs)
                if (overlapMask[pair[0]]) cluster_pairs.push_back(pair);

            // Iteratively remove the branch that overlaps the most with other branches in the cluster
            std::vector<std::vector<bool>> removedFromTopo(B, std::vector<bool>(T, false));
            while (cluster_pairs.size() > 0) {
                std::map<SizePair, float> branch_opposite_plausibility;
                for (const auto& [b0, b1, t] : cluster_pairs) {
                    const auto &topo0 = topologies[t][b0], &topo1 = topologies[t][b1];
                    branch_opposite_plausibility[{b0, t}] += topo1.plausibility * std::sqrt(topo1.length);
                    branch_opposite_plausibility[{b1, t}] += topo0.plausibility * std::sqrt(topo0.length);
                }

                auto [b, t] = std::max_element(branch_opposite_plausibility.begin(), branch_opposite_plausibility.end(),
                                               [](const auto& a, const auto& b) { return a.second < b.second; })
                                  ->first;

                removedFromTopo[b][t] = true;
                cluster_pairs.remove_if(
                    [&](const std::array<std::size_t, 3>& p) { return p[2] == t && (p[0] == b || p[1] == b); });
            }

            // --- Assign the best topology to the remaining branches ---
            for (const auto& b : overlap) {
                float best_plausibility = 0.0f;
                int best_topo = -1;
                for (std::size_t t = 0; t < T; ++t) {
                    if (removedFromTopo[b][t]) continue;

                    if (topologies[t][b].plausibility > best_plausibility) {
                        best_plausibility = topologies[t][b].plausibility;
                        best_topo = t;
                    }
                }
                if (best_topo != -1) setOptiTopo(b, best_topo);
            }

            continue;
        }

        // --- Split the branches that overlap into homo and hetero directional branches ---
        // those that have the same direction across all topologies and those that have different directions. This
        // is because branch with the same direction can be ordered by their head rank and processed sequentially
        // (A* exploration), while every combination of branches with different directions must be explored.
        std::vector<std::size_t> heteroDirBranches, homoDirBranches;
        std::vector<float> homoDirLogits;

        if (overlap.size() == 1)  // If single uncertain av branch, process as hetero to prevent a* exploration
            heteroDirBranches.push_back(overlap[0]);
        else {
            for (const auto& b : overlap) {
                bool same_direction = true;
                for (std::size_t t = 1; t < T; ++t) {
                    if (topologies[t][b].p_dir * topologies[0][b].p_dir < 0) {
                        heteroDirBranches.push_back(b);
                        same_direction = false;
                        break;
                    }
                }
                if (same_direction) homoDirBranches.push_back(b);
            }
            // --- Order the homoDirBranches by ascending head rank ---
            std::sort(homoDirBranches.begin(), homoDirBranches.end(),
                      [&topologies, &T](const std::size_t& b0, const std::size_t& b1) {
                          // Compare the head ranks of the two branches in a common topology
                          std::vector<float> total_plausiblity(T, 0.0f);
                          for (std::size_t t = 0; t < topologies.size(); ++t) {
                              const auto &b0Topo = topologies[t][b0], &b1Topo = topologies[t][b1];
                              if (b0Topo.label != 0 && b1Topo.label != 0)
                                  total_plausiblity[t] =
                                      b0Topo.plausibility * b0Topo.length + b1Topo.plausibility * b1Topo.length;
                          }
                          auto max_plausibility = std::max_element(total_plausiblity.begin(), total_plausiblity.end());
                          if (*max_plausibility > 0) {
                              std::size_t t = max_plausibility - total_plausiblity.begin();
                              return topologies[t][b0].headRank < topologies[t][b1].headRank;
                          }
                          return false;  // If no common topology, keep the original order
                      });

            for (std::size_t homoIdx = 0; homoIdx < homoDirBranches.size(); ++homoIdx) {
                const auto& b = homoDirBranches[homoIdx];
                float dirLogit = homoDirLogits.emplace_back(0.0f);
                for (std::size_t t = 0; t < T; ++t) {
                    const auto& topo = topologies[t][b];
                    dirLogit += topo.p_dir * topo.plausibility * topo.length;
                }
            }
        }

        // --- Generate combinations of branches with heterogeneous directions ---
        struct Combination {
            std::vector<int> heteroTopoIDs;  // Index of the topology for each branch in heteroDirBranches
            std::vector<float> heteroDir;    // Logit direction of each branch in heteroDirBranches
            std::vector<Point> heteroTails;  // Positions of the tails of each branch in heteroDirBranches
            std::vector<int> homoTopoIDs;    // Index of the topology for each branch in homoDirBranches
            float plausibility = 0.0f;
            std::vector<std::list<std::size_t>> branchByTopo;
        };
        std::vector<Combination> heteroDirBranchCombinations(1);
        heteroDirBranchCombinations[0].branchByTopo.resize(T);
        // Reserve space for all combinations to use reference to comb while adding new combinations
        heteroDirBranchCombinations.reserve(pow(T + 1, heteroDirBranches.size()));
        for (const auto& b : heteroDirBranches) {
            for (auto& comb : heteroDirBranchCombinations) {
                // Append combination with the current branch plausible topologies
                for (std::size_t t = 0; t < T; ++t) {
                    if (topologies[t][b].plausibility > plausibility_threshold) {
                        Combination& newComb = heteroDirBranchCombinations.emplace_back(comb);
                        // Check for conflicts with previous branches in the combination
                        bool conflicting = false;
                        for (const auto& existingB : comb.branchByTopo[t]) {
                            if (topologies[t][b].headRank < topologies[t][existingB].tailRank)
                                // If the current branch's head is before the existing branch's tail ...
                                break;  // ... all subsequent existing branches will be after -> skip them

                            if (is_overlapping(b, existingB, t)) {
                                conflicting = true;
                                break;
                            }
                        }

                        if (!conflicting) {
                            newComb.heteroTopoIDs.push_back(t);
                            const auto& bTopo = topologies[t][b];
                            newComb.heteroDir.push_back(bTopo.p_dir);
                            newComb.heteroTails.push_back(tailPos(b, t));
                            newComb.plausibility += bTopo.plausibility * bTopo.length;
                            // Insert the branch in the appropriate branchByTopo, maintaining tailRank order
                            const auto& topo = topologies[t];
                            auto it = newComb.branchByTopo[t].begin();
                            while (it != newComb.branchByTopo[t].end() && topo[*it].tailRank < topo[b].tailRank) ++it;
                            newComb.branchByTopo[t].insert(it, b);
                        } else
                            heteroDirBranchCombinations.pop_back();  // Remove the conflicting combination
                    }
                }
                // Update the current combination to include this branch as False Positive
                comb.heteroTopoIDs.push_back(-1);
                comb.heteroDir.push_back(0.0f);
                comb.heteroTails.push_back(Point(0.0, 0.0));
            }
        }

        // Select lines connecting the overlapping branches
        std::vector<bool> homoDirMask(B, false);
        std::vector<bool> heteroDirMask(B, false);
        for (const auto& b : homoDirBranches) homoDirMask[b] = true;
        for (const auto& b : heteroDirBranches) heteroDirMask[b] = true;

        const auto &HOMO = homoDirBranches.size(), HETERO = heteroDirBranches.size();
        auto to_homo_hetero_index = [&](std::size_t b) -> std::tuple<bool, bool, std::size_t> {
            for (bool homo : {true, false}) {
                if (!(homo ? homoDirMask : heteroDirMask)[b]) continue;
                const auto& branches = homo ? homoDirBranches : heteroDirBranches;
                auto it = std::find(branches.begin(), branches.end(), b);
                if (it != branches.end()) return {true, homo, it - branches.begin()};
                break;
            }
            return {false, false, 0};  // Not found
        };

        // Prepare the base cost of homoDirBranches
        std::vector<std::map<int, float>> base_cost(HOMO);
        for (std::size_t i = 0; i < homoDirBranches.size(); ++i) {
            const auto& b = homoDirBranches[i];
            for (std::size_t t = 0; t < T; ++t) {
                const auto& topo = topologies[t][b];
                if (topo.label != 0 && topo.plausibility > plausibility_threshold)
                    base_cost[i][t] = -topo.plausibility * topo.length;
            }
            base_cost[i][-1] = 0.0f;  // False Positive option
        }

        // Prepare reachable parents
        struct OverlappingAdmissibleParents {
            std::vector<bool> homo;
            std::vector<bool> hetero;
            std::vector<long> non_overlapping;
        };
        std::vector<OverlappingAdmissibleParents> homoParents, heteroParents;
        for (bool initialize_homo : {true, false}) {
            auto& parents = initialize_homo ? homoParents : heteroParents;
            auto SIZE = initialize_homo ? HOMO : HETERO;
            const auto& branches = initialize_homo ? homoDirBranches : heteroDirBranches;

            if (L == 0) {
                parents.resize(
                    SIZE, {std::vector<bool>(HOMO, true), std::vector<bool>(HETERO, true), std::vector<long>(T, ROOT)});
                for (std::size_t i = 0; i < SIZE; ++i) {
                    const auto& b1 = branches[i];
                    for (long t = 0; t < (long)T; ++t) {
                        if (topologies[t][b1].label == 0) continue;
                        std::vector<std::size_t> sameTopoBranches;
                        for (std::size_t b0 = 0; b0 < B; b0++)
                            if (optiTopo[b0].bestTopo == t) sameTopoBranches.push_back(b0);
                        parents[i].non_overlapping[t] = best_parent(b1, t, sameTopoBranches).branch_id;
                    }
                }
            } else {
                parents.resize(SIZE, {std::vector<bool>(HOMO, false), std::vector<bool>(HETERO, false),
                                      std::vector<long>(T, ROOT)});
            }
        }

        // .. and if lines were provided, select valid lines involving the hetero-dir branches
        std::vector<TopoLine> heteroLines;
        if (L != 0) {
            std::vector<std::vector<std::size_t>> homoReachableNonOverlapParents(HOMO);
            std::vector<std::vector<std::vector<std::size_t>>> heteroReachableNonOverlapParents(
                HETERO, std::vector<std::vector<std::size_t>>(T));

            for (const auto& line : allOverlapLines) {
                auto [b1_in_overlap, b1_is_homo, b1_idx] = to_homo_hetero_index(line.b1);
                if (!b1_in_overlap) continue;  // Skip line if b1 is not in this overlap
                auto [b0_in_overlap, b0_is_homo, b0_idx] = to_homo_hetero_index(line.b0);
                // Check direction of b0 (if its direction is homogeneous across topologies)
                if (b0_is_homo && !line.check_t0(homoDirLogits[b0_idx])) continue;

                if (b1_is_homo) {                                         // If b1 is homo-dir
                    if (!line.check_t1(homoDirLogits[b1_idx])) continue;  // Check direction of b1
                    if (!b0_in_overlap) {
                        if (overlapping_branches[line.b0]) continue;
                        if (check_line(line))  // Ensure the line is consistent with optiTopo
                            homoReachableNonOverlapParents[b1_idx].push_back(line.b0);
                    } else if (b0_is_homo)
                        homoParents[b1_idx].homo[b0_idx] = true;
                    else
                        homoParents[b1_idx].hetero[b0_idx] = true;

                } else {
                    if (!b0_in_overlap) {
                        if (overlapping_branches[line.b0]) continue;
                        for (std::size_t t = 0; t < T; t++)
                            if (check_line(line, t)) heteroReachableNonOverlapParents[b1_idx][t].push_back(line.b0);
                    } else
                        heteroLines.push_back(line);
                }
            }
            // For each topology find the optimal non-overlapping parents
            for (std::size_t t = 0; t < topologies.size(); ++t) {
                for (std::size_t homoIdx = 0; homoIdx < HOMO; ++homoIdx) {
                    std::vector<std::size_t> admissibleB0;
                    for (const auto& b0 : homoReachableNonOverlapParents[homoIdx])
                        if (optiTopo[b0].bestTopo == t) admissibleB0.push_back(b0);
                    homoParents[homoIdx].non_overlapping[t] =
                        best_parent(homoDirBranches[homoIdx], t, admissibleB0).branch_id;
                }
                for (std::size_t heteroIdx = 0; heteroIdx < HETERO; ++heteroIdx) {
                    const auto& admissibleB0 = heteroReachableNonOverlapParents[heteroIdx][t];
                    heteroParents[heteroIdx].non_overlapping[t] =
                        best_parent(heteroDirBranches[heteroIdx], t, admissibleB0).branch_id;
                }
            }
        }

        // --- For each combination, find the best topology for the homoDirBranches using A* exploration ---
        for (auto& comb : heteroDirBranchCombinations) {
            // Update reachable parents for heteroDirBranch given the current combination
            if (L != 0) {
                for (auto& heteroParent : heteroParents) {
                    heteroParent.homo.assign(HOMO, false);
                    heteroParent.hetero.assign(HETERO, false);
                }
                std::vector<std::vector<std::size_t>> heteroReachableNonOverlapParents(HETERO);
                for (const auto& line : heteroLines) {
                    auto [b0_in_overlap, b0_is_homo, b0_idx] = to_homo_hetero_index(line.b0);
                    auto [b1_in_overlap, b1_is_homo, b1_idx] = to_homo_hetero_index(line.b1);
                    if (!b0_in_overlap || !b1_in_overlap || comb.heteroTopoIDs[b1_idx] < 0)
                        continue;  // Only consider lines pointing to an active hetero branch
                    if (!line.check_t1(comb.heteroDir[b1_idx]))
                        continue;  // Check direction of b1 according to the combination
                    if (!b0_is_homo) {
                        if (line.check_t0(comb.heteroDir[b0_idx]))  // Check direction of b0 according to the comb.
                            heteroParents[b1_idx].hetero[b0_idx] = true;
                    } else
                        heteroParents[b1_idx].homo[b0_idx] = true;
                }
            }
            // Sort the heteroDirBranches by ascending head rank within each topology
            for (auto& topoBranches : comb.branchByTopo) {
                topoBranches.sort([&topologies](std::size_t b0, std::size_t b1) {
                    for (std::size_t t = 0; t < topologies.size(); ++t)
                        if (topologies[t][b0].label != 0 && topologies[t][b1].label != 0)
                            return topologies[0][b0].headRank < topologies[0][b1].headRank;
                    return false;  // If no common topology, keep the original order
                });
            }

            // Prepare function to compute minimal distance the current branch tail and the heads of assigned
            // branches
            auto distance_to_optimal_parent = [&](const std::vector<int>& assignedTopo, std::size_t b, bool b_is_homo,
                                                  int b_topo) -> float {
                if (b_topo == -1) return 0.0f;  // No distance for False Positive
                std::size_t b_id = b_is_homo ? homoDirBranches[b] : heteroDirBranches[b];
                const auto& bTail = tailPos(b_id, b_topo);
                const auto& reachableParents = b_is_homo ? homoParents[b] : heteroParents[b];

                std::vector<std::size_t> admissibleB0;

                for (std::size_t i = 0; i < assignedTopo.size(); i++)
                    if (assignedTopo[i] == b_topo && (L == 0 || reachableParents.homo[i]))
                        admissibleB0.push_back(homoDirBranches[i]);
                for (std::size_t i = 0; i < comb.heteroTopoIDs.size(); i++)
                    if (comb.heteroTopoIDs[i] == b_topo && (L == 0 || reachableParents.hetero[i]))
                        admissibleB0.push_back(heteroDirBranches[i]);
                if (reachableParents.non_overlapping[b_topo] >= 0)
                    admissibleB0.push_back(reachableParents.non_overlapping[b_topo]);

                const auto& parent = best_parent(b_id, b_topo, admissibleB0);
                if (parent.id <= ROOT) return 0.0f;

                const auto& parentHead = headPos(parent.branch_id, b_topo);
                return distance(parentHead, bTail);
            };

            auto hetero_branch_cost = [&](const std::vector<int>& finalTopo) {
                float cost = 0.0f;
                for (std::size_t heteroI = 0; heteroI < HETERO; ++heteroI)
                    cost += distance_to_optimal_parent(finalTopo, heteroI, false, comb.heteroTopoIDs[heteroI]);
                return cost;
            };

            if (homoDirBranches.empty()) {
                // If there are no homoDirBranches, we can directly compute the cost of the heteroDirBranches...
                comb.plausibility -= hetero_branch_cost({});
                continue;  // ... and skip the A* exploration for this combination
            }

            // Define the marginal cost function for the A* exploration
            auto marginal_cost = [&](const std::vector<int>& assignedTopo, int b_topo) -> float {
                if (b_topo == -1) return 0.0f;  // No marginal cost for False Positive
                std::size_t homoI = assignedTopo.size(), b = homoDirBranches[homoI];
                const auto& topo = topologies[b_topo];
                // Check if the branch conflicts with any previously assigned branch in the same topology
                for (int i = (int)assignedTopo.size() - 1; i >= 0; i--) {
                    if (assignedTopo[i] != b_topo) continue;  // Only check branches assigned to the same topology
                    if (topo[homoDirBranches[i]].headRank < topo[b].tailRank)
                        break;  // No need to check further, as branches are sorted by head rank
                    if (is_overlapping(b, homoDirBranches[i], b_topo))  // Invalidate current topology
                        return std::numeric_limits<float>::infinity();
                }
                const auto& combHeteroBranches = comb.branchByTopo[b_topo];
                for (auto heteroB = combHeteroBranches.rbegin(); heteroB != combHeteroBranches.rend(); ++heteroB) {
                    if (topo[*heteroB].headRank < topo[b].tailRank) break;  // No need to check further
                    if (is_overlapping(b, *heteroB, b_topo))                // Invalidate current topology
                        return std::numeric_limits<float>::infinity();
                }

                // Amongst the branch already assigned, find the nearest head tip of the same topo
                auto cost = distance_to_optimal_parent(assignedTopo, homoI, true, b_topo);  // Weight the distance cost

                if (homoI == homoDirBranches.size() - 1) {
                    // If this is the last branch, also add the cost of the heteroDirBranches
                    auto finalTopo = assignedTopo;
                    finalTopo.push_back(b_topo);            // Add the current branch to the final topology assignment
                    cost += hetero_branch_cost(finalTopo);  // Weight the distance cost of the heteroDirBranches
                }
                return cost;
            };

            // Perform A* exploration to find the optimal topology assignment for the homoDirBranches
            auto [optimalTopoAssignment, totalCost] = astar_explore(base_cost, marginal_cost);
            comb.homoTopoIDs = optimalTopoAssignment;
            comb.plausibility -= totalCost;  // Subtract the distance cost from the total plausibility
        }

        // --- Save the best combination ---
        auto bestComb = std::max_element(
            heteroDirBranchCombinations.begin(), heteroDirBranchCombinations.end(),
            [](const Combination& a, const Combination& b) { return a.plausibility < b.plausibility; });
        for (std::size_t i = 0; i < bestComb->heteroTopoIDs.size(); ++i)
            setOptiTopo(heteroDirBranches[i], bestComb->heteroTopoIDs[i]);
        for (std::size_t i = 0; i < bestComb->homoTopoIDs.size(); ++i)
            setOptiTopo(homoDirBranches[i], bestComb->homoTopoIDs[i]);

        for (const auto& b : overlap) overlapping_branches[b] = false;  // Mark the branches as processed
    }
    // === ASSIGN Line or Branch best parent ===
    std::vector<std::vector<std::size_t>> branch_by_bestTopo(T);
    for (std::size_t b = 0; b < B; ++b)
        if (optiTopo[b].bestTopo >= 0) branch_by_bestTopo[optiTopo[b].bestTopo].push_back(b);

    // Complete non-overlapping lines list with valid overlapping lines
    lines = std::move(nonOverlapLines);
    for (const auto& line : allOverlapLines) {
        if (optiTopo[line.b0].bestTopo != optiTopo[line.b1].bestTopo) continue;  // Different best topologies
        if (line.t0 != (optiTopo[line.b0].dirLogit > 0 ? 1 : 0)) continue;       // Check parent head tip
        if (line.t1 != (optiTopo[line.b1].dirLogit > 0 ? 0 : 1)) continue;       // Check child tail tip
        lines.push_back(line);
    }

    // For each topology, find the best parent for each branch
    for (std::size_t t = 0; t < T; ++t) {
        const auto& branches = branch_by_bestTopo[t];

        std::vector<TopoLine> topoLines;
        for (const auto& line : lines)  // Select lines connecting branches in the current topology
            if (optiTopo[line.b0].bestTopo == (int)t) topoLines.push_back(line);

        for (const auto& b1 : branches) {
            std::vector<std::size_t> admissibleB0, lineIds;
            if (L > 0) {
                // Limit the admissible parents to those that are connected by a line to the current branch
                for (const auto& line : topoLines) {
                    if (line.b1 == b1) {
                        admissibleB0.push_back(line.b0);
                        lineIds.push_back(line.id);
                    }
                }
            } else
                admissibleB0 = branches;  // If no lines, all branches are admissible parents

            optiTopo[b1].parent = best_parent(b1, t, admissibleB0, lineIds).id;
        }
    }

    // === POST-FIX ===
    // Break cycles in the parent assignment
    std::vector<int> parents(B, -1);
    for (std::size_t b = 0; b < B; ++b) {
        const auto& p = optiTopo[b].parent;
        if (p >= 0)
            parents[b] = L > 0 ? lines_acc[p][0] : p;
        else
            parents[b] = -1;
    }

    while (true) {
        const auto& cycles = find_cycles(parents);
        if (cycles.empty()) break;

        for (const auto& cycle : cycles) {
            // Find the branch with the lowest tail rank in the cycle and set it as root (no parent)
            auto rootBranch = std::min_element(cycle.begin(), cycle.end(), [&](std::size_t b0, std::size_t b1) {
                const auto& b0TailRank = topologies[optiTopo[b0].bestTopo][b0].tailRank;
                const auto& b1TailRank = topologies[optiTopo[b1].bestTopo][b1].tailRank;
                return b0TailRank < b1TailRank;
            });

            optiTopo[*rootBranch].parent = ROOT;
            parents[*rootBranch] = -1;  // Update the parent array to reflect the change
        }
    }

    // === PREPARE OUTPUTS ===
    torch::Tensor branch_best_topos = torch::zeros({static_cast<long>(B)}, torch::dtype(torch::kInt));
    torch::Tensor branch_dir = torch::zeros({static_cast<long>(B)}, torch::dtype(torch::kFloat32));
    auto topo_acc = branch_best_topos.accessor<int, 1>();
    auto dir_acc = branch_dir.accessor<float, 1>();
    for (std::size_t b = 0; b < B; ++b) {
        topo_acc[b] = optiTopo[b].bestTopo;
        if (optiTopo[b].bestTopo != -1) dir_acc[b] = optiTopo[b].dirLogit;
    }

    torch::Tensor connectivity = L > 0 ? torch::zeros({(long)L}, torch::dtype(torch::kBool))
                                       : torch::empty({(long)B}, torch::dtype(torch::kInt));

    if (L > 0) {
        auto opti_lines = connectivity.accessor<bool, 1>();
        for (std::size_t b = 0; b < B; ++b) {
            const auto& bOptiTopo = optiTopo[b];
            if (bOptiTopo.parent >= 0)
                opti_lines[bOptiTopo.parent] = true;
            else if (bOptiTopo.parent == ROOT) {
                std::size_t root_line_id = root_line_ids[b][bOptiTopo.dirLogit > 0 ? 0 : 1];
                if (root_line_id < L)
                    opti_lines[root_line_id] = true;
                else
                    std::cerr << "Warning: Branch " << b
                              << " has no valid root line for its direction. Skipping connectivity assignment."
                              << std::endl;
            }
        }
    } else {
        auto opti_parent = connectivity.accessor<int, 1>();
        for (std::size_t b = 0; b < B; ++b) {
            const auto& bOptiTopo = optiTopo[b];
            if (bOptiTopo.parent >= 0)
                opti_parent[b] = bOptiTopo.parent;
            else
                opti_parent[b] = -1;  // No parent
        }
    }

    return {branch_best_topos, branch_dir, connectivity};
}