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
    // #pragma omp parallel for
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
    float normalCos = 0.0f;
    int nNormalCos = 0;
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
        if (p.is_adjacent(prevP)) {
            normalCos += (fuzzy_skeleton[idx] - fuzzy_skeleton[prevIdx]) / p.distance(prevP);
            nNormalCos++;
        }
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

    // → Skip branch if too short or if oriented towards a normal direction to the skeleton
    if (curve.size() < 3 || abs(normalCos / nNormalCos) >= 0.6f)
        return {0, 0.0f, mean(fuzzy_skeleton, curve), curveLength(curve), {0, 0}, {0.0f, 0.0f}};

    float known_label_ratio = static_cast<float>(curve.size()) / static_cast<float>(N);

    // → Search points descendant of the main ancestor ...
    const auto& main_ancestor = most_present_ancestor(curve, topo_labels);
    std::vector<int32_t> curveTmp;
    curveTmp.reserve(curve.size());
    for (const auto& idx : curve)
        if (is_ancestor(main_ancestor, topo_labels[idx], false)) curveTmp.push_back(idx);
    if (curveTmp.size() < 3)
        return {0, 0.0f, mean(fuzzy_skeleton, curveTmp), curveLength(curveTmp), {0, 0}, {0.0f, 0.0f}};
    float valid_ancestor_ratio = static_cast<float>(curveTmp.size()) / static_cast<float>(curve.size());
    curve = curveTmp;
    curveTmp.clear();

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
    if (known_label_ratio < 0.33f || valid_ancestor_ratio < 0.66f || abs(direction) < 0.66f)
        return {0, 0.0f, mean(fuzzy_skeleton, curve), curveLength(curve), {0, 0}, {0.0f, 0.0f}};

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

    // → Sum fuzzy skeleton value as plausibility score
    float plausibility = mean(fuzzy_skeleton, curve);

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

    return {branch_label, direction, plausibility, curveLength(curve), tip_labels, tip_ranks};
}

TopoLabel most_present_ancestor(const std::vector<int32_t>& curve, const Tensor1DAcc<TopoLabel>& topo_labels) {
    std::map<TopoLabel, int> label_count;
    for (const auto& idx : curve) label_count[topo_labels[idx]] += 1;

    for (auto it = label_count.begin(); it != label_count.end(); ++it) {
        for (auto it2 = std::next(it); it2 != label_count.end(); ++it2)
            if (is_ancestor(it->first, it2->first, false)) it->second += it2->second;
    }

    const auto& maxIt = std::max_element(label_count.begin(), label_count.end(),
                                         [](const auto& a, const auto& b) { return a.second < b.second; });
    return maxIt->first;
}