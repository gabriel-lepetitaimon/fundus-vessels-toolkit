#include "rasterize_topo.h"
#include "tree.h"

/**
 * @brief Compare a branch tree to a topology and compute the accuracy metrics.
 *
 * @param branch_list A 2D tensor of size (B, 2) containing the start and end node indices of each branch.
 * @param branch_parents A 1D tensor of size (B) containing the parent branch index of each branch.
 * @param branch_dirs A 1D boolean tensor of size (B) containing the direction of each branch.
 * @param expected_topoIDs A 1D tensor of size (B,) containing the index of the expected topology (Art or Vei) of
 * each branch.
 * @param branch_curves A vector of 2D tensors, each of size (L_i, 2), containing the pixel coordinates of each branch
 * curve.
 * @param nodes_yx A 2D tensor of size (N, 2) containing the pixel coordinates of each node.
 * @param topo_labels A vector of T 2D map of size (H, W) containing the topology labels as rasterized by
 * rasterize_topo.
 * @param topo_ranks A vector of T 2D map of size (H, W) containing the topology ranks as rasterized by rasterize_topo.
 * @param topo_fuzzy_skeleton A vector of T2D tensor of size (H, W) containing the fuzzy skeleton map.
 * @param branch_ids A 1D tensor containing the index of the compared branch. If empty, all branches are compared.
 * @param out_skeleton_map A 2D tensor of size (H, W). If provided, the function will draw the skeleton labels on this
 * map for visualization purposes.
 * @param bezier_interpolate Whether to fill branch curve using Bezier Cubic interpolation.
 *
 * @result An array of 5 tensors containing the following metrics:
 *  - skel_topoIDs_confmat: A 3D tensor of size (B, T, T+1) containing the confusion matrix for each branch. The last
 * dimension corresponds to the predicted topology index, while the second dimension corresponds to the expected
 * topology index. The last index of the second dimension corresponds to the "no prediction" case. If a pixel has a
 * topo_labels!=0 in its ``expected_topology_ids``, it is considered true-positive and assigned the corresponding
 * topology id. Otherwise, it's assigned the topology id with the higher topo_fuzzy_skeleton or -1 if none.
 *  - predicted_topoIDs: A 1D tensor of size (B) containing the most present skel_labels for each branch.
 *  - branch_topoIDs_confmat: A 2D tensor of size (T, T+1) containing the confusion matrix of branch label
 * (expected_topoIDs vs predicted_topoIDs). The last index of the second dimension corresponds to the "no prediction"
 * case.
 *  - dir_validity: A 1D boolean tensor of size (B) indicating whether each branch has a valid direction according to
 * the expected topology.
 *  - parent_validity: A 1D tensor of size (B) indicating whether each branch has a valid parent according to
 * the expected topology. Values are: 0: valid root or parent; -1: missing reconnection; -2: invalid reconnection
 * (should be root); >0: invalid parent (number of bifurcations to reach the expected parent).
 */
std::array<torch::Tensor, 5> read_skeleton_accuracies(
    const torch::Tensor& branch_list, const torch::Tensor& branch_parents, const torch::Tensor& branch_dirs,
    const torch::Tensor& expected_topoIDs, const std::vector<torch::Tensor>& branch_curves,
    const torch::Tensor& nodes_yx, const std::vector<torch::Tensor>& topo_labels,
    const std::vector<torch::Tensor>& topo_ranks, const std::vector<torch::Tensor>& topo_fuzzy_skeleton,
    const torch::Tensor& branch_ids, torch::Tensor& out_skeleton_map, float bezier_interpolate) {
    // === CHECK INPUTS ===
    TORCH_CHECK_VALUE(branch_list.dim() == 2 && branch_list.size(1) == 2 && branch_list.dtype() == torch::kLong,
                      "branch_list must be a 2D tensor of size (B, 2) and of dtype long");
    const std::size_t B = branch_list.size(0);
    auto branchList = branch_list.accessor<long, 2>();
    TORCH_CHECK_VALUE(
        branch_parents.dim() == 1 && branch_parents.size(0) == B && branch_parents.dtype() == torch::kLong,
        "branch_parents must be a 1D tensor of size B and of dtype long");
    auto branchPar = branch_parents.accessor<long, 1>();
    TORCH_CHECK_VALUE(branch_dirs.dim() == 1 && branch_dirs.size(0) == B && branch_dirs.dtype() == torch::kFloat32,
                      "branch_dirs must be a 1D tensor of size B and of dtype float32");
    auto branchDir = branch_dirs.accessor<float, 1>();
    TORCH_CHECK_VALUE(
        expected_topoIDs.dim() == 1 && expected_topoIDs.size(0) == B && expected_topoIDs.dtype() == torch::kLong,
        "expected_topoIDs must be a 1D tensor of size B and of dtype long");
    auto gtLabels = expected_topoIDs.accessor<long, 1>();
    TORCH_CHECK_VALUE(branch_curves.size() == B, "branch_curves must have the same number of elements as branch_list");

    TORCH_CHECK_VALUE(nodes_yx.dim() == 2 && nodes_yx.size(1) == 2 && nodes_yx.dtype() == torch::kFloat32,
                      "nodes_yx must be a 2D tensor of size (N, 2) and of dtype float32");
    std::size_t N = nodes_yx.size(0);
    auto nodesYX = nodes_yx.accessor<float, 2>();

    std::size_t T = topo_labels.size();
    TORCH_CHECK_VALUE(T > 0, "topo_labels and topo_ranks must have at least one topology");
    std::size_t H = 0, W = 0;
    std::vector<Tensor2DAcc<TopoLabel>> topoLabels;
    for (std::size_t i = 0; i < T; ++i) {
        TORCH_CHECK_VALUE(topo_labels[i].dim() == 2 && topo_labels[i].dtype() == torch::kUInt64,
                          "topo_labels must be 2D tensors of dtype uint64");
        if (i == 0)
            H = topo_labels[i].size(0), W = topo_labels[i].size(1);
        else
            TORCH_CHECK_VALUE(topo_labels[i].size(0) == H && topo_labels[i].size(1) == W,
                              "topo_labels must have the same shape for all topologies");
        topoLabels.push_back(topo_labels[i].accessor<TopoLabel, 2>());
    }
    TORCH_CHECK_VALUE(topo_ranks.size() == T, "topo_ranks must have the same number of elements as topo_labels");
    std::vector<Tensor2DAcc<float>> topoRanks;
    for (std::size_t i = 0; i < T; ++i) {
        TORCH_CHECK_VALUE(topo_ranks[i].dim() == 2 && topo_ranks[i].dtype() == torch::kFloat32,
                          "topo_ranks must be 2D tensors of dtype float");
        TORCH_CHECK_VALUE(topo_ranks[i].size(0) == H && topo_ranks[i].size(1) == W,
                          "topo_ranks must have the same shape as topo_labels");
        topoRanks.push_back(topo_ranks[i].accessor<float, 2>());
    }
    TORCH_CHECK_VALUE(topo_fuzzy_skeleton.size() == T,
                      "topo_fuzzy_skeleton must have the same number of elements as topo_labels");
    std::vector<Tensor2DAcc<at::Half>> topoFuzzySkeleton;
    for (std::size_t i = 0; i < T; ++i) {
        TORCH_CHECK_VALUE(topo_fuzzy_skeleton[i].dim() == 2 && topo_fuzzy_skeleton[i].dtype() == torch::kFloat16,
                          "topo_fuzzy_skeleton must be 2D tensors of dtype float16");
        TORCH_CHECK_VALUE(topo_fuzzy_skeleton[i].size(0) == H && topo_fuzzy_skeleton[i].size(1) == W,
                          "topo_fuzzy_skeleton must have the same shape as topo_labels");
        topoFuzzySkeleton.push_back(topo_fuzzy_skeleton[i].accessor<at::Half, 2>());
    }

    std::vector<std::size_t> branchSubset;
    if (branch_ids.numel() > 0) {
        TORCH_CHECK_VALUE(branch_ids.dim() == 1 && branch_ids.dtype() == torch::kLong,
                          "branch_ids must be a 1D tensor of dtype long");
        auto branchIds = branch_ids.accessor<long, 1>();
        for (std::size_t i = 0; i < branch_ids.size(0); ++i) {
            TORCH_CHECK_VALUE(branchIds[i] >= 0 && branchIds[i] < B, "branch_ids must be in the range [0, B)");
            branchSubset.push_back((std::size_t)branchIds[i]);
        }
    } else {
        branchSubset.resize(B);
        std::iota(branchSubset.begin(), branchSubset.end(), 0);
    }
    const std::size_t BS = branchSubset.size();

    bool draw_skeleton_labels = out_skeleton_map.numel() > 0;
    if (draw_skeleton_labels) {
        TORCH_CHECK_VALUE(out_skeleton_map.dim() == 2 && out_skeleton_map.dtype() == torch::kLong,
                          "out_skeleton_map must be a 2D tensor of dtype long");
        TORCH_CHECK_VALUE(out_skeleton_map.size(0) == H && out_skeleton_map.size(1) == W,
                          "out_skeleton_map must have the same shape as topo_labels");
    }

    // === PREPARE OUTPUT TENSORS ===
    torch::Tensor skel_topoIDs_confmat = torch::zeros({BS, T, T + 1}, torch::dtype(torch::kInt64));
    auto skelTopoIDsConfMat = skel_topoIDs_confmat.accessor<long, 3>();
    torch::Tensor predicted_topoIDs = torch::zeros({BS}, torch::dtype(torch::kInt64));
    auto predictedTopoIDs = predicted_topoIDs.accessor<long, 1>();
    torch::Tensor branch_topoIDs_confmat = torch::zeros({T, T + 1}, torch::dtype(torch::kInt64));
    auto branchTopoIDsConfMat = branch_topoIDs_confmat.accessor<long, 2>();
    torch::Tensor dir_validity = torch::zeros({BS}, torch::dtype(torch::kBool));
    auto dirValidity = dir_validity.accessor<bool, 1>();
    torch::Tensor parent_validity = torch::zeros({BS}, torch::dtype(torch::kInt32));
    auto parentValidity = parent_validity.accessor<int, 1>();

    // === COMPUTE METRICS ===
    // ...

    return {skel_topoIDs_confmat, predicted_topoIDs, branch_topoIDs_confmat, dir_validity, parent_validity};
}

/**
 * @brief Compare a single branch curve to the topology and compute the accuracy metrics.
 *
 * @param b_id The index of the branch.
 * @param branch_curve A vector of 2D points containing the pixel coordinates of the branch curve.
 * @param expected_topoID The index of the expected topology (Art or Vei) of the branch.
 * @param topo_labels A vector of T 2D map of size (H, W) containing the topology labels as rasterized by
 * rasterize_topo.
 * @param topo_ranks A vector of T 2D map of size (H, W) containing the topology ranks as rasterized by rasterize_topo.
 * @param skelLabelConfMat A 2D tensor of size (T, T+1) containing the confusion matrix for the branch. The last
 * dimension corresponds to the predicted topology labels, while the second dimension corresponds to the expected
 * topology labels. The last index of the second dimension corresponds to the "no prediction" case.
 * @param outSkelMap A 2D tensor of size (H, W). If provided, the function will draw the skeleton labels on this map for
 * visualization purposes.
 *
 * @result A tuple containing:
 * - The index of the most present skel_label for the branch.
 * - A boolean indicating whether the branch has a valid direction according to the expected topology.
 * - An integer indicating whether the branch has a valid parent according to the expected topology.
 */
std::tuple<int, float, int> read_branch_accuracy(const std::size_t b_id, const CurveYX& branch_curve,
                                                 std::size_t expected_topoID,
                                                 const std::vector<Tensor2DAcc<TopoLabel>>& topo_labels,
                                                 const std::vector<Tensor2DAcc<float>>& topo_ranks,
                                                 const std::vector<Tensor2DAcc<at::Half>>& topoFuzzySkel,
                                                 Tensor2DAcc<long>& skelLabelConfMat, Tensor2DAcc<long>& outSkelMap) {
    const std::size_t H = topo_labels[0].size(0), W = topo_labels[0].size(1);
    const std::size_t T = topo_labels.size();
    bool drawSkel = outSkelMap.size(0) > 0;

    // === CHECK TOPO IDs and DIRECTION ===
    std::vector<int> topoID_count(T, 0);
    float dir = 0.0f;
    IntPoint lastP = IntPoint::Invalid();
    float lastRank = 0.0f;
    for (const auto& p : branch_curve) {
        if (!p.is_inside(H, W)) continue;
        if (topo_labels[expected_topoID][p.y][p.x] != 0) {  // Valid topoID
            skelLabelConfMat[expected_topoID][expected_topoID] += 1;
            if (drawSkel) outSkelMap[p.y][p.x] = expected_topoID + 1;
            if (lastP.is_valid()) {
                float rank = topo_ranks[expected_topoID][p.y][p.x];
                const auto& fuzzySkel = topoFuzzySkel[expected_topoID];
                float normalCos = (fuzzySkel[p.y][p.x] - fuzzySkel[lastP.y][lastP.x]) / p.distance(lastP);
                dir += sign(rank - lastRank) * std::abs(normalCos);
                lastRank = rank, lastP = p;
            }
        } else {  // Invalid topoID
            int predicted_topoID = T + 1;
            at::Half max_plausibility = 0;
            for (std::size_t t = 0; t < topo_labels.size(); ++t) {
                if (topoFuzzySkel[t][p.y][p.x] > max_plausibility) {
                    max_plausibility = topoFuzzySkel[t][p.y][p.x];
                    predicted_topoID = t;
                }
            }
            skelLabelConfMat[expected_topoID][predicted_topoID] += 1;
            if (predicted_topoID < T) topoID_count[predicted_topoID] += 1;
            if (drawSkel) outSkelMap[p.y][p.x] = predicted_topoID + 1;  // +1 to avoid 0 (no prediction)
        }
    }
    int most_present_topoID = std::max_element(topoID_count.begin(), topoID_count.end()) - topoID_count.begin();

    // === CHECK PARENT ===
}