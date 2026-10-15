#include "rasterize_topo.h"
#include "tree.h"

/**
 * @brief Compare a branch tree to a topology and compute the accuracy metrics.
 *
 * @param branch_list A 2D tensor of size (B, 2) containing the start and end node indices of each branch.
 * @param branch_parents A 1D tensor of size (B) containing the parent branch index of each branch.
 * @param branch_dirs A 1D boolean tensor of size (B) containing the direction of each branch.
 * @param expected_topology A 1D tensor of size (B,) containing the expected topology of each branch.
 * @param branch_curves A vector of 2D tensors, each of size (L_i, 2), containing the pixel coordinates of each branch
 * curve.
 * @param nodes_yx A 2D tensor of size (N, 2) containing the pixel coordinates of each node.
 * @param topo_labels A vector of T 2D map of size (H, W) containing the topology labels as rasterized by
 * rasterize_topo.
 * @param topo_ranks A vector of T 2D map of size (H, W) containing the topology ranks as rasterized by rasterize_topo.
 * @param topo_fuzzy_skeleton A vector of T 2D map of size (H, W) containing the fuzzy skeleton as rasterized by
 * rasterize_topo.
 * @param bezier_interpolate Whether to fill branch curve using Bezier Cubic interpolation.
 *
 * @result An array of 6 tensors containing the following metrics:
 *  - skel_branch_id: A 1D tensor of size sum(L_i) containing the branch index of every pixel of branch curves.
 *  - skel_labels: A 1D tensor of size sum(L_i) containing the topology id of every pixels of branch curves.
 *                 If a pixel has a topo_labels!=0 in its ``expected_topology``, it is considered true-positive and
 *                  assigned the corresponding topology id.
 *                 Otherwise, it's assigned the topology id with the higher topo_fuzzy_skeleton or -1 if none.
 *  - branch_labels: A 1D tensor of size (B) containing the most present skel_labels for each branch.
 *  - branch_dir: A 1D tensor of size (B) containing direction for each branch according to its expected topology.
 *  - branch_
 */
std::array<torch::Tensor, 6> read_skeleton_topology(
    const torch::Tensor& branch_list, const torch::Tensor& branch_parents, const torch::Tensor& branch_dirs,
    const torch::Tensor& expected_topology, const std::vector<torch::Tensor>& branch_curves,
    const torch::Tensor& nodes_yx, const std::vector<torch::Tensor>& topo_labels,
    const std::vector<torch::Tensor>& topo_ranks, float bezier_interpolate) {
    TORCH_CHECK_VALUE(branch_list.dim() == 2 && branch_list.size(1) == 2 && branch_list.dtype() == torch::kLong,
                      "branch_list must be a 2D tensor of size (B, 2) and of dtype long");
    const std::size_t B = branch_list.size(0);
    TORCH_CHECK_VALUE(
        branch_parents.dim() == 1 && branch_parents.size(0) == B && branch_parents.dtype() == torch::kLong,
        "branch_parents must be a 1D tensor of size B and of dtype long");
    TORCH_CHECK_VALUE(branch_dirs.dim() == 1 && branch_dirs.size(0) == B && branch_dirs.dtype() == torch::kFloat32,
                      "branch_dirs must be a 1D tensor of size B and of dtype float32");
    TORCH_CHECK_VALUE(nodes_yx.dim() == 2 && nodes_yx.size(1) == 2 && nodes_yx.dtype() == torch::kFloat32,
                      "nodes_yx must be a 2D tensor of size (N, 2) and of dtype float32");
    std::size_t N = nodes_yx.size(0);

    // Prepare output tensors
    auto options_int64 = torch::dtype(torch::kInt64).device(branch_list.device());
    auto options_float32 = torch::dtype(torch::kFloat32).device(branch_list.device());

    torch::Tensor branch_labels = torch::zeros({static_cast<long>(B)}, options_int64);
    torch::Tensor branch_dir = torch::zeros({static_cast<long>(B)}, options_float32);
    torch::Tensor branch_plausibility = torch::zeros({static_cast<long>(B)}, options_float32);
    torch::Tensor branch_length = torch::zeros({static_cast<long>(B)}, options_float32);
    torch::Tensor tips_label = torch::zeros({static_cast<long>(B), 2}, options_int64);
}