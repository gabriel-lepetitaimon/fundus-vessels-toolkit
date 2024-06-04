#include "neighbors.h"



/**
 * @brief Find the first and last endpoint of each branch.
 * 
 * This method compute returns, for each branches, a pair of Point representing the first and last pixels of the branch.
 * The first and last pixels are defined following the branch_list arguments: 
 *   - the first pixel of the branch i is the branch endpoint, which is closest to the node given by branch_list[i][0].
 *   - the last pixel of the branch i is the branch endpoint, which is closest to the node given by branch_list[i][1].
 * In case of conflict, the closest pixel wins.
 * 
 * @param branch_labels A 2D tensor of shape (H, W) containing the branch labels.
 * @param node_yx A 2D tensor of shape (N, 2) containing the coordinates of the nodes.
 * @param branch_list A 2D tensor of shape (B, 2) containing the list of branches.
*/
std::vector<std::array<IntPair, 2>> find_branch_endpoints(const torch::Tensor& branch_labels, const torch::Tensor& node_yx, const torch::Tensor& branch_list) {

    const int B = branch_list.size(0);
    const int H = branch_labels.size(0), W = branch_labels.size(1);
    auto bLabels_acc = branch_labels.accessor<int, 2>();
    auto node_yx_acc = node_yx.accessor<int, 2>();
    auto branch_list_acc = branch_list.accessor<int, 2>();

    // Find the branch start and endpoints
    std::vector<std::array<IntPair, 2>> branches_endpoints(B);
    std::vector<std::atomic_flag> first_endpoints(B);

    // -- Search junctions and endpoints --
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < H-1; y++) {
        for (int x = 1; x < W-1; x++) {
            const int b = bLabels_acc[y][x];

            // If the current pixel is part of a branch ...
            if(b == 0)
                continue;

            // ... check if it's an endpoint.
            bool is_endpoint = false;
            for(const PointWithID& n : NEIGHBORHOOD){
                if(bLabels_acc[y+n.y][x+n.x] == b) {
                    if(!is_endpoint)
                        is_endpoint = true; // First neighbor found
                    else {
                        is_endpoint = false; // Second neighbor found
                        break;               // => not an endpoint
                    }
                }
            }
            if (!is_endpoint)
                continue;

            // If it's an endpoint, store it.
            bool first_endpoint = first_endpoints[b-1].test_and_set();    // TODO: Check with openmp
            branches_endpoints[b-1][first_endpoint] = {y, x};
        }
    }

    // -- Ensure the first endpoints is the closest to the first node --
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        int node0_id = branch_list_acc[b][0], node1_id = branch_list_acc[b][1];
        const Point n0 = {node_yx_acc[node0_id][0], node_yx_acc[node0_id][1]};
        const Point n1 = {node_yx_acc[node1_id][0], node_yx_acc[node1_id][1]};
        const Point p0 = branches_endpoints[b][0];
        const Point p1 = branches_endpoints[b][1];

        const float min_dist = std::min(distance(p0, n0), distance(p1, n1));
        // If the distance n1p0 or n0p1 is shorter than n1p1 and n0p0 ...
        if(distance(n0, p1) < min_dist || distance(n1, p0) < min_dist){
            // ... swap the endpoints.
            branches_endpoints[b][0] = p1.toIntPair();
            branches_endpoints[b][1] = p0.toIntPair();
        }
            
    }

    return branches_endpoints;
}


/**
 * @brief Track the orientation of branches in a vessel graph.
 * 
 * This method compute returns, for each branches, the list of the coordinates of its pixels.
 * The first and last pixels are defined following the branch_list arguments: 
 *   - the first pixel of the branch i is the branch endpoint, which is closest to the node given by branch_list[i][0].
 *   - the last pixel of the branch i is the branch endpoint, which is closest to the node given by branch_list[i][1].
 * In case of conflict, the closest pixel wins.
 * 
 * @param branch_labels A 2D tensor of shape (H, W) containing the branch labels.
 * @param node_yx A 2D tensor of shape (N, 2) containing the coordinates of the nodes.
 * @param branch_list A 2D tensor of shape (B, 2) containing the list of branches.
*/
std::vector<std::vector<IntPair>> track_branches(const torch::Tensor& branch_labels, const torch::Tensor& node_yx, const torch::Tensor& branch_list) {

    const int B = branch_list.size(0);
    const int H = branch_labels.size(0), W = branch_labels.size(1);
    auto bLabels_acc = branch_labels.accessor<int, 2>();

    // Find the branch start and endpoints
    auto branch_endpoints = find_branch_endpoints(branch_labels, node_yx, branch_list);
    std::vector<std::vector<IntPair>> branches_pixels(B);

    // Track the branches
    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        const Point start_p = branch_endpoints[b][0];
        const Point end_p = branch_endpoints[b][1];
        const int bLabel = b+1;

        // Initialize the branch pixels list with the start pixel
        std::vector<IntPair> *branch_pixels = &branches_pixels[b];
        branch_pixels->reserve(16);
        branch_pixels->push_back(start_p.toIntPair());

        // std::cout << "Tracking branch " << bLabel << " from " << start_p << " to " << end_p << std::endl;

        PointWithID current_p = start_p;
        for(const PointWithID& n : NEIGHBORHOOD){
            const Point neighbor = start_p + n;
            if(!neighbor.is_inside(H, W))
                continue;
            if(bLabels_acc[neighbor.y][neighbor.x] == bLabel){
                current_p = PointWithID(neighbor, n.id);
                break;
            }
        }

        // std::cout << "\t" << current_p;

        while(!current_p.isSamePoint(end_p) && branch_pixels->size() < 1000){
            branch_pixels->push_back(current_p.toIntPair());
            
            // Track the next pixel of the branch...
            //  (TRACK_NEXT_NEIGHBORS is used to avoid tracking back and to favor the pixels diametrically opposed to the previous one.)
            for (const int& n_id : TRACK_NEXT_NEIGHBORS[current_p.id]){
                const Point neighbor = NEIGHBORHOOD[n_id] + current_p;
                if(!neighbor.is_inside(H, W))
                    continue;
                if(bLabels_acc[neighbor.y][neighbor.x] == bLabel){
                    current_p = PointWithID(neighbor, n_id);
                    // std::cout << " -> " << current_p;
                    break;
                }
            }
            // ... and store it.
        }

        // std::cout << "->" << end_p << std::endl;

        // Append the final pixel
        branch_pixels->push_back(end_p.toIntPair());
    }

    return branches_pixels;
}

std::vector<torch::Tensor> track_branches_to_torch(const torch::Tensor& branch_labels, const torch::Tensor& node_yx, const torch::Tensor& branch_list) {
    auto branches_pixels = track_branches(branch_labels, node_yx, branch_list);

    std::vector<torch::Tensor> branches_pixels_torch;
    branches_pixels_torch.reserve(branches_pixels.size());

    for(const std::vector<IntPair>& branch : branches_pixels){
        torch::Tensor branch_tensor = torch::zeros({(long)branch.size(), 2}, torch::kInt32);
        auto branch_acc = branch_tensor.accessor<int, 2>();
        for(int i = 0; i < branch.size(); i++){
            branch_acc[i][0] = branch[i][0];
            branch_acc[i][1] = branch[i][1];
        }
        branches_pixels_torch.push_back(branch_tensor);
    }

    return branches_pixels_torch;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{   
    m.def("find_branch_endpoints", &find_branch_endpoints, "Find the first and last endpoint of each branch.");
    m.def("track_branches", &track_branches_to_torch, "Track the orientation of branches in a vessel graph.");
}
