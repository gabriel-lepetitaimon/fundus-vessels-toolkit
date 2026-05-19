#include "metrics.h"

#include "skeleton.h"

/**
 * @brief Among all pairs of connected pixels in cc1, compute the proportion of pairs that are also connected in cc2.
 *
 * @param cc1 The first connected component as an integer vector of shape (N,).
 * @param cc2 The second connected component as an integer vector of shape (N,).
 * @param n_cc1 The number of pixels in cc1.
 * @param n_cc2 The number of pixels in cc2.
 *
 * @return (ratio1, n_pairs1), (ratio2, n_pairs2) where:
 * - ratio1 is the proportion of pairs of connected pixels in cc1 that are also connected in cc2, and n_pairs1 is the
 * total number of pairs of connected pixels in cc1.
 * - ratio2 is the proportion of pairs of connected pixels in cc2 that are also connected in cc1, and n_pairs2 is the
 * total number of pairs of connected pixels in cc2
 *
 */
std::array<std::pair<double, long>, 2> valid_path_ratio(const torch::Tensor& cc1, const torch::Tensor& cc2, long n_cc1,
                                                        long n_cc2) {
    auto cc1_acc = cc1.accessor<int32_t, 1>();
    auto cc2_acc = cc2.accessor<int32_t, 1>();
    std::vector<std::vector<int>> count_cc1_cc2(n_cc1, std::vector<int>(n_cc2, 0));
    std::vector<int> count_cc1(n_cc1, 0);
    std::vector<int> count_cc2(n_cc2, 0);
    for (int i = 0; i < cc1.size(0); i++) {
        const auto &cc1_id = cc1_acc[i], cc2_id = cc2_acc[i];
        if (cc1_id > 0) {
            count_cc1[cc1_id - 1]++;
            if (cc2_id > 0) {
                count_cc2[cc2_id - 1]++;
                count_cc1_cc2[cc1_id - 1][cc2_id - 1]++;
            }
        } else if (cc2_id > 0)
            count_cc2[cc2_id - 1]++;
    }

    long n_pairs_cc1 = 0, n_pairs_cc2 = 0, n_valid_pairs = 0;
    for (long cc1_i = 0; cc1_i < n_cc1; cc1_i++) {
        const auto& n = count_cc1[cc1_i];
        n_pairs_cc1 += n * (n - 1) / 2;

        for (long cc2_i = 0; cc2_i < n_cc2; cc2_i++) {
            const auto& k = count_cc1_cc2[cc1_i][cc2_i];
            n_valid_pairs += k * (k - 1) / 2;
        }
    }
    for (long cc2_i = 0; cc2_i < n_cc2; cc2_i++) {
        const auto& n = count_cc2[cc2_i];
        n_pairs_cc2 += n * (n - 1) / 2;
    }

    double ratio_cc1 = n_pairs_cc1 > 0 ? (double)n_valid_pairs / n_pairs_cc1 : 1;
    double ratio_cc2 = n_pairs_cc2 > 0 ? (double)n_valid_pairs / n_pairs_cc2 : 1;
    return {{{ratio_cc1, n_pairs_cc1}, {ratio_cc2, n_pairs_cc2}}};
}

/**
 * @brief Compute the shortest path length between all pairs of branches in the skeleton.
 *
 * @param skeleton The skeleton of the vessel graph as a boolean tensor of shape (H, W).
 *
 * @return A tuple containing:
 *  - p_yx: an integer tensor of shape (P) containing the coordinates of each pixel.
 *  - p_branch: an integer tensor of shape (P) containing the indices of the branches to which each pixel belongs.
 *  - p_pos: a float tensor of shape (P) containing the distance of each pixel to the start of its branch.
 *  - branch_length: a float tensor of shape (B) mapping each branch index to its length.
 *  - shortest_path: a float tensor of shape (B,B,2,2) containing the shortest path length between each pair of branches
 * start and end points. The distance between a branch and itself is 0, and the distance between two branches that are
 * not connected is +inf. The cell (b1, b2, 0, 0) contains the shortest path length between the start point of b1 and
 * the start point of b2, (b1, b2, 0, 1) contains the shortest path length between the start point of b1 and the end
 * point of b2, etc.
 *  - branch_subgraph: a integer tensor of shape (B,) containing the index of the subgraph to which each branch belongs.
 * Two branches belong to the same subgraph if they are connected by a path in the skeleton.
 *
 * The shortest path between two pixels at coord yx1 and yx2 of the skeleton can therefore be computed as:
 * p1 = skel_id[yx1], p2 = skel_id[yx2]
 * b1 = branch[p1], b2 = branch[p2]
 * p1_pos = p_pos[p1]
 * p2_pos = p_pos[p2]
 * shortest_paths = shortest_path[b1][b2]
 * shortest_paths[0,:] += p1_pos
 * shortest_paths[1,:] += branch_length[b1] - p1_pos
 * shortest_paths[:,0] += p2_pos
 * shortest_paths[:,1] += branch_length[b2] - p2_pos
 * shortest_path_length = min(shortest_paths)
 */
std::array<torch::Tensor, 6> shortest_skeleton_path_length(torch::Tensor& skeleton) {
    const float INF = std::numeric_limits<float>::infinity();

    // === Parse graph from skeleton ===
    const auto [edge_list, curves, node_yx] = parse_skeleton_to_graph(skeleton);
    for (const auto& yx : node_yx) skeleton[yx.y][yx.x] = 0;  // remove nodes from skeleton to get clean branches

    std::size_t P = 0, B = curves.size(), N = node_yx.size();
    for (auto& curve : curves) P += curve.size();

    // === Analyze branches curves ===
    // measure branch lengths and assign each pixel to a branch and a distance to the branch midpoint
    torch::Tensor p_yx = torch::empty({(long)P, 2}, torch::kInt);
    torch::Tensor p_branch = torch::full(P, -1, torch::kInt);
    torch::Tensor p_pos = torch::empty(P, torch::kFloat32);
    torch::Tensor branch_len = torch::zeros(B, torch::kFloat32);

    auto p_yx_acc = p_yx.accessor<int, 2>();
    auto p_branch_acc = p_branch.accessor<int, 1>();
    auto p_pos_acc = p_pos.accessor<float, 1>();
    auto branch_len_acc = branch_len.accessor<float, 1>();

    for (std::size_t b = 0, p = 0; b < B; b++) {
        const auto& curve = curves[b];
        float& length = branch_len_acc[b];
        for (std::size_t i = 1; i < curve.size(); i++, p++) {
            const auto& yx = curve[i];
            p_yx_acc[p][0] = yx.y;
            p_yx_acc[p][1] = yx.x;
            skeleton[yx.y][yx.x] = p;  // assign pixel to id p
            p_pos_acc[p] = length;
            p_branch_acc[p] = b;
            length += distance(yx, curve[i - 1]);
        }
    }

    // === Shortest path between branches ===
    torch::Tensor shortest_node_path = torch::full({(long)N, (long)N}, INF, torch::kFloat32);
    auto shortest_node_path_acc = shortest_node_path.accessor<float, 2>();
    // Initialize distances between adjacent nodes and self-distances
    for (std::size_t n = 0; n < N; n++) shortest_node_path_acc[n][n] = 0;
    for (const auto& e : edge_list)
        shortest_node_path_acc[e.start][e.end] = shortest_node_path_acc[e.end][e.start] = branch_len_acc[e.id];

    // Compute shortest paths between all pairs of branches using Floyd–Warshall algorithm
    for (std::size_t k = 0; k < N; k++) {
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < N; j++) {
                float new_dist = shortest_node_path_acc[i][k] + shortest_node_path_acc[k][j];
                if (new_dist < shortest_node_path_acc[i][j]) shortest_node_path_acc[i][j] = new_dist;
            }
        }
    }

    // Reshape into shortest path between branches start and end points
    std::vector<std::set<std::size_t>> branch_adj_list(B);
    torch::Tensor shortest_path = torch::full({(long)B, (long)B, 2, 2}, INF, torch::kFloat32);
    auto shortest_path_acc = shortest_path.accessor<float, 4>();
    for (std::size_t b1 = 0; b1 < B; b1++) {
        const auto &b1_n1 = edge_list[b1].start, &b1_n2 = edge_list[b1].end;
        for (std::size_t b2 = 0; b2 < B; b2++) {
            const auto &b2_n1 = edge_list[b2].start, &b2_n2 = edge_list[b2].end;
            if (!(shortest_node_path_acc[b1_n1][b2_n1] < INF)) continue;  // if branches are not connected, keep +inf
            shortest_path_acc[b1][b2][0][0] = shortest_node_path_acc[b1_n1][b2_n1];
            shortest_path_acc[b1][b2][0][1] = shortest_node_path_acc[b1_n1][b2_n2];
            shortest_path_acc[b1][b2][1][0] = shortest_node_path_acc[b1_n2][b2_n1];
            shortest_path_acc[b1][b2][1][1] = shortest_node_path_acc[b1_n2][b2_n2];
            branch_adj_list[b1].insert(b2);
            branch_adj_list[b2].insert(b1);
        }
    }

    // Solve connected components of branches to get branch subgraph
    torch::Tensor branch_subgraph = torch::full(B, -1, torch::kInt);
    auto branch_subgraph_acc = branch_subgraph.accessor<int, 1>();

    int subgraph_i = 0;
    for (std::size_t b = 0; b < B; b++) {
        if (branch_subgraph_acc[b] != -1) continue;  // already visited

        std::queue<std::size_t> q({b});
        do {
            std::size_t b0 = q.front();
            q.pop();
            if (branch_subgraph_acc[b0] != -1) continue;  // already visited
            branch_subgraph_acc[b0] = subgraph_i;
            for (const auto& other_b : branch_adj_list[b0]) q.push(other_b);
        } while (!q.empty());

        subgraph_i++;
    }

    return {p_yx, p_branch, p_pos, branch_len, shortest_path, branch_subgraph};
}
