#include "edit_distance.h"

/*************************************************************************************************
 *             === SHORTEST PATH ===
 ************************************************************************************************/

std::array<torch::Tensor, 2> shortest_secondary_path(const torch::Tensor& edge_list, const torch::Tensor& primary_nodes,
                                                     const std::size_t n_nodes, bool directed_edge) {
    const std::size_t n_primary = primary_nodes.size(0);

    auto edge_list_acc = edge_list.accessor<int, 2>();
    auto primary_acc = primary_nodes.accessor<int, 1>();
    auto primary_sorted = tensor_to_vector<int>(primary_nodes);
    std::sort(primary_sorted.begin(), primary_sorted.end());
    // auto secondary_acc = secondary_nodes.accessor<int, 1>();

    /*
    // Initialize node lookup table
    std::vector<int> node_lookup(n_nodes, -1);
    std::size_t i_primary = 0, i_secondary = 0;
    int primary = primary_acc[0];
    for (std::size_t i = 0; i < n_nodes; i++) {
        if (i == primary) {
            node_lookup[i_primary] = i;
            primary = ++i_primary < n_primary ? primary_acc[i_primary] : -1;
        } else {
            node_lookup[n_primary + i_secondary] = i;
            i_secondary++;
        }
    }
    // TODO: also compute inverse lookup and apply it to edge_list before computing the adjacency list.
    */

    // Initialize the distance matrix
    torch::Tensor distance_tensor = torch::full({(long)n_primary, (long)n_nodes}, -1, torch::kInt32);
    auto distance_acc = distance_tensor.accessor<int, 2>();
    for (std::size_t i = 0; i < n_primary; i++) distance_acc[i][primary_acc[i]] = 0;

    // Initialize the backtrack matrix
    torch::Tensor backtrack_edge_node = torch::full({(long)n_primary, (long)n_nodes, 2}, -1, torch::kInt32);
    auto backtrack_edge_node_acc = backtrack_edge_node.accessor<int, 3>();

    // Initialize adjacency list
    auto const& adjacency_list = edge_list_to_adjlist(edge_list_acc, n_nodes, directed_edge);

    // Compute distance from primary nodes to all other nodes
    for (std::size_t p_primary_id = 0; p_primary_id < n_primary; p_primary_id++) {
        auto p_distance_acc = distance_acc[p_primary_id];
        auto p_backtrack_acc = backtrack_edge_node_acc[p_primary_id];

        std::queue<int> to_visit;
        const int p_node = primary_acc[p_primary_id];
        for (auto const& edge : adjacency_list[p_node]) {
            int next_node = edge.other(p_node);
            to_visit.push(next_node);
            p_backtrack_acc[next_node][0] = edge.id;
            p_backtrack_acc[next_node][1] = p_node;
        }

        int d = 1;
        std::queue<int> next_to_visit;
        while (!to_visit.empty()) {
            // Iterate over all nodes marked to visit
            while (!to_visit.empty()) {
                int node = to_visit.front();
                to_visit.pop();

                // Update the distance of the neighbor
                p_distance_acc[node] = d;

                // If n is a primary node don't follow edges
                if (std::binary_search(primary_sorted.begin(), primary_sorted.end(), node)) continue;
                // if (node < (int)n_primary) continue;

                // Otherwise iterate over all neighbors of the current node
                for (auto const& edge : adjacency_list[node]) {
                    int next_node = edge.other(node);
                    // Skip if already visited or if the edge is the one that led to the current node
                    if (p_distance_acc[next_node] != -1) continue;
                    // Update the backtrack matrix
                    p_backtrack_acc[next_node][0] = edge.id;
                    p_backtrack_acc[next_node][1] = node;
                    // Add the neighbor to the list of nodes to visit
                    next_to_visit.push(next_node);
                }
            }

            // When all nodes have been visited, swap the queues ...
            std::swap(to_visit, next_to_visit);
            std::queue<int> empty;
            std::swap(next_to_visit, empty);

            // ... and increment the distance
            d++;

            // The loop stops when no more nodes were marked in next_to_visit
        }
    }
    return {distance_tensor, backtrack_edge_node};
}

std::vector<std::list<int>> backtrack_edges(const torch::Tensor& backtrack_edge_node,
                                            const torch::Tensor& src_dst_nodes, const torch::Tensor& primary_nodes) {
    auto backtrack_acc = backtrack_edge_node.accessor<int, 3>();
    auto src_dst_acc = src_dst_nodes.accessor<int, 2>();
    auto primary_acc = tensor_to_vector<int>(primary_nodes);

    const std::size_t n_paths = src_dst_nodes.size(0), n_nodes = backtrack_edge_node.size(1);
    std::vector<std::list<int>> paths(n_paths);
    for (std::size_t i = 0; i < n_paths; i++) {
        const int primary_src = src_dst_acc[i][0], dst = src_dst_acc[i][1];

        if (backtrack_acc[primary_src][dst][0] == -1) continue;  // No path between primary_src and dst

        const int src = primary_acc[primary_src];
        const auto& src_backtrack = backtrack_acc[primary_src];
        std::list<int>& path = paths[i];

        int current_node = dst;
        while (current_node != src) {
            if (current_node < 0 || current_node >= (int)n_nodes) break;
            const auto& backtrack_info = src_backtrack[current_node];
            path.push_front(backtrack_info[0]);
            current_node = backtrack_info[1];
        }
    }

    return paths;
}
