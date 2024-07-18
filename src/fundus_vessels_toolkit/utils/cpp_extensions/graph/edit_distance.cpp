#include "edit_distance.h"

std::array<torch::Tensor, 2> shortest_secondary_path(const torch::Tensor& edge_list, const torch::Tensor& primary_nodes,
                                                     const torch::Tensor& secondary_nodes) {
    const std::size_t n_primary = primary_nodes.size(0), n_secondary = secondary_nodes.size(0);
    const std::size_t n_nodes = n_primary + n_secondary;

    auto edge_list_acc = edge_list.accessor<int, 2>();
    auto primary_acc = primary_nodes.accessor<int, 1>();
    auto secondary_acc = secondary_nodes.accessor<int, 1>();

    // Initialize node lookup table
    std::vector<int> node_lookup(n_nodes, -1);
    for (std::size_t i = 0; i < n_primary; i++) node_lookup[primary_acc[i]] = i;
    for (std::size_t i = 0; i < n_secondary; i++) node_lookup[secondary_acc[i]] = i + n_primary;

    // Initialize the distance matrix
    torch::Tensor distance_tensor = torch::full({(long)n_primary, (long)n_nodes}, -1, torch::kInt32);
    auto distance_acc = distance_tensor.accessor<int, 2>();
    for (std::size_t i = 0; i < n_nodes; i++) distance_acc[primary_acc[i]][i] = 0;

    // Initialize the backtrack matrix
    torch::Tensor backtrack_edge_node = torch::full({(long)n_primary, (long)n_nodes, 2}, -1, torch::kInt32);
    auto backtrack_edge_node_acc = backtrack_edge_node.accessor<int, 3>();

    // Initialize adjacency list
    auto const& adjacency_list = edge_list_to_adjlist(edge_list_acc, n_nodes);

    // Compute distance from primary nodes to all other nodes
    for (std::size_t p_primary_id = 0; p_primary_id < n_primary; p_primary_id++) {
        std::queue<int> to_visit, next_to_visit;
        const int p = primary_acc[p_primary_id];
        for (auto const& edge : adjacency_list[p]) {
            int neighbor = edge.other(p);
            to_visit.push(neighbor);
            backtrack_edge_node_acc[p][neighbor][0] = edge.id;
            backtrack_edge_node_acc[p][neighbor][1] = p;
        }

        int d = 1;
        while (!to_visit.empty()) {
            // Iterate over all nodes marked to visit
            while (!to_visit.empty()) {
                int node = to_visit.front();
                to_visit.pop();

                // Update the distance of the neighbor
                distance_acc[p_primary_id][node] = d;

                // If n is a primary node don't follow edges
                if (node < (int)n_primary) continue;

                // Otherwise oterate over all neighbors of the current node
                for (auto const& edge : adjacency_list[node]) {
                    // Skip if already visited or if the edge is the one that led to the current node
                    if (distance_acc[p_primary_id][edge.other(node)] != -1) continue;
                    // Update the backtrack matrix
                    backtrack_edge_node_acc[p_primary_id][edge.other(node)][0] = edge.id;
                    backtrack_edge_node_acc[p_primary_id][edge.other(node)][1] = node;
                    // Add the neighbor to the list of nodes to visit
                    next_to_visit.push(edge.other(node));
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
