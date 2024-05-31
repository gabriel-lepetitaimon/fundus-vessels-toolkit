#include "common.h"
#include <pybind11/pybind11.h>
#include <set>

std::vector<std::set<int>> edges_to_adjacency_list(std::list<std::vector<int>> edges_list, int n_nodes=-1) {
    if (n_nodes == -1) {
        n_nodes = 0;
        for (auto edge : edges_list) {
            n_nodes = std::max(n_nodes, std::max(edge[0], edge[1]));
        }
        n_nodes++;
    }

    std::vector<std::set<int>> adjacency_list(n_nodes);
    for (auto edge : edges_list) {
        int u = edge[0];
        int v = edge[1];
        adjacency_list[u].insert(v);
        adjacency_list[v].insert(u);
    }
    return adjacency_list;
}

std::list<std::vector<int>> solve_clusters(std::list<std::vector<int>> edges_list) {
    // Create adjacency list from edges list
    const auto adjacency_list = edges_to_adjacency_list(edges_list);

    // Initialize variables
    int n_nodes = adjacency_list.size();
    std::vector<bool> visited(n_nodes, false);
    std::list<std::vector<int>> clusters;

    // DFS to find connected components
    for (int i = 0; i < n_nodes; i++) {
        if (visited[i]) {
            continue;
        }

        std::vector<int> cluster;
        std::stack<int> stack;
        stack.push(i);
        visited[i] = true;

        while (!stack.empty()) {
            int u = stack.top();
            stack.pop();
            cluster.push_back(u);

            for (int v : adjacency_list[u]) {
                if (!visited[v]) {
                    stack.push(v);
                    visited[v] = true;
                }
            }
        }

        clusters.push_back(cluster);
    }

    return clusters;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve_clusters", &solve_clusters, "Solve clusters from edges list");
}
