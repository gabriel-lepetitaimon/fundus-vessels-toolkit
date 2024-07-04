#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <set>

std::vector<std::set<int>> edges_to_adjacency_list(std::list<std::vector<int>> edges_list, int n_nodes = -1) {
    if (n_nodes == -1) {
        n_nodes = 0;
        // Find maximum node index
        for (auto edge : edges_list) n_nodes = std::max(n_nodes, std::max(edge[0], edge[1]));

        // n_nodes = max_node_id + 1
        n_nodes++;
    }

    std::vector<std::set<int>> adjacency_list(n_nodes);
    for (auto edge : edges_list) {
        for (int u : edge) {
            for (int v : edge)
                if (u != v) adjacency_list[u].insert(v);
        }
    }
    return adjacency_list;
}

std::list<std::vector<int>> solve_clusters(std::list<std::vector<int>> edges_list, bool drop_singletons = true) {
    // Find number of nodes
    int n_nodes = 0;
    for (auto edge : edges_list) {
        for (int v : edge) n_nodes = std::max(n_nodes, v);
    }
    n_nodes++;

    // Create adjacency list from edges list
    std::vector<std::set<int>> adjacency_list(n_nodes);
    for (auto edge : edges_list) {
        int u = edge[0];
        for (auto v = edge.begin() + 1; v != edge.end(); v++) {
            adjacency_list[u].insert(*v);
            adjacency_list[*v].insert(u);
        }
    }

    // DFS to find connected components
    std::vector<bool> visited(n_nodes, false);
    std::list<std::vector<int>> clusters;
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

        if (!drop_singletons || cluster.size() > 1) clusters.push_back(cluster);
    }

    return clusters;
}

std::list<std::list<int>> solve_1d_chains(std::list<std::vector<int>> chains) {
    std::list<std::list<int>> solved_chains;
    for (auto chain : chains) {
        // Ensure unique elements
        for (auto solved_chain : solved_chains) {
            auto it = solved_chain.begin();
            auto const &penultimate = --solved_chain.end();
            while (++it != penultimate) {
                if (std::find(chain.begin(), chain.end(), *it) != chain.end()) {
                    PyErr_SetString(PyExc_ValueError, "Chain contains duplicate elements");
                    return {};
                }
            }
        }

        bool chain_found = false;
        // Try to append chain to existing solved chains
        for (auto &solved_chain : solved_chains) {
            auto const &front = chain.front();
            auto const &back = chain.back();
            auto const &solved_begin = solved_chain.begin();
            auto const &solved_back = solved_chain.back();

            if (solved_back == front) {
                solved_chain.insert(solved_chain.end(), chain.begin() + 1, chain.end());
                chain_found = true;
                break;
            } else if (solved_back == back) {
                solved_chain.insert(solved_chain.end(), chain.rbegin() + 1, chain.rend());
                chain_found = true;
                break;
            } else if (*solved_begin == back) {
                solved_chain.insert(solved_begin, chain.begin(), chain.end() - 1);
                chain_found = true;
                break;
            } else if (*solved_begin == front) {
                solved_chain.insert(solved_begin, chain.rbegin(), chain.rend() - 1);
                chain_found = true;
                break;
            }
        }
        if (!chain_found) solved_chains.push_back(std::list<int>(chain.begin(), chain.end()));
    }

    return solved_chains;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve_clusters", &solve_clusters, "Solve clusters from edges list");
    m.def("solve_1d_chains", &solve_1d_chains, "Solve 1D chains clusters from chains list");
}
