#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <set>

template <typename T>
using Tensor2DAcc = at::TensorAccessor<T, 2UL, at::DefaultPtrTraits, signed long>;

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

/*****************************************************************************
 *                    === REDUCE CLUSTERS ===
 *****************************************************************************/
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

std::list<std::vector<int>> iterative_reduce_clusters(const torch::Tensor &edgeList, const torch::Tensor &edgeWeight,
                                                      float maxWeight) {
    // Sort edges by weight
    auto const &argsort = torch::argsort(edgeWeight, 0, false);
    auto const &sortedWeights = edgeWeight.index_select(0, argsort);

    // Remove any edge with weight > maxWeight
    auto const &weight = sortedWeights.accessor<float, 1>();
    int nEdges = weight.size(0);
    for (; nEdges > 0; nEdges--) {
        if (weight[nEdges - 1] <= maxWeight) break;
    }
    if (nEdges == 0) return {};

    // Sort and select edges
    auto const &sortedEdges = edgeList.index_select(0, argsort);
    auto const &edges = sortedEdges.accessor<int, 2>();
    std::list<std::tuple<int, int, float, float>> sortedEdgesList;
    for (int i = 0; i < nEdges; i++) sortedEdgesList.push_back({edges[i][0], edges[i][1], weight[i], weight[i]});

    // Reduce clusters
    std::map<int, std::pair<int, float>> nodeToCluster;  // {node: (clusterId, weightToClusterCenter)}
    std::vector<std::list<int>> clusters;                // List of clusters of nodeId

    auto update_edges_total_weight = [&](int nodeID, float deltaW) {
        for (auto it = sortedEdgesList.begin(); it != sortedEdgesList.end(); it++) {
            if (std::get<0>(*it) == nodeID || std::get<1>(*it) == nodeID) {
                const float totalW = (std::get<3>(*it) += deltaW);
                if (totalW > maxWeight) it = sortedEdgesList.erase(it);
            }
        }
    };

    while (!sortedEdgesList.empty()) {
        auto [u, v, w_uv, totalW_uv] = sortedEdgesList.front();
        sortedEdgesList.pop_front();

        auto itU = nodeToCluster.find(u);
        auto itV = nodeToCluster.find(v);

        if (itU == nodeToCluster.end() && itV == nodeToCluster.end()) {
            // If neither u nor v were already added to the clusters, create a new cluster
            int clusterId = clusters.size();
            clusters.push_back({u, v});
            // The weight to the cluster center is half the edge weight
            nodeToCluster[u] = {clusterId, w_uv / 2};
            nodeToCluster[v] = {clusterId, w_uv / 2};

        } else if (itU == nodeToCluster.end() || itV == nodeToCluster.end()) {
            if (itU == nodeToCluster.end()) {
                std::swap(u, v);
                std::swap(itU, itV);
            }

            // If only u was added to the clusters, add v to u's cluster
            int clusterId = std::get<0>(itU->second);
            float w_u = std::get<1>(itU->second);
            std::size_t n = clusters[clusterId].size();
            // Compute weight of v to the cluster and check that it is below the threshold
            float w_v = (w_uv + w_u) * n / (n + 1);
            if (w_v > maxWeight) continue;
            // Compute the delta weight of all nodes already in the cluster (except u) and check threshold
            float deltaW = w_u - (w_v - w_uv);
            bool valid = true;
            for (int nodeId : clusters[clusterId]) {
                if (nodeId == u) continue;
                float totalW = std::get<1>(nodeToCluster[nodeId]) + deltaW;
                if (totalW > maxWeight) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;  // If the threshold was exceeded, don't cluster v with u

            // Update weight of u
            float w_u_new = abs(w_v - w_uv);
            update_edges_total_weight(u, w_u_new - w_u);
            nodeToCluster[u] = {clusterId, w_u_new};
            w_u = w_u_new;

            // Update weight of existing nodes
            for (int nodeId : clusters[clusterId]) {
                auto [i, w_node] = nodeToCluster[nodeId];
                nodeToCluster[nodeId] = {i, w_node + deltaW};
                update_edges_total_weight(nodeId, deltaW);
            }
            // Add v to cluster
            clusters[clusterId].push_back(v);
            nodeToCluster[v] = {clusterId, w_v};
            update_edges_total_weight(v, w_v);

        } else if (std::get<0>(itU->second) != std::get<0>(itV->second)) {
            // Merge the smallest cluster (cluster of v) into the largest (cluster of u)
            if (clusters[std::get<0>(itU->second)].size() < clusters[std::get<0>(itV->second)].size()) {
                std::swap(u, v);
                std::swap(itU, itV);
            }

            int cluster1Id = std::get<0>(itU->second), cluster2Id = std::get<0>(itV->second);
            auto &cluster1 = clusters[cluster1Id], &cluster2 = clusters[cluster2Id];

            // Compute delta W for all nodes in cluster 1 and 2 (except u and v)
            float w_u = std::get<1>(itU->second), w_v = std::get<1>(itV->second);
            int n1 = cluster1.size(), n2 = cluster2.size();
            float deltaW1 = (w_u + w_uv) * n2 / (n1 + n2);
            float deltaW2 = (w_v + w_uv) * n1 / (n1 + n2);
            // Check that the new weight of all nodes in cluster 1 and 2 are below the threshold
            bool valid = true;
            for (int nodeId : cluster1) {
                if (nodeId == u) continue;
                float totalW = std::get<1>(nodeToCluster[nodeId]) + deltaW1;
                if (totalW > maxWeight) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;
            for (int nodeId : cluster2) {
                if (nodeId == v) continue;
                float totalW = std::get<1>(nodeToCluster[nodeId]) + deltaW2;
                if (totalW > maxWeight) {
                    valid = false;
                    break;
                }
            }
            if (!valid) continue;

            // Update weight in cluster1 (largest)
            for (int nodeId : cluster1) {
                if (nodeId == u) continue;
                auto [i, w_node] = nodeToCluster[nodeId];
                nodeToCluster[nodeId] = {i, w_node + deltaW1};
                update_edges_total_weight(nodeId, deltaW1);
            }
            // Update weight of u
            float w_u_new = abs(w_u - deltaW1);
            nodeToCluster[u] = {cluster1Id, w_u_new};
            update_edges_total_weight(u, w_u_new - w_u);

            // Update weight and cluster of v
            float w_v_new = abs(w_v - deltaW2);
            nodeToCluster[v] = {cluster1Id, w_v_new};
            cluster1.push_back(v);
            update_edges_total_weight(v, w_v_new - w_v);

            // Update weight and cluster of nodes in cluster2 (smallest)
            for (int nodeId : cluster2) {
                if (nodeId == v) continue;
                auto [i, w_node] = nodeToCluster[nodeId];
                nodeToCluster[nodeId] = {cluster1Id, w_node + deltaW2};
                cluster1.push_back(nodeId);
                update_edges_total_weight(nodeId, deltaW2);
            }
            cluster2.clear();
        } else {
            continue;
        }

        // Sort edge list by total weight
        sortedEdgesList.sort([](auto const &a, auto const &b) { return std::get<3>(a) < std::get<3>(b); });
    }

    // Remove empty clusters
    std::list<std::vector<int>> nonEmptyClusters;
    for (auto const &cluster : clusters) {
        if (!cluster.empty()) nonEmptyClusters.push_back(std::vector<int>(cluster.begin(), cluster.end()));
    }

    return nonEmptyClusters;
}

/*****************************************************************************
 *                    === CLUSTERS BY DISTANCE ===
 *****************************************************************************/
typedef std::tuple<float, int, int> EdgeWithDistance;

struct Point {
    float y, x;
};
struct Cluster {
    std::set<int> nodes;
    Point pos;
};

float distance(const at::TensorAccessor<float, 1UL, at::DefaultPtrTraits, signed long> &p1,
               const at::TensorAccessor<float, 1UL, at::DefaultPtrTraits, signed long> &p2) {
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
}
float distance(const Cluster &c1, const Cluster &c2) {
    return sqrt(pow(c1.pos.x - c2.pos.x, 2) + pow(c1.pos.y - c2.pos.y, 2));
}

Cluster merge_clusters(const Cluster &c1, const Cluster &c2) {
    Cluster merged;
    int n1 = c1.nodes.size(), n2 = c2.nodes.size();
    merged.nodes = c1.nodes;
    merged.nodes.insert(c2.nodes.begin(), c2.nodes.end());
    merged.pos.x = (c1.pos.x * n1 + c2.pos.x * n2) / (n1 + n2);
    merged.pos.y = (c1.pos.y * n1 + c2.pos.y * n2) / (n1 + n2);
    return merged;
}

std::list<std::set<int>> iterative_cluster_by_distance(torch::Tensor pos, float max_distance, torch::Tensor edge_list) {
    TORCH_CHECK_VALUE(pos.dim() == 2 && pos.size(1) == 2, "Input tensor must be of shape (N, 2).");
    auto const &pos_acc = pos.accessor<float, 2>();
    int N = (int)pos.size(0);

    TORCH_CHECK_VALUE(edge_list.dim() == 2 && edge_list.size(1) == 2, "Edge list must be of shape (E, 2).");

    std::priority_queue<EdgeWithDistance, std::vector<EdgeWithDistance>, std::greater<EdgeWithDistance>> edges;
    if (edge_list.size(0) == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                float edge_dist = distance(pos_acc[i], pos_acc[j]);
                if (edge_dist <= max_distance) edges.push({edge_dist, i, j});
            }
        }
    } else {
        auto const &edgeAccessor = edge_list.accessor<int, 2>();
        for (int i = 0; i < edge_list.size(0); i++) {
            int u = edgeAccessor[i][0], v = edgeAccessor[i][1];
            if (u > v) std::swap(u, v);
            float edge_dist = distance(pos_acc[u], pos_acc[v]);
            if (edge_dist <= max_distance) edges.push({edge_dist, u, v});
        }
    }

    std::list<Cluster> clusters;
    for (int i = 0; i < N; i++) clusters.push_back({{i}, {pos_acc[i][0], pos_acc[i][1]}});

    while (!edges.empty()) {
        auto [dist, u, v] = edges.top();
        edges.pop();

        auto itU = clusters.begin(), itV = clusters.begin();
        for (; itU != clusters.end(); itU++) {
            if (itU->nodes.find(u) != itU->nodes.end()) break;
        }
        for (; itV != clusters.end(); itV++) {
            if (itV->nodes.find(v) != itV->nodes.end()) break;
        }
        if (itU == itV) continue;

        if (itU->nodes.size() < itV->nodes.size()) std::swap(itU, itV);
        if (distance(*itU, *itV) > max_distance) continue;
        *itU = merge_clusters(*itU, *itV);
        clusters.erase(itV);
    }

    std::list<std::set<int>> clustersList;
    for (auto const &cluster : clusters) clustersList.push_back(cluster.nodes);
    return clustersList;
}

/*****************************************************************************
 *                    === REDUCE CHAINS ===
 *****************************************************************************/
std::list<std::list<int>> solve_1d_chains(std::list<std::vector<int>> chains) {
    std::list<std::list<int>> solved_chains;
    for (auto chain : chains) {
        // Ensure each index is not part of the middle of another chain
        for (auto solved_chain : solved_chains) {
            auto it = solved_chain.begin();
            it++;
            auto penultimate = solved_chain.end();
            penultimate--;

            while (it != penultimate) {
                TORCH_CHECK_VALUE(std::find(chain.begin(), chain.end(), *it) == chain.end(), "Index ", *it,
                                  " is part of multiple chains.");
                it++;
            }
        }

        int chain_found = 0;
        // Try to append chain to existing solved chains
        auto it = solved_chains.begin();
        for (; it != solved_chains.end(); it++) {
            auto &solved_chain = *it;
            auto const &front = chain.front();
            auto const &back = chain.back();
            auto const &solved_begin = solved_chain.begin();
            auto const &solved_back = solved_chain.back();

            if (solved_back == front) {
                solved_chain.insert(solved_chain.end(), chain.begin() + 1, chain.end());
                chain_found = 1;
                break;
            } else if (solved_back == back) {
                solved_chain.insert(solved_chain.end(), chain.rbegin() + 1, chain.rend());
                chain_found = 1;
                break;
            } else if (*solved_begin == back) {
                solved_chain.insert(solved_begin, chain.begin(), chain.end() - 1);
                chain_found = -1;
                break;
            } else if (*solved_begin == front) {
                solved_chain.insert(solved_begin, chain.rbegin(), chain.rend() - 1);
                chain_found = -1;
                break;
            }
        }
        if (!chain_found)
            solved_chains.push_back(std::list<int>(chain.begin(), chain.end()));
        else {
            // If a chain was found, ensure that, if the new extremity is a part of another chain, the chains are merged
            auto &chain = *it;
            const int extremity = chain_found == 1 ? chain.back() : chain.front();
            auto other_it = it;
            other_it++;
            if (chain_found == 1) {
                for (; other_it != solved_chains.end(); other_it++) {
                    if (other_it->front() == extremity) {
                        chain.insert(chain.end(), ++other_it->begin(), other_it->end());
                        solved_chains.erase(other_it);
                        break;
                    } else if (other_it->back() == extremity) {
                        chain.insert(chain.end(), ++other_it->rbegin(), other_it->rend());
                        solved_chains.erase(other_it);
                        break;
                    }
                }
            } else {
                for (; other_it != solved_chains.end(); other_it++) {
                    if (other_it->front() == extremity) {
                        chain.insert(chain.begin(), other_it->rbegin(), --other_it->rend());
                        solved_chains.erase(other_it);
                        break;
                    } else if (other_it->back() == extremity) {
                        chain.insert(chain.begin(), other_it->begin(), --other_it->end());
                        solved_chains.erase(other_it);
                        break;
                    }
                }
            }
        }
    }

    return solved_chains;
}

std::vector<torch::Tensor> remove_consecutive_duplicates(const torch::Tensor &tensor, bool return_index = false) {
    TORCH_CHECK_VALUE(tensor.dim() == 2 && tensor.size(0) > 0, "Input tensor must be 2D and non empty.");
    std::size_t K = (std::size_t)tensor.size(1);
    auto outTensor = torch::empty({tensor.size(0), tensor.size(1)}, tensor.options());
    auto indexTensor = torch::empty({tensor.size(0)}, tensor.options().dtype(torch::kInt32));
    auto out = outTensor.accessor<int, 2>(), acc = tensor.accessor<int, 2>();
    auto id = indexTensor.accessor<int, 1>();

    for (std::size_t k = 0; k < K; k++) out[0][k] = acc[0][k];
    std::size_t j = 1;
    for (std::size_t i = 1; i < (std::size_t)tensor.size(0); i++) {
        for (std::size_t k = 0; k < K; k++) {
            if (acc[i][k] != acc[i - 1][k]) {
                for (std::size_t l = 0; l < K; l++) out[j][l] = acc[i][l];
                id[j] = i;
                j++;
                break;
            }
        }
    }

    outTensor = outTensor.slice(0, 0, j);
    if (return_index) {
        indexTensor = indexTensor.slice(0, 0, j);
        return {outTensor, indexTensor};
    } else
        return {outTensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("solve_clusters", &solve_clusters, "Solve clusters from edges list");
    m.def("iterative_reduce_clusters", &iterative_reduce_clusters, "Iteratively reduce clusters from edge list");
    m.def("iterative_cluster_by_distance", &iterative_cluster_by_distance, "Iteratively cluster by distance");
    m.def("solve_1d_chains", &solve_1d_chains, "Solve 1D chains clusters from chains list");
    m.def("remove_consecutive_duplicates", &remove_consecutive_duplicates, "Remove consecutive duplicates from tensor");
}
