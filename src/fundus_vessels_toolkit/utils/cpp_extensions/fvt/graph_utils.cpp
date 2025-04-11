#include "graph.h"

EdgeList terminal_edges(const EdgeList &edgeList, bool directed) {
    EdgeList terminalEdges;
    terminalEdges.reserve(edgeList.size());

    auto const &nodesRank = nodes_rank(edgeList, directed);
    for (const auto &edge : edgeList) {
        if (nodesRank[edge.start] == 1 || nodesRank[edge.end] == 1) terminalEdges.push_back(edge);
    }
    return terminalEdges;
}

EdgeList terminal_edges(const GraphAdjList &adjacency) {
    EdgeList terminalEdges;
    terminalEdges.reserve(adjacency.size());

    auto const &nodesRank = nodes_rank(adjacency);
    for (int nodeId = 0; nodeId < (int)adjacency.size(); nodeId++) {
        if (nodesRank[nodeId] == 1) terminalEdges.push_back(*(adjacency[nodeId].begin()));
    }
    return terminalEdges;
}

std::vector<int> nodes_rank(const GraphAdjList &adjacency) {
    std::vector<int> nodesRank;
    nodesRank.reserve(adjacency.size());
    for (const auto &nodes_adjacence : adjacency) nodesRank.push_back((int)nodes_adjacence.size());
    return nodesRank;
}

std::vector<int> nodes_rank(const EdgeList &edgeList, bool onlyIncoming) {
    std::vector<int> nodesRank;
    nodesRank.reserve(edgeList.size());

    for (const auto &edge : edgeList) {
        int maxEdgeID = std::max(edge.start, edge.end);
        if (maxEdgeID >= (int)nodesRank.size()) nodesRank.resize(maxEdgeID + 1, 0);

        if (!onlyIncoming) nodesRank[edge.start]++;
        nodesRank[edge.end]++;
    }
    return nodesRank;
}

std::size_t nodesCount(const EdgeList &edgeList) {
    std::size_t nodesCount = 0;
    for (const auto &edge : edgeList)
        nodesCount = std::max({nodesCount, (std::size_t)edge.start, (std::size_t)edge.end});
    return nodesCount + 1;
}

std::vector<std::list<int>> connected_components(const EdgeList &edgeList, int N) {
    GraphAdjList adjList = edge_list_to_adjlist(edgeList, N);
    return connected_components(adjList);
}

std::vector<std::list<int>> connected_components(const GraphAdjList &adjList) {
    std::vector<bool> node_assigned(adjList.size(), false);
    std::vector<std::list<int>> components;

    for (int i = 0; i < (int)adjList.size(); i++) {
        if (node_assigned[i]) continue;

        components.push_back({i});
        auto &component = components.back();

        std::queue<int> nodesQueue;
        nodesQueue.push(i);
        node_assigned[i] = true;

        while (!nodesQueue.empty()) {
            int node = nodesQueue.front();
            nodesQueue.pop();

            for (const auto &neighbor : adjList[node]) {
                if (node_assigned[neighbor.end]) continue;
                node_assigned[neighbor.end] = true;
                component.push_back(neighbor.end);
                nodesQueue.push(neighbor.end);
            }
        }
    }

    return components;
}

/***************************************************************************
 *             === Maximum Weighted Independent Set ===
 ***************************************************************************/
std::pair<std::vector<int>, float> recursive_MWIS(const std::vector<std::set<int>> &adjList,
                                                  const std::vector<float> &weights, std::list<int> subset) {
    if (subset.empty()) return {{}, 0};
    if (subset.size() == 1) return {{subset.front()}, weights[subset.front()]};

    int node = subset.front();
    const float &nodeWeight = weights[node];
    subset.pop_front();
    const auto &neighbors = adjList[node];

    // == Remove the node from the subset and compute the MWIS ===
    auto [set1, weight1] = recursive_MWIS(adjList, weights, subset);

    // == If the node is not connected to any other node in the subset, return the MWIS ==
    std::size_t N = subset.size();
    subset.remove_if([&neighbors](int n) { return neighbors.find(n) != neighbors.end(); });
    if (N == subset.size()) {
        set1.push_back(node);
        return {set1, weight1 + nodeWeight};
    }

    // == Otherwise, compute the MWIS on the subset without the neighbors ==
    auto [set2, weight2] = recursive_MWIS(adjList, weights, subset);
    weight2 += nodeWeight;
    // If the MWIS on the subset containing the neighbors is the best, return it
    if (weight1 > weight2 || (weight1 == weight2 && set1.size() <= set2.size() + 1)) return {set1, weight1};

    // Otherwise, add back the node to the subset and return the MWIS
    set2.push_back(node);
    return {set2, weight2};
}

std::vector<int> maximum_weighted_independent_set(std::vector<IntPair> edges_list, std::vector<float> weights) {
    const auto &adjList = edge_list_to_adjlist(edges_list, weights.size(), false, false);
    const auto &cc = connected_components(adjList);

    std::vector<std::set<int>> simpleAdjList(adjList.size());
    for (const auto &edges : adjList) {
        for (const auto &edge : edges) simpleAdjList[edge.start].insert(edge.end);
    }

    std::vector<int> mwis;
#pragma omp parallel for reduction(merge : mwis) schedule(dynamic)
    for (auto component : cc) {
        component.sort([&simpleAdjList](int a, int b) { return simpleAdjList[a].size() > simpleAdjList[b].size(); });
        auto [set, weight] = recursive_MWIS(simpleAdjList, weights, component);
        mwis.insert(mwis.end(), set.begin(), set.end());
    }

    return mwis;
}
