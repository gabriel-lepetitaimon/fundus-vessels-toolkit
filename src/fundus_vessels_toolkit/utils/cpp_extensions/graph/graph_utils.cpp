#include "graph.h"

EdgeList terminal_edges(const EdgeList &edgeList) {
    EdgeList terminalEdges;
    terminalEdges.reserve(edgeList.size());

    auto const &nodesRank = nodes_rank(edgeList);
    for (const auto &edge : edgeList) {
        if (nodesRank[edge.start] == 1 || nodesRank[edge.end] == 1) terminalEdges.push_back(edge);
    }
    return terminalEdges;
}

EdgeList terminal_edges(const GraphAdjList &adjacency) {
    EdgeList terminalEdges;
    terminalEdges.reserve(adjacency.size());

    auto const &nodesRank = nodes_rank(adjacency);
    for (int i = 0; i < (int)adjacency.size(); i++) {
        for (const auto &edge : adjacency[i]) {
            if (edge.start == i && nodesRank[edge.end] == 1) terminalEdges.push_back(edge);
        }
    }
    return terminalEdges;
}

std::vector<int> nodes_rank(const GraphAdjList &adjacency) {
    std::vector<int> nodesRank;
    nodesRank.reserve(adjacency.size());
    for (const auto &nodes_adjacence : adjacency) nodesRank.push_back((int)nodes_adjacence.size());
    return nodesRank;
}

std::vector<int> nodes_rank(const EdgeList &edgeList) {
    std::vector<int> nodesRank;
    nodesRank.reserve(edgeList.size());

    for (const auto &edge : edgeList) {
        int maxEdgeID = std::max(edge.start, edge.end);
        if (maxEdgeID > (int)nodesRank.size()) nodesRank.resize(maxEdgeID);

        nodesRank[edge.start]++;
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