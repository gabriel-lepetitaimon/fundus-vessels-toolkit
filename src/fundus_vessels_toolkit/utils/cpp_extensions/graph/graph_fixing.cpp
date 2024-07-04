#include "graph.h"

void remove_branches(std::vector<Edge> branchesToRemove, std::vector<CurveYX> &branchCurves,
                     Tensor2DAccessor<int> &branchesLabelMap, EdgeList &edge_list) {
    // Ensure the branchesToRemove are sorted by branch ID.
    std::sort(branchesToRemove.begin(), branchesToRemove.end(),
              [](const Edge &a, const Edge &b) { return a.id < b.id; });

    auto itToRemove = branchesToRemove.begin();
    std::size_t nBranches = 0;
    for (std::size_t i = 0; i < branchCurves.size(); i++) {
        if (itToRemove != branchesToRemove.end() && itToRemove->id == (int)i) {
            // If the branch is to be removed:
            //  - Remove the branch from the branchesLabelMap
            auto const &curve = branchCurves[i];
            for (auto const &p : curve) branchesLabelMap[p.y][p.x] = 0;
            //  - Increment the iterator to search for the next branch to remove
            itToRemove++;
        } else {
            // If the branch is not to be removed:
            //  - Copy the corresponding curve and edge to their new position
            branchCurves[nBranches] = branchCurves[i];
            edge_list[nBranches] = edge_list[i];
            edge_list[nBranches].id = nBranches;
            nBranches++;
        }
    }

    branchCurves.resize(nBranches);
    edge_list.resize(nBranches);
}

void remove_singleton_nodes(EdgeList &edge_list, std::vector<IntPoint> &nodeCoords, Tensor2DAccessor<int> &labelMap) {
    auto const N = nodeCoords.size();
    std::set<std::size_t> presentNodesID;
    for (auto const &edge : edge_list) {
        presentNodesID.insert(edge.start);
        presentNodesID.insert(edge.end);
    }
    std::vector<std::size_t> missingNodesID;
    std::size_t i = 0;
    for (auto const &nodeID : presentNodesID) {
        while (i < nodeID) missingNodesID.push_back(i++);
        i++;
    }
    while (i < N) missingNodesID.push_back(i++);

    remove_nodes(missingNodesID, edge_list, nodeCoords, labelMap);
}

void remove_nodes(std::vector<std::size_t> nodesIdToRemove, EdgeList &edge_list, std::vector<IntPoint> &nodeCoords,
                  Tensor2DAccessor<int> &labelMap) {
    // Ensure the nodesIdToRemove are sorted
    std::sort(nodesIdToRemove.begin(), nodesIdToRemove.end());

    auto itToRemove = nodesIdToRemove.begin();
    std::size_t nNodes = 0;
    std::vector<int> lookupTable;
    lookupTable.reserve(nodeCoords.size());

    for (std::size_t i = 0; i < nodeCoords.size(); i++) {
        if (itToRemove != nodesIdToRemove.end() && *itToRemove == i) {
            // If the node is to be removed:
            //  - Remove the node from the labelMap
            auto const &p = nodeCoords[i];
            labelMap[p.y][p.x] = 0;
            //  - Store -1 in the lookup table
            lookupTable.push_back(-1);
            //  - Increment the iterator to search for the next node to remove
            itToRemove++;
        } else {
            // If the branch is not to be removed:
            //  - Copy the corresponding coordinates to their new position
            nodeCoords[nNodes] = nodeCoords[i];
            //  - Update the node in the labelMap
            auto const &p = nodeCoords[nNodes];
            labelMap[p.y][p.x] = nNodes;
            //  - Store the new index in the lookup table
            lookupTable.push_back(nNodes);
            // - Increment the number of nodes
            nNodes++;
        }
    }

    // Update the node coordinates size
    nodeCoords.resize(nNodes);

    // Update the edge list
    for (auto &edge : edge_list) {
        edge.start = lookupTable[edge.start];
        edge.end = lookupTable[edge.end];
    }
}