#ifndef GRAPH_H
#define GRAPH_H

#include "common.h"

/**************************************************************************************
 *              === GRAPH_UTILS.CPP ===
 **************************************************************************************/
EdgeList terminal_edges(const EdgeList &edgeList, bool directed = false);
EdgeList terminal_edges(const GraphAdjList &adjList);
std::vector<int> nodes_rank(const EdgeList &edgeList, bool onlyIncoming = false);
std::vector<int> nodes_rank(const GraphAdjList &edgeList);
std::size_t nodesCount(const EdgeList &edgeList);

std::vector<std::list<int>> connected_components(const EdgeList &edgeList, int N = -1);
std::vector<std::list<int>> connected_components(const GraphAdjList &adjList);
std::vector<int> maximum_weighted_independent_set(std::vector<IntPair> edges, std::vector<float> weights);

/**************************************************************************************
 *              === GRAPH_FIXING.CPP ===
 **************************************************************************************/
void remove_branches(std::vector<Edge> branchesToRemove, std::vector<CurveYX> &branchCurves,
                     Tensor2DAcc<int> &branchesLabelMap, EdgeList &edge_list);

void remove_nodes(std::vector<std::size_t> nodesIdToRemove, EdgeList &edge_list, std::vector<IntPoint> &nodeCoords,
                  Tensor2DAcc<int> &labelMap);

void remove_singleton_nodes(EdgeList &edge_list, std::vector<IntPoint> &nodeCoords, Tensor2DAcc<int> &labelMap);

#endif  // GRAPH_H