#ifndef GRAPH_H
#define GRAPH_H

#include "common.h"

/**************************************************************************************
 *              === GRAPH_UTILS.CPP ===
 **************************************************************************************/
EdgeList terminal_edges(const EdgeList &edgeList);
EdgeList terminal_edges(const GraphAdjList &edgeList);
std::vector<int> nodes_rank(const EdgeList &edgeList);
std::vector<int> nodes_rank(const GraphAdjList &edgeList);
std::size_t nodesCount(const EdgeList &edgeList);

/**************************************************************************************
 *              === GRAPH_FIXING.CPP ===
 **************************************************************************************/
void remove_branches(std::vector<Edge> branchesToRemove, std::vector<CurveYX> &branchCurves,
                     Tensor2DAccessor<int> &branchesLabelMap, EdgeList &edge_list);

void remove_nodes(std::vector<std::size_t> nodesIdToRemove, EdgeList &edge_list, std::vector<IntPoint> &nodeCoords,
                  Tensor2DAccessor<int> &labelMap);

void remove_singleton_nodes(EdgeList &edge_list, std::vector<IntPoint> &nodeCoords, Tensor2DAccessor<int> &labelMap);

#endif  // GRAPH_H