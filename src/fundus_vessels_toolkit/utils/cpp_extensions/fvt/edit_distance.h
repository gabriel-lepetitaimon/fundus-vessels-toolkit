#ifndef EDIT_DISTANCE_H
#define EDIT_DISTANCE_H

#include "common.h"

std::array<torch::Tensor, 2> shortest_secondary_path(const torch::Tensor& edge_list, const torch::Tensor& primary_nodes,
                                                     const std::size_t n_nodes, bool directed_edge = false);

std::vector<std::list<int>> backtrack_edges(const torch::Tensor& backtrack_edge_node,
                                            const torch::Tensor& src_dst_nodes, const torch::Tensor& primary_nodes);

#endif  // EDIT_DISTANCE_H