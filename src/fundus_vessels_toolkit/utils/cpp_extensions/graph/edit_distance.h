#include "common.h"

std::array<torch::Tensor, 2> shortest_secondary_path(const torch::Tensor& edge_list, const torch::Tensor& primary_nodes,
                                                     const torch::Tensor& secondary_nodes);