#ifndef SKELETON_H
#define SKELETON_H

#include "common.h"

torch::Tensor skeletonize(const torch::Tensor& segMap);
torch::Tensor skeletonize_av(const torch::Tensor& segMap);

std::tuple<EdgeList, std::vector<CurveYX>, std::vector<IntPoint>> parse_skeleton_to_graph(torch::Tensor& labelMap);

torch::Tensor detect_skeleton_nodes(torch::Tensor skeleton, bool fix_hollow = true, bool remove_single_endpoints = true,
                                    bool return_skeleton_rank = false);

torch::Tensor detect_skeleton_nodes_debug(torch::Tensor skeleton);

#endif  // SKELETON_H
