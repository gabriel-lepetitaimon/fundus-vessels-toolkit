#ifndef SKELETON_H
#define SKELETON_H

#include "common.h"

std::tuple<EdgeList, std::vector<CurveYX>, std::vector<IntPoint>> parse_skeleton_to_graph(torch::Tensor& labelMap);

torch::Tensor detect_skeleton_nodes(torch::Tensor skeleton, bool fix_hollow = true,
                                    bool remove_single_endpoints = true);

#endif  // SKELETON_H
