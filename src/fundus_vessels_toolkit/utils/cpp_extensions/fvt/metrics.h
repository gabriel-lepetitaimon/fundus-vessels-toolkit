#ifndef METRICS_H
#define METRICS_H

#include "common.h"

std::array<torch::Tensor, 6> shortest_skeleton_path_length(torch::Tensor& skeleton);
std::array<std::pair<double, long>, 2> valid_path_ratio(const torch::Tensor& cc1, const torch::Tensor& cc2, long n_cc1,
                                                        long n_cc2);

#endif  // METRICS_H