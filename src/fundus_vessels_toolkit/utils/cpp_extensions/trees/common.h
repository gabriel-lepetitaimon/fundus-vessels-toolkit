#ifndef TORCH_COMMON_H
#define TORCH_COMMON_H

#include <torch/extension.h>

/*******************************************************************************************************************
 *             === TORCH ===
 *******************************************************************************************************************/

torch::Tensor vector_to_tensor(const std::vector<int>& vec);

template <typename T>
std::vector<torch::Tensor> vectors_to_tensors(const std::vector<std::vector<T>>& vec) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(vec.size());
    for (const auto& v : vec) tensors.push_back(vector_to_tensor(v));
    return tensors;
}

#endif