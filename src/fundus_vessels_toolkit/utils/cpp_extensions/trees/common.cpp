#include "common.h"

/*******************************************************************************************************************
 *             === TORCH ===
 *******************************************************************************************************************/

torch::Tensor vector_to_tensor(const std::vector<int>& vec) {
    torch::Tensor tensor = torch::empty({(long)vec.size()}, torch::kInt32);
    auto accessor = tensor.accessor<int, 1>();
    for (int i = 0; i < (int)vec.size(); i++) accessor[i] = vec[i];
    return tensor;
}