#include <torch/extension.h>

using namespace torch::indexing;

/**
    Computes the matrix:
        [[cos(α), cos(2α), ..., cos(kα)],
         [sin(α), sin(2α), ..., sin(kα)]]

        Args:
            cos_sin_a: The tensor [cos(α), sin(α)] of shape [2, n0, n1, ...]

            k: The max k

        Returns: The tensor cos_sin_ka of shape [2, k, n0, n1, ...], where:
                    cos_sin_ka[0] = [cos(α), cos(2α), ..., cos(kα)]
                    cos_sin_ka[1] = [sin(α), sin(2α), ..., sin(kα)]
*/
torch::Tensor cos_sin_ka_stack(torch::Tensor cos_sin_a, int k){
    const auto cos_a = cos_sin_a[0];
    const auto sin_a = cos_sin_a[1];
    TensorList
    torch::addcmul()
}

