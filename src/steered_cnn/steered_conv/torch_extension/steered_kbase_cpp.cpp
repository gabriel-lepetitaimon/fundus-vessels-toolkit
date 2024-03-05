#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

std::vector<at::Tensor> conv2d_cosSinAlpha_forward(
        torch::Tensor input,    // shape: (b, n_in, h, w)
        torch::Tensor weights,  // shape: (n_out, n_in, K)
        torch::Tensor real_kernels, 
        torch::Tensor imag_kernels,  // shape: (n_out, n_in, K)
        torch::Tensor cos_sin_alpha) {
    torch::
}

std::vector<at::Tensor> conv2d_cosSinKAlpha_forward(
        torch::Tensor input,    // shape: (b, n_in, h, w)
        torch::Tensor weights,  // shape: (n_out, n_in, K)
        torch::Tensor real_kernels,
        torch::Tensor imag_kernels,  // shape: (n_out, n_in, K)
        torch::Tensor cos_sin_kalpha) {
    torch::
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv2d_forward, "steerable_kbase conv2d forward");
}