#include <torch/extension.h>
#include <iostream>
#include <vector>

std::vector<torch::Tensor> forward(
    const int H,
    const int W,
    const torch::Tensor u,
    const torch::Tensor cov2d,
    const torch::Tensor alpha,
    const torch::Tensor depth,
    const torch::Tensor color
    );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &forward, "rasterize 2d guassian (CUDA)");
}
