#include <torch/extension.h>
#include <iostream>
#include <vector>

std::vector<torch::Tensor> rasterizGuassian2D(
    torch::Tensor u,
    torch::Tensor cov2d,
    torch::Tensor alpha,
    torch::Tensor depth,
    torch::Tensor color,
    int H,
    int W);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rasterize", &rasterizGuassian2D, "rasterize 2d guassian (CUDA)");
}
