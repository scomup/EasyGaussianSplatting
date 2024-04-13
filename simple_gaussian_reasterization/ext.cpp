#include <torch/extension.h>
#include <iostream>
#include <vector>

std::vector<torch::Tensor> rasterizGuassian2DCUDA(
    torch::Tensor u,
    torch::Tensor areas,
    torch::Tensor cov2d_inv,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor color,
    int H,
    int W);

// forward.md 5.
// Use CUDA rasterization for acceleration
std::vector<at::Tensor> rasterizGuassian2D(
    torch::Tensor u,
    torch::Tensor areas,
    torch::Tensor cov2d_inv,
    torch::Tensor opacity,
    torch::Tensor depth,
    torch::Tensor color,
    int H,
    int W)
{
  return rasterizGuassian2DCUDA(u, areas, cov2d_inv, opacity, depth, color, H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("rasterize", &rasterizGuassian2D, "rasterize 2d guassian (CUDA)");
}
