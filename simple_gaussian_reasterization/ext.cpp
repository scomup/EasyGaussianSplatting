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

std::vector<torch::Tensor> backward(
    const int H,
    const int W,
    const torch::Tensor us,
    const torch::Tensor cov2d,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor colors,
    const torch::Tensor contrib,
    const torch::Tensor final_tau, 
    const torch::Tensor patch_offset_per_tile, 
    const torch::Tensor gs_id_per_patch,
    const torch::Tensor dloss_dgammas);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("forward", &forward, "rasterize 2d guassian (CUDA)");
  m.def("backward", &backward, "rasterize 2d guassian (CUDA)");
}
