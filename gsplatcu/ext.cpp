/* Copyright:
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> splat(
    const int height,
    const int width,
    const torch::Tensor us,
    const torch::Tensor cinv2ds,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor colors,
    const torch::Tensor areas);

std::vector<torch::Tensor> splatB(
    const int height,
    const int width,
    const torch::Tensor us,
    const torch::Tensor cinv2ds,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor colors,
    const torch::Tensor contrib,
    const torch::Tensor final_tau,
    const torch::Tensor patch_range_per_tile,
    const torch::Tensor gsid_per_patch,
    const torch::Tensor dloss_dgammas);

std::vector<torch::Tensor> inverseCov2D(const torch::Tensor cov2ds,
                                        const torch::Tensor depths,
                                        const bool calc_J);


std::vector<torch::Tensor> computeCov3D(const torch::Tensor rots,
                                        const torch::Tensor scales,
                                        const torch::Tensor depths,
                                        const bool calc_J);

std::vector<torch::Tensor> computeCov2D(const torch::Tensor cov3ds,
                                        const torch::Tensor pcs,
                                        const torch::Tensor Rcw,
                                        const torch::Tensor depths,
                                        const float focal_x,
                                        const float focal_y,
                                        const float width,
                                        const float height,
                                        const bool calc_J);

std::vector<torch::Tensor> project(const torch::Tensor pws,
                                   const torch::Tensor Rcw,
                                   const torch::Tensor tcw,
                                   float focal_x,
                                   float focal_y,
                                   float center_x,
                                   float center_y,
                                   const bool calc_J);

std::vector<torch::Tensor> sh2Color(const torch::Tensor shs,
                                    const torch::Tensor pws,
                                    const torch::Tensor twc,
                                    const bool calc_J);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("splat", &splat, "splat 2d gaussian to image");
  m.def("splatB", &splatB, "splat 2d gaussian to image backward");
  m.def("inverseCov2D", &inverseCov2D, "inverse 2D covariances");
  m.def("computeCov3D", &computeCov3D, "compute 3D covariances");
  m.def("computeCov2D", &computeCov2D, "compute 2D covariances");
  m.def("project", &project, "project point to image");
  m.def("sh2Color", &sh2Color, "covert SH to color");
}