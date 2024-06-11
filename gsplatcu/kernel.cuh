/* Copyright:
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <torch/extension.h>
#include "common.cuh"

inline __device__ void fetch2shared(
    int32_t n,
    const int2 range,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2d,
    const float *__restrict__ alphas,
    const float3 *__restrict__ colors,
    float2 *shared_pos2d,
    float3 *shared_cinv2d,
    float *shared_alpha,
    float3 *shared_color);

__global__ void createKeys(
    const int gs_num,
    const float* __restrict__ depths,
    const uint32_t* __restrict__ patch_offset_per_gs,
    const uint4* __restrict__ rects,
    dim3 grid,
    uint64_t* __restrict__ patch_keys,
    int* __restrict__ gsid_per_patch);

__global__ void getRects(
    const int gs_num,
    const float* __restrict__ us,
    int2* __restrict__ areas,
    float* __restrict__ depths,
    const dim3 grid,
    uint4 *__restrict__ gs_rects,
    uint* __restrict__ patch_num_per_gs);

__global__ void getRanges(
    const int patch_num,
    const uint64_t *__restrict__ patch_keys,
    int2 *__restrict__ patch_range_per_tile);

__global__ void draw __launch_bounds__(BLOCK *BLOCK)(
    const int width,
    const int height,
    const int2 *__restrict__ patch_range_per_tile,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2d,
    const float *__restrict__ alphas,
    const float3 *__restrict__ colors,
    float *__restrict__ image,
    int *__restrict__ contrib,
    float *__restrict__ final_tau);

__global__ void inverseCov2D(
    int gs_num,
    const float3 *__restrict__ cov2d,
    float *__restrict__ depths,
    float3 *__restrict__ cinv2d,
    int2 *__restrict__ areas,
    float *__restrict__ dcinv2d_dcov2ds = nullptr);

__global__ void computeCov3D(
    int32_t gs_num,
    const float4 *__restrict__ rots,
    const float3 *__restrict__ scales,
    const float *__restrict__ depths,
    float *__restrict__ cov3ds,
    float *__restrict__ dcov3d_drots = nullptr,
    float *__restrict__ dcov3d_dscales = nullptr);

__global__ void computeCov2D(
    int32_t gs_num,
    const float *__restrict__ cov3ds,
    const float3 *__restrict__ pcs,
    const float *__restrict__ Rcw,
    const float *__restrict__ depths,
    const float focal_x,
    const float focal_y,
    const float tan_fovx,
    const float tan_fovy,
    float3 *__restrict__ cov2ds,
    float *__restrict__ dcov2d_dcov3ds = nullptr,
    float *__restrict__ dcov2d_dpcs = nullptr);

__global__ void project(
    int32_t gs_num,
    const float3 *__restrict__ pws,
    const float *__restrict__ Rcw,
    const float3 *__restrict__ tcw,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    float2 *__restrict__ us,
    float3 *__restrict__ pcs,
    float *__restrict__ depths,
    float *__restrict__ du_dpcs = nullptr);

__global__ void sh2Color(
    int32_t gs_num,
    const float3 *__restrict__ shs,
    const float3 *__restrict__ pws,
    const float3 *__restrict__ twc,
    const int sh_dim3,
    float3 *__restrict__ colors,
    float *__restrict__ dcolor_dshs = nullptr,
    float *__restrict__ dcolor_dpws = nullptr);

__global__ void __launch_bounds__(BLOCK *BLOCK)
    drawB(
        const int width, 
        const int height,
        const int2 *__restrict__ patch_range_per_tile,
        const int *__restrict__ gsid_per_patch,
        const float2 *__restrict__ us,
        const float3 *__restrict__ cinv2ds,
        const float *__restrict__ alphas,
        const float3 *__restrict__ colors,
        const float *__restrict__ final_tau,
        const int *__restrict__ contrib,
        const float *__restrict__ dloss_dgammas,
        float2 *__restrict__ dloss_dus,
        float3 *__restrict__ dloss_dcinv2ds,
        float *__restrict__ dloss_dalphas,
        float *__restrict__ dloss_dcolors);