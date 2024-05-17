/* Copyright:
 * This file is part of pygauspilt.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <torch/extension.h>
#include "common.cuh"

inline __device__ void fetch2shared(
    int32_t n,
    const int2 range,
    const int *__restrict__ gs_id_per_patch,
    const float *__restrict__ us,
    const float *__restrict__ cinv2d,
    const float *__restrict__ alphas,
    const float *__restrict__ colors,
    float2 *shared_pos2d,
    float3 *shared_cinv2d,
    float *shared_alpha,
    float3 *shared_color);

__global__ void createKey(const int gs_num,
                          const dim3 grid,
                          const uint4 *__restrict__ rects,
                          const float *__restrict__ depths,
                          const uint *__restrict__ patch_offset_per_gs,
                          uint64_t *__restrict__ patch_keys,
                          int *__restrict__ gs_id_per_patch);

__global__ void getRect(
    const int width,
    const int height,
    int gs_num,
    const float *__restrict__ us,
    const float *__restrict__ areas,
    const float *__restrict__ depths,
    const dim3 grid,
    uint4 *__restrict__ gs_rects,
    uint *__restrict__ patch_num_per_gs);

__global__ void getRange(
    const int patch_num,
    const uint64_t *__restrict__ patch_keys,
    int *__restrict__ patch_range_per_tile);

__global__ void draw __launch_bounds__(BLOCK *BLOCK)(
    const int width,
    const int height,
    const int *__restrict__ patch_range_per_tile,
    const int *__restrict__ gs_id_per_patch,
    const float *__restrict__ us,
    const float *__restrict__ cinv2d,
    const float *__restrict__ alphas,
    const float *__restrict__ colors,
    float *__restrict__ image,
    int *__restrict__ contrib,
    float *__restrict__ final_tau);

__global__ void inverseCov2D(
    int gs_num,
    const float *__restrict__ cov2d,
    float *__restrict__ cinv2d,
    float *__restrict__ areas);

__global__ void computeCov3D(
    int32_t gs_num,
    const float *__restrict__ rots,
    const float *__restrict__ scales,
    float *__restrict__ cov3ds);

__global__ void computeCov2D(
    int32_t gs_num,
    const float *__restrict__ cov3ds,
    const float *__restrict__ pcs,
    const float *__restrict__ Rcw,
    const float focal_x,
    const float focal_y,
    float *__restrict__ cov2ds);

__global__ void project(
    int32_t gs_num,
    const float *__restrict__ pws,
    const float *__restrict__ Rcw,
    const float *__restrict__ tcw,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    float *__restrict__ us,
    float *__restrict__ pcs);

__global__ void sh2Color(
    int32_t gs_num,
    const float *__restrict__ shs,
    const float *__restrict__ pws,
    const float *__restrict__ twc,
    const int sh_dim,
    float *__restrict__ colors);