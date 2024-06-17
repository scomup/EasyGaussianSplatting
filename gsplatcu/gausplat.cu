/* Copyright:
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "kernel.cuh"


std::vector<torch::Tensor> splat(
    const int height,
    const int width,
    const torch::Tensor us,
    const torch::Tensor cinv2ds,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor colors,
    const torch::Tensor areas)
{
    auto float_opts = us.options().dtype(torch::kFloat32);
    auto int_opts = us.options().dtype(torch::kInt32);
    torch::Tensor image = torch::full({3, height, width}, 0.0, float_opts);
    torch::Tensor contrib = torch::full({height, width}, 0, int_opts);
    torch::Tensor final_tau = torch::full({height, width}, 0, float_opts);

    //gs:    2d gaussian;  a projection of a 3d gaussian onto a 2d image
    //tile:  a 16x16 area of 2d image
    //patch: a 2d gaussian may cover on many tiles, a 2d gaussian on a tile is called a patch

    // the total number of 2d gaussian.
    int gs_num = us.sizes()[0]; 
    
    dim3 grid(DIV_ROUND_UP(width, BLOCK), DIV_ROUND_UP(height, BLOCK), 1);
	dim3 block(BLOCK, BLOCK, 1);

    thrust::device_vector<uint4> gs_rects(gs_num);
    thrust::device_vector<uint>  patch_num_per_gs(gs_num);
    thrust::device_vector<uint>  patch_offset_per_gs(gs_num);

    getRects<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE >>>(
        gs_num,
        us.contiguous().data_ptr<float>(),
        (int2*)areas.contiguous().data_ptr<int>(),
        depths.contiguous().data_ptr<float>(),
        grid,
        thrust::raw_pointer_cast(gs_rects.data()),
        thrust::raw_pointer_cast(patch_num_per_gs.data()));
    CHECK_CUDA(DEBUG);

    thrust::inclusive_scan(patch_num_per_gs.begin(), patch_num_per_gs.end(), patch_offset_per_gs.begin());

    // patch_num: The total number of patches needs to be drawn
    uint patch_num = (uint)patch_offset_per_gs[gs_num - 1];  // copy to cpu memory

    thrust::device_vector<uint64_t> patch_keys(patch_num);
    thrust::device_vector<int> gsid_per_patch(patch_num);

    createKeys <<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>> (
        gs_num,
        depths.data_ptr<float>(),
        thrust::raw_pointer_cast(patch_offset_per_gs.data()),
        thrust::raw_pointer_cast(gs_rects.data()),
        grid,
        thrust::raw_pointer_cast(patch_keys.data()),
        thrust::raw_pointer_cast(gsid_per_patch.data()));
    CHECK_CUDA(DEBUG);

    thrust::sort_by_key(patch_keys.begin(), patch_keys.end(), gsid_per_patch.begin());
    const uint tile_num = grid.x * grid.y;
    torch::Tensor patch_range_per_tile = torch::full({tile_num, 2}, 0, int_opts);

    if (patch_num > 0)
    {
        getRanges <<<DIV_ROUND_UP(patch_num, BLOCK_SIZE), BLOCK_SIZE >>> (
            patch_num,
            thrust::raw_pointer_cast(patch_keys.data()),
            (int2*)patch_range_per_tile.contiguous().data_ptr<int>());
    }

    draw<<< grid, block >>> (
        width,
        height,
        (int2*)patch_range_per_tile.contiguous().data_ptr<int>(),
        thrust::raw_pointer_cast(gsid_per_patch.data()),
        (float2*)us.contiguous().data_ptr<float>(),
        (float3*)cinv2ds.contiguous().data_ptr<float>(),
        alphas.contiguous().data_ptr<float>(),
        (float3*)colors.contiguous().data_ptr<float>(),
        image.contiguous().data_ptr<float>(),
        contrib.contiguous().data_ptr<int>(),
        final_tau.contiguous().data_ptr<float>());
    CHECK_CUDA(DEBUG);

    torch::Tensor gsid_per_patch_tensor = torch::from_blob(thrust::raw_pointer_cast(gsid_per_patch.data()), 
        {static_cast<long>(gsid_per_patch.size())}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kCUDA);

    return {image, contrib, final_tau, patch_range_per_tile, gsid_per_patch_tensor};
}

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
    const torch::Tensor dloss_dgammas)
{
    // backward version of splat.
    int gs_num = us.sizes()[0]; 
    dim3 grid(DIV_ROUND_UP(width, BLOCK), DIV_ROUND_UP(height, BLOCK), 1);
	dim3 block(BLOCK, BLOCK, 1);
    
    auto float_opts = us.options().dtype(torch::kFloat32);
    auto int_opts = us.options().dtype(torch::kInt32);
    torch::Tensor image = torch::full({3, height, width}, 0.0, float_opts);
    torch::Tensor dloss_dalphas = torch::full({gs_num, 1, 1}, 0, float_opts);
    torch::Tensor dloss_dcolors = torch::full({gs_num, 1, 3}, 0, float_opts);
    torch::Tensor dloss_dcinv2ds = torch::full({gs_num, 1, 3}, 0, float_opts);
    torch::Tensor dloss_dus = torch::full({gs_num, 1, 2}, 0, float_opts);

    drawB<<<grid, block>>>(
        width,
        height,
        (int2*)patch_range_per_tile.contiguous().data_ptr<int>(),
        gsid_per_patch.contiguous().data_ptr<int>(),
        (float2*)us.contiguous().data_ptr<float>(),
        (float3*)cinv2ds.contiguous().data_ptr<float>(),
        alphas.contiguous().data_ptr<float>(),
        (float3*)colors.contiguous().data_ptr<float>(),
        final_tau.contiguous().data_ptr<float>(),
        contrib.contiguous().data_ptr<int>(),
        dloss_dgammas.contiguous().data_ptr<float>(),
        (float2*)dloss_dus.contiguous().data_ptr<float>(),
        (float3*)dloss_dcinv2ds.contiguous().data_ptr<float>(),
        dloss_dalphas.contiguous().data_ptr<float>(),
        dloss_dcolors.contiguous().data_ptr<float>());
    CHECK_CUDA(DEBUG);
   return {dloss_dus, dloss_dcinv2ds, dloss_dalphas, dloss_dcolors};
}


std::vector<torch::Tensor> computeCov3D(const torch::Tensor rots,
                                        const torch::Tensor scales,
                                        const torch::Tensor depths,
                                        const bool calc_J)
{
    auto float_opts = rots.options().dtype(torch::kFloat32);
    auto int_opts = rots.options().dtype(torch::kInt32);
    int gs_num = rots.sizes()[0];
    torch::Tensor cov3ds = torch::full({gs_num, 6}, 0.0, float_opts);

    torch::Tensor dcov3d_drots;
    torch::Tensor dcov3d_dscales;

    if (calc_J)
    {
        dcov3d_drots = torch::full({gs_num, 6, 4}, 0.0, float_opts);
        dcov3d_dscales = torch::full({gs_num, 6, 3}, 0.0, float_opts);
    }

    computeCov3D<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        (float4*)rots.contiguous().data_ptr<float>(),
        (float3*)scales.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        cov3ds.contiguous().data_ptr<float>(),
        calc_J ? dcov3d_drots.contiguous().data_ptr<float>() : nullptr,
        calc_J ? dcov3d_dscales.contiguous().data_ptr<float>() : nullptr);
    CHECK_CUDA(DEBUG);

    if (calc_J)
    {
        return {cov3ds, dcov3d_drots, dcov3d_dscales};
    }
    else
    {
        return {cov3ds};
    }
}

std::vector<torch::Tensor> computeCov2D(const torch::Tensor cov3ds,
                                        const torch::Tensor pcs,
                                        const torch::Tensor Rcw,
                                        const torch::Tensor depths,
                                        const float focal_x,
                                        const float focal_y,
                                        const float width,
                                        const float height,
                                        const bool calc_J)
{
    auto float_opts = pcs.options().dtype(torch::kFloat32);
    auto int_opts = pcs.options().dtype(torch::kInt32);
    int gs_num = pcs.sizes()[0];
    torch::Tensor cov2ds = torch::full({gs_num, 3}, 0.0, float_opts);

    torch::Tensor dcov2d_dcov3ds;
    torch::Tensor dcov2d_dpcs;

    if (calc_J)
    {
        dcov2d_dcov3ds = torch::full({gs_num, 3, 6}, 0.0, float_opts);
        dcov2d_dpcs = torch::full({gs_num, 3, 3}, 0.0, float_opts);
    }

    const float tan_fovx = width / (2 * focal_x);
    const float tan_fovy = height / (2 * focal_y);

    computeCov2D<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        cov3ds.contiguous().data_ptr<float>(),
        (float3*)pcs.contiguous().data_ptr<float>(),
        Rcw.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        (float3*)cov2ds.contiguous().data_ptr<float>(),
        calc_J ? dcov2d_dcov3ds.contiguous().data_ptr<float>() : nullptr,
        calc_J ? dcov2d_dpcs.contiguous().data_ptr<float>(): nullptr);
    CHECK_CUDA(DEBUG);

    if (calc_J)
    {
        return {cov2ds, dcov2d_dcov3ds, dcov2d_dpcs};
    }
    else
    {
        return {cov2ds};
    }
}

std::vector<torch::Tensor> project(const torch::Tensor pws,
                                   const torch::Tensor Rcw,
                                   const torch::Tensor tcw,
                                   float focal_x,
                                   float focal_y,
                                   float center_x,
                                   float center_y,
                                   const bool calc_J)
{
    auto float_opts = pws.options().dtype(torch::kFloat32);
    auto int_opts = pws.options().dtype(torch::kInt32);
    int gs_num = pws.sizes()[0]; 
    torch::Tensor us = torch::full({gs_num, 2}, 0.0, float_opts);
    torch::Tensor pcs = torch::full({gs_num, 3}, 0.0, float_opts);
    torch::Tensor depths = torch::full({gs_num}, 0.0, float_opts);
    torch::Tensor du_dpcs;
    if (calc_J)
    {
        du_dpcs = torch::full({gs_num, 2, 3}, 0.0, float_opts);
    }
    project<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        (float3*)pws.contiguous().data_ptr<float>(),
        Rcw.contiguous().data_ptr<float>(),
        (float3*)tcw.contiguous().data_ptr<float>(),
        focal_x,
        focal_y,
        center_x,
        center_y,
        (float2*)us.contiguous().data_ptr<float>(),
        (float3*)pcs.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        calc_J ? du_dpcs.contiguous().data_ptr<float>() : nullptr);
    CHECK_CUDA(DEBUG);

    if (calc_J)
    {
        return {us, pcs, depths, du_dpcs};
    }
    else
    {
        return {us, pcs, depths};
    }
}

std::vector<torch::Tensor> sh2Color(const torch::Tensor shs,
                                    const torch::Tensor pws,
                                    const torch::Tensor twc,
                                    const bool calc_J)
{
    auto float_opts = pws.options().dtype(torch::kFloat32);
    int gs_num = pws.sizes()[0]; 
    int sh_dim = shs.sizes()[1];
    int sh_dim3 = shs.sizes()[1] / 3; 
    torch::Tensor colors = torch::full({gs_num, 3}, 0.0, float_opts);
    torch::Tensor dcolor_dshs;
    torch::Tensor dcolor_dpws;

    //TODO(liu): change max_level by sh_dim

    if (calc_J)
    {
        dcolor_dshs = torch::full({gs_num, 1, sh_dim3}, 0.0, float_opts);
        dcolor_dpws = torch::full({gs_num, 3, 3}, 0.0, float_opts);
    }

    sh2Color<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        (float3*)shs.contiguous().data_ptr<float>(),
        (float3*)pws.contiguous().data_ptr<float>(),
        (float3*)twc.contiguous().data_ptr<float>(),
        sh_dim3,
        (float3*)colors.contiguous().data_ptr<float>(),
        calc_J ? dcolor_dshs.contiguous().data_ptr<float>() : nullptr,
        calc_J ? dcolor_dpws.contiguous().data_ptr<float>() : nullptr);
    CHECK_CUDA(DEBUG);

    if (calc_J)
    {
        return {colors, dcolor_dshs, dcolor_dpws};
    }
    else
    {
        return {colors};
    }
}

std::vector<torch::Tensor> inverseCov2D(const torch::Tensor cov2ds,
                                        const torch::Tensor depths,
                                        const bool calc_J)
{
    auto float_opts = cov2ds.options().dtype(torch::kFloat32);
    auto int_opts = cov2ds.options().dtype(torch::kInt32);
    int gs_num = cov2ds.sizes()[0];
    torch::Tensor cinv2ds = torch::full({gs_num, 3}, 0.0, float_opts);
    torch::Tensor areas = torch::full({gs_num, 2}, 0.0, int_opts);
    torch::Tensor dcinv2d_dcov2d;

    if (calc_J)
    {
        dcinv2d_dcov2d = torch::full({gs_num, 3, 3}, 0.0, float_opts);
    }

    inverseCov2D<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        (float3*)cov2ds.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        (float3*)cinv2ds.contiguous().data_ptr<float>(),
        (int2*)areas.contiguous().data_ptr<int>(),
        calc_J ? dcinv2d_dcov2d.contiguous().data_ptr<float>() : nullptr);
    CHECK_CUDA(DEBUG);

    if (calc_J)
    {
        return {cinv2ds, areas, dcinv2d_dcov2d};
    }
    else
    {
        return {cinv2ds, areas};
    }
}