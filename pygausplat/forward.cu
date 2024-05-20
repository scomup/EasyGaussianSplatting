/* Copyright:
 * This file is part of pygausplat.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include "kernel.cuh"



std::vector<torch::Tensor> forward(
    const int height,
    const int width,
    const torch::Tensor us,
    const torch::Tensor cov2d,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor colors)
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
    thrust::device_vector<float>  cinv2d(gs_num * 3);
    thrust::device_vector<float>  areas(gs_num * 2);

    inverseCov2D<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        cov2d.contiguous().data_ptr<float>(),
        thrust::raw_pointer_cast(cinv2d.data()),
        thrust::raw_pointer_cast(areas.data()));
    cudaDeviceSynchronize();

    getRect<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        width,
        height,
        gs_num,
        us.contiguous().data_ptr<float>(),
        thrust::raw_pointer_cast(areas.data()),
        depths.contiguous().data_ptr<float>(),
        grid,
        thrust::raw_pointer_cast(gs_rects.data()),
        thrust::raw_pointer_cast(patch_num_per_gs.data()));
    cudaDeviceSynchronize();

    thrust::inclusive_scan(patch_num_per_gs.begin(), patch_num_per_gs.end(), patch_offset_per_gs.begin());

    // patch_num: The total number of patches needs to be drawn
    uint patch_num = (uint)patch_offset_per_gs[gs_num - 1];  // copy to cpu memory

    thrust::device_vector<uint64_t> patch_keys(patch_num);
    thrust::device_vector<int> gs_id_per_patch(patch_num);
    
    createKey<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        grid,
        thrust::raw_pointer_cast(gs_rects.data()),
        depths.contiguous().data_ptr<float>(),
        thrust::raw_pointer_cast(patch_offset_per_gs.data()),
        thrust::raw_pointer_cast(patch_keys.data()),
        thrust::raw_pointer_cast(gs_id_per_patch.data()));
    cudaDeviceSynchronize();

    thrust::sort_by_key(patch_keys.begin(), patch_keys.end(), gs_id_per_patch.begin());

    const uint tile_num = grid.x * grid.y;
    torch::Tensor patch_range_per_tile = torch::full({tile_num, 2}, 0, int_opts);

    getRange<<<DIV_ROUND_UP(patch_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        patch_num,
        thrust::raw_pointer_cast(patch_keys.data()),
        patch_range_per_tile.contiguous().data_ptr<int>());
    cudaDeviceSynchronize();

    draw<<<grid, block>>>(
        width,
        height,
        patch_range_per_tile.contiguous().data_ptr<int>(),
        thrust::raw_pointer_cast(gs_id_per_patch.data()),
        us.contiguous().data_ptr<float>(),
        thrust::raw_pointer_cast(cinv2d.data()),
        alphas.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        image.contiguous().data_ptr<float>(),
        contrib.contiguous().data_ptr<int>(),
        final_tau.contiguous().data_ptr<float>());
    cudaDeviceSynchronize();

    torch::Tensor gsid_per_patch_torch = torch::from_blob(thrust::raw_pointer_cast(gs_id_per_patch.data()), 
        {static_cast<long>(gs_id_per_patch.size())}, torch::TensorOptions().dtype(torch::kInt32)).to(torch::kCUDA);

    return {image, contrib, final_tau, patch_range_per_tile, gsid_per_patch_torch};
}


std::vector<torch::Tensor> computeCov3D(const torch::Tensor rots, const torch::Tensor scales)
{
    auto float_opts = rots.options().dtype(torch::kFloat32);
    auto int_opts = rots.options().dtype(torch::kInt32);
    int gs_num = rots.sizes()[0]; 
    torch::Tensor cov3ds = torch::full({gs_num, 6}, 0.0, float_opts);

    computeCov3D<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        rots.contiguous().data_ptr<float>(),
        scales.contiguous().data_ptr<float>(),
        cov3ds.contiguous().data_ptr<float>());
    cudaDeviceSynchronize();

    // the total number of 2d gaussian.
    return {cov3ds};
}

std::vector<torch::Tensor> computeCov2D(const torch::Tensor cov3ds,
                                        const torch::Tensor pcs,
                                        const torch::Tensor Rcw,
                                        float focal_x, float focal_y)
{
    auto float_opts = pcs.options().dtype(torch::kFloat32);
    auto int_opts = pcs.options().dtype(torch::kInt32);
    int gs_num = pcs.sizes()[0]; 
    torch::Tensor cov2ds = torch::full({gs_num, 3}, 0.0, float_opts);

    computeCov2D<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        cov3ds.contiguous().data_ptr<float>(),
        pcs.contiguous().data_ptr<float>(),
        Rcw.contiguous().data_ptr<float>(),
        focal_x,
        focal_y,
        cov2ds.contiguous().data_ptr<float>());
    cudaDeviceSynchronize();

    // the total number of 2d gaussian.
    return {cov2ds};
}

std::vector<torch::Tensor> project(const torch::Tensor pws,
                                        const torch::Tensor Rcw,
                                        const torch::Tensor tcw,
                                        float focal_x, float focal_y,
                                        float center_x, float center_y)
{
    auto float_opts = pws.options().dtype(torch::kFloat32);
    auto int_opts = pws.options().dtype(torch::kInt32);
    int gs_num = pws.sizes()[0]; 
    torch::Tensor us = torch::full({gs_num, 2}, 0.0, float_opts);
    torch::Tensor pcs = torch::full({gs_num, 3}, 0.0, float_opts);

    project<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        pws.contiguous().data_ptr<float>(),
        Rcw.contiguous().data_ptr<float>(),
        tcw.contiguous().data_ptr<float>(),
        focal_x,  focal_y,
        center_x, center_y,
        us.contiguous().data_ptr<float>(),
        pcs.contiguous().data_ptr<float>());
    cudaDeviceSynchronize();

    // the total number of 2d gaussian.
    return {us, pcs};
}

std::vector<torch::Tensor> sh2Color(const torch::Tensor shs,
                                    const torch::Tensor pws,
                                    const torch::Tensor twc,
                                    const bool calc_J)
{
    auto float_opts = pws.options().dtype(torch::kFloat32);
    int gs_num = pws.sizes()[0]; 
    int sh_dim = shs.sizes()[1]; 
    torch::Tensor colors = torch::full({gs_num, 3}, 0.0, float_opts);

    if (calc_J)
    {
        torch::Tensor dc_dshs = torch::full({gs_num, 3, sh_dim}, 0.0, float_opts);
        torch::Tensor dc_dpws = torch::full({gs_num, 3, 3}, 0.0, float_opts);
    
        sh2Color<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
            gs_num,
            shs.contiguous().data_ptr<float>(),
            pws.contiguous().data_ptr<float>(),
            twc.contiguous().data_ptr<float>(),
            sh_dim,
            colors.contiguous().data_ptr<float>(),
            true,
            dc_dshs.contiguous().data_ptr<float>(),
            dc_dpws.contiguous().data_ptr<float>());
        cudaDeviceSynchronize();

        return {colors, dc_dshs, dc_dpws};
    }
    else
    {
        sh2Color<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
            gs_num,
            shs.contiguous().data_ptr<float>(),
            pws.contiguous().data_ptr<float>(),
            twc.contiguous().data_ptr<float>(),
            sh_dim,
            colors.contiguous().data_ptr<float>());
        cudaDeviceSynchronize();

        return {colors};
    }
}
