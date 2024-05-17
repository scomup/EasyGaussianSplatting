#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include "common.h"


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
    float3 *shared_color)
{
    int i = blockDim.x * threadIdx.y + threadIdx.x;  // block idx
    int j = range.x + n * BLOCK_SIZE + i;  // patch idx
    if (j < range.y)
    {
        int gs_id = gs_id_per_patch[j];
        shared_pos2d[i].x = us[gs_id * 2];
        shared_pos2d[i].y = us[gs_id * 2 + 1];
        shared_cinv2d[i].x = cinv2d[gs_id * 3];
        shared_cinv2d[i].y = cinv2d[gs_id * 3 + 1];
        shared_cinv2d[i].z = cinv2d[gs_id * 3 + 2];
        shared_alpha[i] =   alphas[gs_id];
        shared_color[i].x = colors[gs_id * 3];
        shared_color[i].y = colors[gs_id * 3 + 1];
        shared_color[i].z = colors[gs_id * 3 + 2];
    }
}

__global__ void createKey(const int gs_num,
                          const dim3 grid,
                          const uint4 *__restrict__ rects,
                          const float *__restrict__ depths,
                          const uint *__restrict__ patch_offset_per_gs,
                          uint64_t *__restrict__ patch_keys,
                          int *__restrict__ gs_id_per_patch)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= gs_num)
		return;
    uint32_t off = (idx == 0) ? 0 : patch_offset_per_gs[idx - 1];
    uint4 rect = rects[idx];

		for (uint y = rect.y; y < rect.w; y++)
		{
			for (uint x = rect.x; x < rect.z; x++)
			{
				uint64_t key = (y * grid.x + x);
				key <<= 32;
                uint32_t depth_cm = depths[idx] * 1000; // mm
				key |= depth_cm;
				patch_keys[off] = key;
				gs_id_per_patch[off] = idx;
				off++;
			}
		}
}

__global__ void getRect(
    const int width,
    const int height,
    int gs_num,
    const float *__restrict__ us,
    const float *__restrict__ areas,
    const float *__restrict__ depths,
    const dim3 grid,
    uint4 *__restrict__ gs_rects,
    uint *__restrict__ patch_num_per_gs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= gs_num)
		return;

    float d = depths[idx];
    float2 u = {us[idx*2], us[idx*2 + 1]};

    float x_norm =  u.x / width * 2.f - 1.f;
    float y_norm =  u.y / height * 2.f - 1.f;
    if (abs(x_norm) > 1.3 || abs(y_norm) > 1.3 || d < 0.1 || d > 100)
    {
        gs_rects[idx] = {0, 0, 0, 0};
        patch_num_per_gs[idx] = 0;
		return;
    }

    float xs = areas[idx*2];
    float ys = areas[idx*2 + 1];

    uint4 rect = {
		min(grid.x, max((int)0, (int)((u.x - xs) / BLOCK))),  // min_x
		min(grid.y, max((int)0, (int)((u.y - ys) / BLOCK))),  // min_y
        min(grid.x, max((int)0, (int)(DIV_ROUND_UP(u.x + xs, BLOCK)))),  // max_x
		min(grid.y, max((int)0, (int)(DIV_ROUND_UP(u.y + ys, BLOCK))))   // max_y
	};

    gs_rects[idx] = rect;
    patch_num_per_gs[idx] = (rect.z - rect.x) * (rect.w - rect.y);
}

__global__ void getRange(
    const int patch_num,
    const uint64_t *__restrict__ patch_keys,
    int *__restrict__ patch_range_per_tile)
{
    const int cur_patch = blockIdx.x * blockDim.x + threadIdx.x;

    if (cur_patch >= patch_num)
        return;

    const int prv_patch = cur_patch == 0 ? 0 : cur_patch - 1;

    uint32_t cur_tile = patch_keys[cur_patch] >> 32;
    uint32_t prv_tile = patch_keys[prv_patch] >> 32;

    if (cur_patch == 0)
        patch_range_per_tile[2*cur_tile] = 0;
    else if (cur_patch == patch_num - 1)
        patch_range_per_tile[2*cur_tile + 1] = patch_num;

    if (prv_tile != cur_tile)
    {
        patch_range_per_tile[2*prv_tile + 1] = cur_patch;
        patch_range_per_tile[2*cur_tile] = cur_patch;
    }
}


__global__ void  draw __launch_bounds__(BLOCK * BLOCK)(
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
    float *__restrict__ final_tau)

{
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix = {tile.x * BLOCK + threadIdx.x,
                       tile.y * BLOCK + threadIdx.y};

    const int tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx = width * pix.y + pix.x;

    const bool inside = pix.x < width && pix.y < height;
    const int2 range = {patch_range_per_tile[2 * tile_idx],
                        patch_range_per_tile[2 * tile_idx + 1]};

	const int gs_num = range.y - range.x;

    // not patch for this tile.
    if (gs_num == 0)
        return;

	bool thread_is_finished = !inside;

	__shared__ float2 shared_pos2d[BLOCK_SIZE];
	__shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha[BLOCK_SIZE];
    __shared__ float3 shared_color[BLOCK_SIZE];


    float3 finial_color = {0, 0, 0};

    float tau = 1.0f;

    int cont = 0;
    int cont_tmp = 0;

    // for all 2d gaussian 
    for (int i = 0; i < gs_num; i++)
    {
        int finished_thread_num = __syncthreads_count(thread_is_finished);

        if (finished_thread_num == BLOCK_SIZE)
            break;

        int j = i % BLOCK_SIZE;

        if (j == 0)
        {
            // fetch 2d gaussian data to share memory
            fetch2shared(i / BLOCK_SIZE,
                         range,
                         gs_id_per_patch,
                         us,
                         cinv2d,
                         alphas,
                         colors,
                         shared_pos2d,
                         shared_cinv2d,
                         shared_alpha,
                         shared_color);
            __syncthreads();
        }

        if(thread_is_finished)
            continue;

        // get 2d gaussian info for current tile (pix share the same info within the tile)
        float2 u = shared_pos2d[j];
        float3 cinv = shared_cinv2d[j];
        float alpha = shared_alpha[j];
        float3 color = shared_color[j];
        float2 d = u - pix;

        cont_tmp = cont_tmp + 1;

        // forward.md (5.1)
        // mahalanobis squared distance for 2d gaussian to this pix
        float maha_dist = max(0.0f,  mahaSqDist(cinv, d));

        float alpha_prime = min(0.99f, alpha * exp( -0.5f * maha_dist));
        if (alpha_prime < 0.002f)
            continue;

        // forward.md (5)
        finial_color +=  tau * alpha_prime * color;
        cont = cont_tmp;   // how many gs contribute to this pixel. 

        // forward.md (5.2)
        tau = tau * (1.f - alpha_prime);

        // if(pix.x == 167 && pix.y == 392)
        // {
        //     printf("%i, finial_color %f %f %f\n", i, finial_color.x, finial_color.y, finial_color.z);
        // }

        if (tau < 0.0001f)
        {
            thread_is_finished = true;
            continue;
        }
    }

    if (inside)
    {
        image[height * width * 0 + pix_idx] = finial_color.x;
        image[height * width * 1 + pix_idx] = finial_color.y;
        image[height * width * 2 + pix_idx] = finial_color.z;
        contrib[pix_idx] = cont;
        final_tau[pix_idx] = tau;
    }
}

__global__ void inverseCov2D(
    int gs_num,
    const float *__restrict__ cov2d,
    float *__restrict__ cinv2d,
    float *__restrict__ areas)
{
    // compute inverse of cov2d
    // Determine the drawing area of 2d Gaussian.

    const int gs_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (gs_id >= gs_num)
		return;
    // forward.md 5.3
    const float a = cov2d[gs_id * 3];
    const float b = cov2d[gs_id * 3 + 1];
    const float c = cov2d[gs_id * 3 + 2];

    const float det = a*c - b*b;
    if (det == 0.0f)
		return;

    const float det_inv = 1.f/det;
    cinv2d[gs_id * 3 + 0] =  det_inv * c;
    cinv2d[gs_id * 3 + 1] = -det_inv * b;
    cinv2d[gs_id * 3 + 2] =  det_inv * a;
    areas[gs_id * 2 + 0] =  3 * sqrt(abs(a));
    areas[gs_id * 2 + 1] =  3 * sqrt(abs(c));
}

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


__global__ void computeCov3D(
    int32_t gs_num,
    const float *__restrict__ rots,
    const float *__restrict__ scales,
    float *__restrict__ cov3ds)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
		return;

    float w = rots[i * 4 + 0];
    float x = rots[i * 4 + 1];
    float y = rots[i * 4 + 2];
    float z = rots[i * 4 + 3];
    float s0 = scales[i * 3 + 0];
    float s1 = scales[i * 3 + 1];
    float s2 = scales[i * 3 + 2];
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    Matrix<3, 3> M = {
        (1.f - 2.f * (y2 + z2)) * s0, (2.f * (x * y - z * w)) * s1, (2.f * (x * z + y * w)) * s2,
        (2.f * (x * y + z * w)) * s0, (1.f - 2.f * (x2 + z2)) * s1, (2.f * (y * z - x * w)) * s2,
        (2.f * (x * z - y * w)) * s0, (2.f * (y * z + x * w)) * s1, (1.f - 2.f * (x2 + y2)) * s2};

    Matrix<3, 3> Sigma = M * M.transpose();

    cov3ds[i * 6 + 0] = Sigma(0, 0);
    cov3ds[i * 6 + 1] = Sigma(0, 1);
    cov3ds[i * 6 + 2] = Sigma(0, 2);
    cov3ds[i * 6 + 3] = Sigma(1, 1);
    cov3ds[i * 6 + 4] = Sigma(1, 2);
    cov3ds[i * 6 + 5] = Sigma(2, 2);
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

__global__ void computeCov2D(
    int32_t gs_num,
    const float *__restrict__ cov3ds,
    const float *__restrict__ pcs,
    const float *__restrict__ Rcw,
    const float focal_x,
    const float focal_y,
    float *__restrict__ cov2ds)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
		return;

    const float x = pcs[i * 3 + 0];
    const float y = pcs[i * 3 + 1];
    const float z = pcs[i * 3 + 2];
    const float cov3d0 = cov3ds[i * 6 + 0];
    const float cov3d1 = cov3ds[i * 6 + 1];
    const float cov3d2 = cov3ds[i * 6 + 2];
    const float cov3d3 = cov3ds[i * 6 + 3];
    const float cov3d4 = cov3ds[i * 6 + 4];
    const float cov3d5 = cov3ds[i * 6 + 5];

    float z2 = z * z;

    Matrix<3, 3> W = {
        Rcw[0], Rcw[1], Rcw[2],
        Rcw[3], Rcw[4], Rcw[5],
        Rcw[6], Rcw[7], Rcw[8]};

    Matrix<2, 3> J = {
        focal_x / z, 0.0f, -(focal_x * x) / z2,
        0.0f, focal_y / z, -(focal_y * y) / z2};

    Matrix<3, 3> Sigma = {
        cov3d0, cov3d1, cov3d2,
        cov3d1, cov3d3, cov3d4,
        cov3d2, cov3d4, cov3d5};

    Matrix<2, 3> M = J * W;

    Matrix<2, 2> Sigma_prime = M * Sigma * M.transpose();

    // make sure the cov2d is not too small.
    cov2ds[i * 3 + 0] = Sigma_prime(0, 0) + 0.3;
    cov2ds[i * 3 + 1] = Sigma_prime(0, 1);
    cov2ds[i * 3 + 2] = Sigma_prime(1, 1) + 0.3;
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
    float *__restrict__ pcs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
		return;

    Matrix<3, 1> pw = {pws[i * 3 + 0], pws[i * 3 + 1], pws[i * 3 + 2]}; 

    Matrix<3, 1> _tcw = {tcw[0], tcw[1], tcw[2]}; 

    Matrix<3, 3> _Rcw = {
        Rcw[0], Rcw[1], Rcw[2],
        Rcw[3], Rcw[4], Rcw[5],
        Rcw[6], Rcw[7], Rcw[8]};

    Matrix<3, 1> pc = _Rcw * pw + _tcw;

    const float x = pc(0);
    const float y = pc(1);
    const float z = pc(2);

    const float u0 = x * focal_x / z + center_x;
    const float u1 = y * focal_y / z + center_y;

    // make sure the cov2d is not too small.
    us[i * 2 + 0] = u0;
    us[i * 2 + 1] = u1;
    pcs[i * 3 + 0] = x;
    pcs[i * 3 + 1] = y;
    pcs[i * 3 + 2] = z;
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