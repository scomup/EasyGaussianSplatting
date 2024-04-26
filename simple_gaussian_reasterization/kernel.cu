#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include "common.h"


inline __device__ void fetch2shared(
    int32_t n,
    const uint2 range,
    const uint *__restrict__ patchs_gs_id,
    const float *__restrict__ us,
    const float *__restrict__ cov2d_inv,
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
        int gs_id = patchs_gs_id[j];
        shared_pos2d[i].x = us[gs_id * 2];
        shared_pos2d[i].y = us[gs_id * 2 + 1];
        shared_cinv2d[i].x = cov2d_inv[gs_id * 3];
        shared_cinv2d[i].y = cov2d_inv[gs_id * 3 + 1];
        shared_cinv2d[i].z = cov2d_inv[gs_id * 3 + 2];
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
                          const uint *__restrict__ offsets,
                          uint64_t *__restrict__ patch_keys,
                          uint *__restrict__ patchs_gs_id)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= gs_num)
		return;
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
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
				patchs_gs_id[off] = idx;
				off++;
			}
		}
}

__global__ void getRect(
    const int W,
    const int H,
    int gs_num,
    const float *__restrict__ us,
    const float *__restrict__ areas,
    const float *__restrict__ depths,
    const dim3 grid,
    uint4 *__restrict__ gs_rects,
    uint *__restrict__ gs_patch_num)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= gs_num)
		return;

    float d = depths[idx];
    float2 u = {us[idx*2], us[idx*2 + 1]};

    float x_norm =  u.x / W * 2.f - 1.f;
    float y_norm =  u.y / H * 2.f - 1.f;
    if (abs(x_norm) > 1.3 || abs(y_norm) > 1.3 || d < 0.1 || d > 100)
    {
        gs_rects[idx] = {0, 0, 0, 0};
        gs_patch_num[idx] = 0;
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
    gs_patch_num[idx] = (rect.z - rect.x) * (rect.w - rect.y);
}

__global__ void getRange(
    const int patch_num,
    const uint64_t *__restrict__ patch_keys,
    uint2 *__restrict__ gs_ranges)
{
    const int cur_patch = blockIdx.x * blockDim.x + threadIdx.x;

    if (cur_patch >= patch_num)
        return;

    const int prv_patch = cur_patch == 0 ? 0 : cur_patch - 1;

    uint32_t cur_tile = patch_keys[cur_patch] >> 32;
    uint32_t prv_tile = patch_keys[prv_patch] >> 32;

    if (cur_patch == 0)
        gs_ranges[cur_tile].x = 0;
    else if (cur_patch == patch_num - 1)
        gs_ranges[cur_tile].y = patch_num;

    if (prv_tile != cur_tile)
    {
        gs_ranges[prv_tile].y = cur_patch;
        gs_ranges[cur_tile].x = cur_patch;
    }
}

__global__ void  draw __launch_bounds__(BLOCK * BLOCK)(
    const int W,
    const int H,
    const uint2 *__restrict__ gs_ranges,
    const uint *__restrict__ patchs_gs_id,
    const float *__restrict__ us,
    const float *__restrict__ cov2d_inv,
    const float *__restrict__ alphas,
    const float *__restrict__ colors,
    float *__restrict__ image)

{
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix = {tile.x * BLOCK + threadIdx.x,
                       tile.y * BLOCK + threadIdx.y};

    const int tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx = W * pix.y + pix.x;

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = gs_ranges[tile_idx];


	bool thread_is_finished = !inside;

	__shared__ float2 shared_pos2d[BLOCK_SIZE];
	__shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha[BLOCK_SIZE];
    __shared__ float3 shared_color[BLOCK_SIZE];

	const int gs_num = range.y - range.x;

    float3 finial_color = {0, 0, 0};

    float tau = 1.0f;

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
                         patchs_gs_id,
                         us,
                         cov2d_inv,
                         alphas,
                         colors,
                         shared_pos2d,
                         shared_cinv2d,
                         shared_alpha,
                         shared_color);
            __syncthreads();
        }

        // get 2d gaussian info for current tile (pix share the same info within the tile)
        float2 u = shared_pos2d[j];
        float3 cinv = shared_cinv2d[j];
        float alpha = shared_alpha[j];
        float3 color = shared_color[j];
        float2 d = u - pix;

        // forward.md (5.1)
        // mahalanobis squared distance for 2d gaussian to this pix
        float maha_dist = max(0.0f,  mahaSqDist(cinv, d));

        float alpha_prime = min(0.99f, alpha * exp( -0.5f * maha_dist));

        if (alpha_prime < 0.002f)
            continue;

        // forward.md (5)
        finial_color +=  tau * alpha_prime * color;

        // forward.md (5.2)
        float tau_new = tau * (1.f - alpha_prime);

        if (tau_new < 0.0001f)
        {
            thread_is_finished = true;
            continue;
        }
        tau = tau_new;
    }

    if (inside)
    {
        image[H * W * 0 + pix_idx] = finial_color.x;
        image[H * W * 1 + pix_idx] = finial_color.y;
        image[H * W * 2 + pix_idx] = finial_color.z;
    }
}

__global__ void inverseCov2D(
    int gs_num,
    const float *__restrict__ cov2d,
    float *__restrict__ cov2d_inv,
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

    const float det_inv = 1./(a*c - b*b);
    cov2d_inv[gs_id * 3 + 0] =  det_inv * c;
    cov2d_inv[gs_id * 3 + 1] = -det_inv * b;
    cov2d_inv[gs_id * 3 + 2] =  det_inv * a;
    areas[gs_id * 2 + 0] =  3 * sqrt(a);
    areas[gs_id * 2 + 1] =  3 * sqrt(c);
}

std::vector<torch::Tensor> rasterizGuassian2DCUDA(
    torch::Tensor us,
    torch::Tensor cov2d,
    torch::Tensor alphas,
    torch::Tensor depths,
    torch::Tensor colors,
    int H,
    int W)
{
    auto float_opts = us.options().dtype(torch::kFloat32);
    torch::Tensor image = torch::full({3, H, W}, 0.0, float_opts);

    //gs:    2d gaussian;  a projection of a 3d gaussian onto a 2d image
    //tile:  a 16x16 area of 2d image
    //patch: a 2d gaussian may cover on many tiles, a 2d gaussian on a tile is called a patch

    // the total number of 2d gaussian.
    int gs_num = us.sizes()[0]; 
    
    dim3 grid(DIV_ROUND_UP(W, BLOCK), DIV_ROUND_UP(H, BLOCK), 1);
	dim3 block(BLOCK, BLOCK, 1);

    thrust::device_vector<uint4> gs_rects(gs_num);
    thrust::device_vector<uint>  gs_patch_num(gs_num);
    thrust::device_vector<uint>  gs_patch_offsets(gs_num);
    thrust::device_vector<float>  cov2d_inv(gs_num * 3);
    thrust::device_vector<float>  areas(gs_num * 2);

    inverseCov2D<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        cov2d.contiguous().data<float>(),
        thrust::raw_pointer_cast(cov2d_inv.data()),
        thrust::raw_pointer_cast(areas.data()));

    getRect<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        W,
        H,
        gs_num,
        us.contiguous().data<float>(),
        thrust::raw_pointer_cast(areas.data()),
        depths.contiguous().data<float>(),
        grid,
        thrust::raw_pointer_cast(gs_rects.data()),
        thrust::raw_pointer_cast(gs_patch_num.data()));

    thrust::inclusive_scan(gs_patch_num.begin(), gs_patch_num.end(), gs_patch_offsets.begin());

    // patch_num: The total number of patches needs to be drawn
    uint patch_num = (uint)gs_patch_offsets[gs_num - 1];  // copy to cpu memory

    thrust::device_vector<uint64_t> patch_keys(patch_num);
    thrust::device_vector<uint> patch_gs_ids(patch_num);
    
    createKey<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        grid,
        thrust::raw_pointer_cast(gs_rects.data()),
        depths.contiguous().data<float>(),
        thrust::raw_pointer_cast(gs_patch_offsets.data()),
        thrust::raw_pointer_cast(patch_keys.data()),
        thrust::raw_pointer_cast(patch_gs_ids.data()));

    thrust::sort_by_key(patch_keys.begin(), patch_keys.end(), patch_gs_ids.begin());

    thrust::device_vector<uint2> gs_ranges(gs_num);

    getRange<<<DIV_ROUND_UP(patch_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        patch_num,
        thrust::raw_pointer_cast(patch_keys.data()),
        thrust::raw_pointer_cast(gs_ranges.data()));

    draw<<<grid, block>>>(
        W,
        H,
        thrust::raw_pointer_cast(gs_ranges.data()),
        thrust::raw_pointer_cast(patch_gs_ids.data()),
        us.contiguous().data<float>(),
        thrust::raw_pointer_cast(cov2d_inv.data()),
        alphas.contiguous().data<float>(),
        colors.contiguous().data<float>(),
        image.contiguous().data<float>());
    /*
    */
    return {image};
}
