/* Copyright:
 * This file is part of pygausplat.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include "common.cuh"

inline __device__ void fetch2sharedBack(
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
    float3 *shared_color,
    int *shared_gsid)
{
    int i = blockDim.x * threadIdx.y + threadIdx.x;  // block idx
    int j = range.y - n * BLOCK_SIZE - i - 1;  // patch idx

    if (j >= range.x)
    {
        int gs_id = gs_id_per_patch[j];
        shared_gsid[i] = gs_id;
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

__global__ void inverseCov2DBack(
    int gs_num,
    const float *__restrict__ cov2ds,
    float *__restrict__ cinv2ds,
    float *__restrict__ dcinv2d_dcov2ds)
{
    // compute inverse of cov2d
    // Determine the drawing area of 2d Gaussian.

    const int gs_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (gs_id >= gs_num)
		return;
    // forward.md 5.3
    const float a = cov2ds[gs_id * 3];
    const float b = cov2ds[gs_id * 3 + 1];
    const float c = cov2ds[gs_id * 3 + 2];

    const float det = a*c - b*b;
    if (det == 0.0f)
		return;

    const float det_inv = 1.0f/(det);
    const float det_inv2 = 1.0f / ((det * det) + 0.0000001f);

    cinv2ds[gs_id * 3 + 0] =  det_inv * c;
    cinv2ds[gs_id * 3 + 1] = -det_inv * b;
    cinv2ds[gs_id * 3 + 2] =  det_inv * a;
    dcinv2d_dcov2ds[gs_id * 9 + 0] = -c*c*det_inv2;
    dcinv2d_dcov2ds[gs_id * 9 + 1] = 2*b*c*det_inv2;
    dcinv2d_dcov2ds[gs_id * 9 + 2] = -a*c*det_inv2 + det_inv;
    dcinv2d_dcov2ds[gs_id * 9 + 3] = b*c*det_inv2;
    dcinv2d_dcov2ds[gs_id * 9 + 4] = -2*b*b*det_inv2 - det_inv;
    dcinv2d_dcov2ds[gs_id * 9 + 5] = a*b*det_inv2;
    dcinv2d_dcov2ds[gs_id * 9 + 6] = -a*c*det_inv2 + det_inv;
    dcinv2d_dcov2ds[gs_id * 9 + 7] = 2*a*b*det_inv2;
    dcinv2d_dcov2ds[gs_id * 9 + 8] = -a*a*det_inv2;


}

__global__ void calcDlossDcov2d(
    int gs_num,
    const float *__restrict__ dloss_dcinv2ds,
    const float *__restrict__ dcinv2d_dcov2ds,
    float *__restrict__ dloss_dcov2ds)
{
    const int gs_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (gs_id >= gs_num)
		return;
        
    float3 dloss_dcinv2d = {dloss_dcinv2ds[gs_id*3 + 0],
                            dloss_dcinv2ds[gs_id*3 + 1],
                            dloss_dcinv2ds[gs_id*3 + 2]};

    if (dloss_dcinv2d.x == 0.0f &&
        dloss_dcinv2d.y == 0.0f &&
        dloss_dcinv2d.z == 0.0f)
        return;

    float3 dcinv2d_dcov2d0 = {dcinv2d_dcov2ds[gs_id*9 + 0],
                              dcinv2d_dcov2ds[gs_id*9 + 3],
                              dcinv2d_dcov2ds[gs_id*9 + 6]};

    float3 dcinv2d_dcov2d1 = {dcinv2d_dcov2ds[gs_id*9 + 1],
                              dcinv2d_dcov2ds[gs_id*9 + 4],
                              dcinv2d_dcov2ds[gs_id*9 + 7]};

    float3 dcinv2d_dcov2d2 = {dcinv2d_dcov2ds[gs_id*9 + 2],
                              dcinv2d_dcov2ds[gs_id*9 + 5],
                              dcinv2d_dcov2ds[gs_id*9 + 8]};

    dloss_dcov2ds[gs_id*3 + 0] = dot(dloss_dcinv2d, dcinv2d_dcov2d0);
    dloss_dcov2ds[gs_id*3 + 1] = dot(dloss_dcinv2d, dcinv2d_dcov2d1);
    dloss_dcov2ds[gs_id*3 + 2] = dot(dloss_dcinv2d, dcinv2d_dcov2d2);
    // if(isnan(dot(dloss_dcinv2d, dcinv2d_dcov2d0)))
    // {
    //     printf("gs_id: %d gs_num %d \n", gs_id, gs_num);
    //     printf("dloss_dcinv2d: %f %f %f \n", dloss_dcinv2d.x, dloss_dcinv2d.y, dloss_dcinv2d.z);
    //     printf("dcinv2d_dcov2d0: %f %f %f \n", dcinv2d_dcov2d0.x, dcinv2d_dcov2d0.y, dcinv2d_dcov2d0.z);
    //     printf("dcinv2d_dcov2d0: %f\n", dcinv2d_dcov2ds[gs_id*9 + 0]);
    // }

}

__global__ void  drawBack __launch_bounds__(BLOCK * BLOCK)(
    const int width,
    const int height,
    const int *__restrict__ patch_range_per_tile,
    const int *__restrict__ gs_id_per_patch,
    const float *__restrict__ us,
    const float *__restrict__ cinv2d,
    const float *__restrict__ alphas,
    const float *__restrict__ colors,
    const int *__restrict__ contrib,
    const float *__restrict__ final_tau,
    const float *__restrict__ dloss_dgammas,
    float *__restrict__ dloss_dus,
    float *__restrict__ dloss_dcinv2ds,
    float *__restrict__ dloss_dalphas,
    float *__restrict__ dloss_dcolors)

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
    __shared__ int shared_gsid[BLOCK_SIZE];


    float3 gamma_cur2last = {0, 0, 0}; // the accumulated color of the pix from current to last gaussians (backward)

    float3 dloss_dgamma = {dloss_dgammas[0 * height * width + pix_idx],
                           dloss_dgammas[1 * height * width + pix_idx],
                           dloss_dgammas[2 * height * width + pix_idx]};

    float tau = final_tau[pix_idx];
    int cont = contrib[pix_idx];

    // if (pix.x == 687 && pix.y == 269)
    // {
    //     printf("tile%d gs_num %d, cont%d range %d %d\n", tile_idx, gs_num, cont, range.x, range.y);
    // }

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
            // fetch to shared memory by backward order 
            fetch2sharedBack(i / BLOCK_SIZE,
                         range,
                         gs_id_per_patch,
                         us,
                         cinv2d,
                         alphas,
                         colors,
                         shared_pos2d,
                         shared_cinv2d,
                         shared_alpha,
                         shared_color,
                         shared_gsid);
            __syncthreads();
        }

        // becasuse we fetch data by backward, we skip i < gs_num - cont
        if ( i < gs_num - cont)
            continue;

        float2 u = shared_pos2d[j];
        float3 cinv2d = shared_cinv2d[j];
        float alpha = shared_alpha[j];
        float3 color = shared_color[j];
        int gs_id = shared_gsid[j];
        float2 d = u - pix;
        float maha_dist = max(0.0f,  mahaSqDist(cinv2d, d));
        float g = exp(-0.5f * maha_dist);
        float alpha_prime = min(0.99f, alpha * g);

        if (alpha_prime < 0.002f)
            continue;

        if (gs_id == 687 && pix.y == 269)
        {
            printf("tile%d gs_num %d, cont%d range %d %d\n", tile_idx, gs_num, cont, range.x, range.y);
        }


        tau = tau / (1 - alpha_prime);

        float3 dgamma_dalphaprime = tau * (color - gamma_cur2last);
        float dalphaprime_dalpha = g;
        float dloss_dalphaprime = dot(dloss_dgamma, dgamma_dalphaprime); 
        float dloss_dalpha = dloss_dalphaprime * dalphaprime_dalpha;
    
        atomicAdd(&dloss_dalphas[gs_id], dloss_dalpha);

        float dgamma_dcolor = tau * alpha_prime;
        float3 dloss_dcolor = dloss_dgamma * dgamma_dcolor;
        atomicAdd(&dloss_dcolors[gs_id * 3 + 0], dloss_dcolor.x);
        atomicAdd(&dloss_dcolors[gs_id * 3 + 1], dloss_dcolor.y);
        atomicAdd(&dloss_dcolors[gs_id * 3 + 2], dloss_dcolor.z);

        float2 dalphaprime_du = {(-cinv2d.x*d.x - cinv2d.y*d.y) * alpha_prime, 
                                 (-cinv2d.y*d.x - cinv2d.z*d.y) * alpha_prime};
        float2 dloss_du = dloss_dalphaprime * dalphaprime_du;

        atomicAdd(&dloss_dus[gs_id * 2 + 0], dloss_du.x);
        atomicAdd(&dloss_dus[gs_id * 2 + 1], dloss_du.y);

        float3 dalphaprime_dcinv2d = {-0.5f * alpha_prime * (d.x * d.x),
                                      -alpha_prime * (d.x * d.y),
                                      -0.5f * alpha_prime * (d.y * d.y)};
        float3 dloss_dcinv2d = dloss_dalphaprime * dalphaprime_dcinv2d;
        
        atomicAdd(&dloss_dcinv2ds[gs_id * 3 + 0], dloss_dcinv2d.x);
        atomicAdd(&dloss_dcinv2ds[gs_id * 3 + 1], dloss_dcinv2d.y);
        atomicAdd(&dloss_dcinv2ds[gs_id * 3 + 2], dloss_dcinv2d.z);
    
        // update gamma_cur2last for next iteration.
        gamma_cur2last = alpha_prime * color + (1 - alpha_prime) * gamma_cur2last;
    }
}

std::vector<torch::Tensor> backward(
    const int height,
    const int width,
    const torch::Tensor us,
    const torch::Tensor cov2ds,
    const torch::Tensor alphas,
    const torch::Tensor depths,
    const torch::Tensor colors,
    const torch::Tensor contrib,
    const torch::Tensor final_tau, 
    const torch::Tensor patch_range_per_tile, 
    const torch::Tensor gs_id_per_patch,
    const torch::Tensor dloss_dgammas)
{
    int gs_num = us.sizes()[0]; 
    dim3 grid(DIV_ROUND_UP(width, BLOCK), DIV_ROUND_UP(height, BLOCK), 1);
	dim3 block(BLOCK, BLOCK, 1);
    
    auto float_opts = us.options().dtype(torch::kFloat32);
    auto int_opts = us.options().dtype(torch::kInt32);
    torch::Tensor image = torch::full({3, height, width}, 0.0, float_opts);
    torch::Tensor dloss_dalphas = torch::full({gs_num}, 0, float_opts);
    torch::Tensor dloss_dcolors = torch::full({gs_num, 3}, 0, float_opts);
    torch::Tensor dloss_dcinv2ds = torch::full({gs_num, 3}, 0, float_opts);
    torch::Tensor dloss_dcov2ds = torch::full({gs_num, 3}, 0, float_opts);
    torch::Tensor dloss_dus = torch::full({gs_num, 2}, 0, float_opts);

    torch::Tensor cinv2ds = torch::full({gs_num, 3}, 0, float_opts);
    torch::Tensor dcinv2d_dcov2ds = torch::full({gs_num, 9}, 0, float_opts);

    inverseCov2DBack<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        cov2ds.contiguous().data_ptr<float>(),
        cinv2ds.contiguous().data_ptr<float>(),
        dcinv2d_dcov2ds.contiguous().data_ptr<float>());
    cudaDeviceSynchronize();

    drawBack<<<grid, block>>>(
        width,
        height,
        patch_range_per_tile.contiguous().data_ptr<int>(),
        gs_id_per_patch.contiguous().data_ptr<int>(),
        us.contiguous().data_ptr<float>(),
        cinv2ds.contiguous().data_ptr<float>(),
        alphas.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        contrib.contiguous().data_ptr<int>(),
        final_tau.contiguous().data_ptr<float>(),
        dloss_dgammas.contiguous().data_ptr<float>(),
        dloss_dus.contiguous().data_ptr<float>(),
        dloss_dcinv2ds.contiguous().data_ptr<float>(),
        dloss_dalphas.contiguous().data_ptr<float>(),
        dloss_dcolors.contiguous().data_ptr<float>());
    cudaDeviceSynchronize();

    calcDlossDcov2d<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        dloss_dcinv2ds.contiguous().data_ptr<float>(),
        dcinv2d_dcov2ds.contiguous().data_ptr<float>(),
        dloss_dcov2ds.contiguous().data_ptr<float>());
    
   return {dloss_dus, dloss_dcov2ds, dloss_dalphas, dloss_dcolors};
}
