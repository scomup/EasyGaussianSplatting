#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <torch/extension.h>
#include "common.h"

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
    int j = range.y + n * BLOCK_SIZE - i - 1;  // patch idx

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
    const float *__restrict__ cov2d,
    float *__restrict__ cinv2d,
    float *__restrict__ dcinv2d_dcov2d)
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
    const float det_inv2 = det_inv * det_inv;
    cinv2d[gs_id * 3 + 0] =  det_inv * c;
    cinv2d[gs_id * 3 + 1] = -det_inv * b;
    cinv2d[gs_id * 3 + 2] =  det_inv * a;
    dcinv2d_dcov2d[gs_id * 9 + 0] = -c*c*det_inv2;
    dcinv2d_dcov2d[gs_id * 9 + 1] = 2*b*c*det_inv2;
    dcinv2d_dcov2d[gs_id * 9 + 2] = -a*c*det_inv2 + det_inv;
    dcinv2d_dcov2d[gs_id * 9 + 3] = b*c*det_inv2;
    dcinv2d_dcov2d[gs_id * 9 + 4] = -2*b*b*det_inv2 - det_inv;
    dcinv2d_dcov2d[gs_id * 9 + 5] = a*b*det_inv2;
    dcinv2d_dcov2d[gs_id * 9 + 6] = -a*c*det_inv2 + det_inv;
    dcinv2d_dcov2d[gs_id * 9 + 7] = 2*a*b*det_inv2;
    dcinv2d_dcov2d[gs_id * 9 + 8] = -a*a*det_inv2;
}

__global__ void  drawBack __launch_bounds__(BLOCK * BLOCK)(
    const int W,
    const int H,
    const int *__restrict__ patch_offset_per_tile,
    const int *__restrict__ gs_id_per_patch,
    const float *__restrict__ us,
    const float *__restrict__ cinv2d,
    const float *__restrict__ alphas,
    const float *__restrict__ colors,
    const int *__restrict__ contrib,
    const float *__restrict__ final_tau,
    const float *__restrict__ dloss_dgammas,
    float *__restrict__ dloss_dalphas,
    float *__restrict__ dloss_dcolors)

{
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix = {tile.x * BLOCK + threadIdx.x,
                       tile.y * BLOCK + threadIdx.y};

    const int tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx = W * pix.y + pix.x;

	const bool inside = pix.x < W && pix.y < H;
	const int2 range = {patch_offset_per_tile[tile_idx], 
                        patch_offset_per_tile[tile_idx + 1]};


	bool thread_is_finished = !inside;

	__shared__ float2 shared_pos2d[BLOCK_SIZE];
	__shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float  shared_alpha[BLOCK_SIZE];
    __shared__ float3 shared_color[BLOCK_SIZE];
    __shared__ int shared_gsid[BLOCK_SIZE];

	const int gs_num = range.y - range.x;

    float3 gamma_cur2last = {0, 0, 0}; // the accumulated color of the pix from current to last gaussians (backward)

    float3 dloss_dgamma = {dloss_dgammas[0 * H * W + pix_idx],
                           dloss_dgammas[1 * H * W + pix_idx],
                           dloss_dgammas[2 * H * W + pix_idx]};

    float tau = final_tau[pix_idx];
    int cont = contrib[pix_idx];

    if (pix.x == 16 && pix.y == 16)
    {
        printf("backward\n");
        printf("tile_idx: %d gs_num:%d range:%d %d cont %d\n", tile_idx, gs_num, range.x, range.y, cont);
    }

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
        tau = tau / (1 - alpha_prime);

        float3 dgamma_dalphaprime = tau * (color - gamma_cur2last);
        float dalphaprime_dalpha = g;
        float3 dgamma_dalpha = dgamma_dalphaprime * dalphaprime_dalpha; 
        float dloss_dalpha = dot(dloss_dgamma, dgamma_dalpha);
    
        atomicAdd(&dloss_dalphas[gs_id], dloss_dalpha);

        float dgamma_dcolor = tau * alpha_prime;
        float3 dloss_dcolor = dloss_dgamma * dgamma_dcolor;
        atomicAdd(&dloss_dcolors[gs_id * 3 + 0], dloss_dcolor.x);
        atomicAdd(&dloss_dcolors[gs_id * 3 + 1], dloss_dcolor.y);
        atomicAdd(&dloss_dcolors[gs_id * 3 + 2], dloss_dcolor.z);

        float2 dalphaprime_du = {(-cinv2d.x*d.x - cinv2d.y*d.y) * alpha_prime, 
                                 (-cinv2d.y*d.x - cinv2d.z*d.y) * alpha_prime};
        float3 dalphaprime_dcinv2d ={-0.5 * alpha_prime * (d.x * d.x), 
                                           -alpha_prime * (d.x * d.y), 
                                     -0.5 * alpha_prime * (d.y * d.y)};

        // update gamma_cur2last for next iteration.
        gamma_cur2last = alpha_prime * color + (1 - alpha_prime) * gamma_cur2last;
        //if(gs_id == 1)
        //printf("x: %d y: %d dloss_dalpha: %f final dloss_dalpha:%f\n", 
        //    pix.x, pix.y, dloss_dalpha, dloss_dalphas[gs_id]);


        if (pix.x == 20 && pix.y == 15)
        {
            printf("x: %d y: %d  gsid: %d cont: %d \n",pix.x, pix.y, shared_gsid[j], cont);

            //printf("id: %d  dloss_dgamma: %f dgamma_dalpha:%f %f %f\n", 
            //    gs_num - i - 1,
            //    dloss_dalpha,
            //    dgamma_dalpha.x, dgamma_dalpha.y, dgamma_dalpha.z);
            //printf("id: %d  dgamma_dcolor: %f  dgamma_dalpha:%f %f %f\n", 
            //    gs_num - i - 1,
            //    dgamma_dcolor,
            //    dgamma_dalpha.x, dgamma_dalpha.y, dgamma_dalpha.z);
            //printf("id: %d  dgamma_du:\n %f %f %f\n %f %f %f\n %f %f %f\n", 
            //    gs_num - i - 1,
            //    dgamma_dcinv2d00, dgamma_dcinv2d01, dgamma_dcinv2d02,
            //    dgamma_dcinv2d10, dgamma_dcinv2d11, dgamma_dcinv2d12,
            //    dgamma_dcinv2d20, dgamma_dcinv2d21, dgamma_dcinv2d22);
        }

        /*
        

        // float dloss_dalpahprime = dot(dloss_dgamma, dgamma_dalphaprime);
        // forward.md (5.1)
        // mahalanobis squared distance for 2d gaussian to this pix
        
        float maha_dist = max(0.0f,  mahaSqDist(cinv, d));

        float alpha_prime = min(0.99f, alpha * exp( -0.5f * maha_dist));

        if (alpha_prime < 0.002f)
            continue;

        // forward.md (5)
        finial_color +=  tau * alpha_prime * color;
        cont = cont + 1;  // how many gs contribute to this pixel. 

        // forward.md (5.2)
        float tau_new = tau * (1.f - alpha_prime);

        if (tau_new < 0.0001f)
        {
            thread_is_finished = true;
            continue;
        }
        tau = tau_new;
        */
    }

    //if (inside)
    //{
    //    image[H * W * 0 + pix_idx] = finial_color.x;
    //    image[H * W * 1 + pix_idx] = finial_color.y;
    //    image[H * W * 2 + pix_idx] = finial_color.z;
    //    contrib[pix_idx] = cont;
    //    final_tau[pix_idx] = tau;
    //}
}


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
    const torch::Tensor dloss_dgammas)
{
    int gs_num = us.sizes()[0]; 
    dim3 grid(DIV_ROUND_UP(W, BLOCK), DIV_ROUND_UP(H, BLOCK), 1);
	dim3 block(BLOCK, BLOCK, 1);
    
    auto float_opts = us.options().dtype(torch::kFloat32);
    auto int_opts = us.options().dtype(torch::kInt32);
    torch::Tensor image = torch::full({3, H, W}, 0.0, float_opts);
    torch::Tensor dloss_dalphas = torch::full({gs_num, 1}, 0, float_opts);
    torch::Tensor dloss_dcolors = torch::full({gs_num, 3}, 0, float_opts);
    thrust::device_vector<float>  cinv2d(gs_num * 3);
    thrust::device_vector<float>  dcinv2d_dcov2d(gs_num * 9);

    inverseCov2DBack<<<DIV_ROUND_UP(gs_num, BLOCK_SIZE), BLOCK_SIZE>>>(
        gs_num,
        cov2d.contiguous().data_ptr<float>(),
        thrust::raw_pointer_cast(cinv2d.data()),
        thrust::raw_pointer_cast(dcinv2d_dcov2d.data()));
    cudaDeviceSynchronize();
    

    drawBack<<<grid, block>>>(
        W,
        H,
        patch_offset_per_tile.contiguous().data_ptr<int>(),
        gs_id_per_patch.contiguous().data_ptr<int>(),
        us.contiguous().data_ptr<float>(),
        thrust::raw_pointer_cast(cinv2d.data()),
        alphas.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        contrib.contiguous().data_ptr<int>(),
        final_tau.contiguous().data_ptr<float>(),
        dloss_dgammas.contiguous().data_ptr<float>(),
        dloss_dalphas.contiguous().data_ptr<float>(),
        dloss_dcolors.contiguous().data_ptr<float>());
    
    cudaDeviceSynchronize();
    

   return {dloss_dalphas, dloss_dcolors};
    
    


}
