/* Copyright:
 * This file is part of pygausplat.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include "kernel.cuh"
#include "matrix.cuh"

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

__global__ void sh2Color(
    int32_t gs_num,
    const float *__restrict__ shs,
    const float *__restrict__ pws,
    const float *__restrict__ twc,
    const int sh_dim,
    float *__restrict__ colors)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num)
        return;

    // level0
    float3 sh00 = {shs[i * sh_dim + 0], shs[i * sh_dim + 1], shs[i * sh_dim + 2]};
    float3 color = {0.5f, 0.5f, 0.5f};
    color += SH_C0_0 * sh00;

    if (sh_dim > 3)
    {
        // level1
        float x = pws[3 * i + 0] - twc[0];
        float y = pws[3 * i + 1] - twc[1];
        float z = pws[3 * i + 2] - twc[2];

        float norm = sqrt(x*x + y*y + z*z);
        x = x / norm;
        y = y / norm;
        z = z / norm;

        float3 sh10 = {shs[i * sh_dim + 3], shs[i * sh_dim + 4], shs[i * sh_dim + 5]};
        float3 sh11 = {shs[i * sh_dim + 6], shs[i * sh_dim + 7], shs[i * sh_dim + 8]};
        float3 sh12 = {shs[i * sh_dim + 9], shs[i * sh_dim + 10], shs[i * sh_dim + 11]};
        color += SH_C1_0 * y * sh10 +
                 SH_C1_1 * z * sh11 +
                 SH_C1_2 * x * sh12;

        if (sh_dim > 12)
        {
            // level2
            const float xx = x * x;
            const float yy = y * y;
            const float zz = z * z;
            const float xy = x * y;
            const float yz = y * z;
            const float xz = x * z;
            float3 sh20 = {shs[i * sh_dim + 12], shs[i * sh_dim + 13], shs[i * sh_dim + 14]};
            float3 sh21 = {shs[i * sh_dim + 15], shs[i * sh_dim + 16], shs[i * sh_dim + 17]};
            float3 sh22 = {shs[i * sh_dim + 18], shs[i * sh_dim + 19], shs[i * sh_dim + 20]};
            float3 sh23 = {shs[i * sh_dim + 21], shs[i * sh_dim + 22], shs[i * sh_dim + 23]};
            float3 sh24 = {shs[i * sh_dim + 24], shs[i * sh_dim + 25], shs[i * sh_dim + 26]};
            color += SH_C2_0 * xy * sh20 +
                     SH_C2_1 * yz * sh21 +
                     SH_C2_2 * (2.0f * zz - xx - yy) * sh22 +
                     SH_C2_3 * xz * sh23 +
                     SH_C2_4 * (xx - yy) * sh24;
            if (sh_dim > 27)
            {
                // level3
                float3 sh30 = {shs[i * sh_dim + 27], shs[i * sh_dim + 28], shs[i * sh_dim + 29]};
                float3 sh31 = {shs[i * sh_dim + 30], shs[i * sh_dim + 31], shs[i * sh_dim + 32]};
                float3 sh32 = {shs[i * sh_dim + 33], shs[i * sh_dim + 34], shs[i * sh_dim + 35]};
                float3 sh33 = {shs[i * sh_dim + 36], shs[i * sh_dim + 37], shs[i * sh_dim + 38]};
                float3 sh34 = {shs[i * sh_dim + 39], shs[i * sh_dim + 40], shs[i * sh_dim + 41]};
                float3 sh35 = {shs[i * sh_dim + 42], shs[i * sh_dim + 43], shs[i * sh_dim + 44]};
                float3 sh36 = {shs[i * sh_dim + 45], shs[i * sh_dim + 46], shs[i * sh_dim + 47]};
                if(i == 0)
                {
                    float3 c31 = SH_C3_1 * xy * z * sh31 ;
                    float3 c32 = SH_C3_2 * y * (4.0f * zz - xx - yy) * sh32;
                    float3 c33 = SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh33;
                    float3 c34 = SH_C3_4 * x * (4.0f * zz - xx - yy) * sh34;
                    float3 c35 = SH_C3_5 * z * (xx - yy) * sh35;
                    float3 c36 = SH_C3_6 * x * (xx - 3.0f * yy) * sh36;
                    float3 c = color + c31 + c32 + c33 + c34 + c35 + c36;
                }
                color += SH_C3_0 * y * (3.0f * xx - yy) * sh30 +
                         SH_C3_1 * xy * z * sh31 +
                         SH_C3_2 * y * (4.0f * zz - xx - yy) * sh32 +
                         SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh33 +
                         SH_C3_4 * x * (4.0f * zz - xx - yy) * sh34 +
                         SH_C3_5 * z * (xx - yy) * sh35 +
                         SH_C3_6 * x * (xx - 3.0f * yy) * sh36;
            }
        }
    }
    colors[3 * i + 0] = color.x;
    colors[3 * i + 1] = color.y;
    colors[3 * i + 2] = color.z;
}
