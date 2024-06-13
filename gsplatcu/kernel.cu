/* Copyright:
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include "kernel.cuh"
#include "matrix.cuh"

#define MIN_DEPTH (0.2)
#define BAD_MARKER (-1.f)

inline __device__ void fetch2shared(
    int32_t n,
    const bool is_backward,
    const int2 range,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float3 *__restrict__ colors,
    float2 *shared_pos2d,
    float3 *shared_cinv2d,
    float *shared_alpha,
    float3 *shared_color,
    int *shared_gsid)
{
    int i = blockDim.x * threadIdx.y + threadIdx.x;  // block idx
    int j; // patch idx
    if (is_backward == true)
        j = range.y - n * BLOCK_SIZE - i - 1;
    else
        j = range.x + n * BLOCK_SIZE + i;

    if (j < range.y && j >= range.x)
    {
        int gs_id = gsid_per_patch[j];
        shared_gsid[i] = gs_id;
        shared_pos2d[i] = us[gs_id];
        shared_cinv2d[i] = cinv2ds[gs_id];
        shared_alpha[i] = alphas[gs_id];
        shared_color[i] = colors[gs_id];
    }
}

__global__ void createKeys(
    const int gs_num,
    const float* __restrict__ depths,
    const uint32_t* __restrict__ patch_offset_per_gs,
    const uint4* __restrict__ rects,
    const dim3 grid,
    uint64_t* __restrict__ patch_keys,
    int* __restrict__ gsid_per_patch)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
        return;

    const float depth = depths[i];
    if (depth < MIN_DEPTH)
        return;

    uint32_t off = (i == 0) ? 0 : patch_offset_per_gs[i - 1];
    uint4 rect = rects[i];

    for (uint y = rect.y; y < rect.w; y++)
    {
        for (uint x = rect.x; x < rect.z; x++)
        {
            uint64_t key = (y * grid.x + x);
            key <<= 32;
            uint32_t depth_cm = depth * 1000; // mm
            key |= depth_cm;
            patch_keys[off] = key;
            gsid_per_patch[off] = i;
            off++;
        }
    }
}

__global__ void getRects(
    const int gs_num,
    const float* __restrict__ us,
    int2* __restrict__ areas,
    float* __restrict__ depths,
    const dim3 grid,
    uint4 *__restrict__ gs_rects,
    uint* __restrict__ patch_num_per_gs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gs_num)
        return;

    patch_num_per_gs[idx] = 0;

    if (depths[idx] < MIN_DEPTH)
        return;

    float2 u = { us[2 * idx], us[2 * idx + 1]};

    float xs = areas[idx].x;
    float ys = areas[idx].y;

    uint4 rect = {
        min(grid.x, max((int)0, (int)((u.x - xs) / BLOCK))),            // min_x
        min(grid.y, max((int)0, (int)((u.y - ys) / BLOCK))),            // min_y
        min(grid.x, max((int)0, (int)(DIV_ROUND_UP(u.x + xs, BLOCK)))), // max_x
        min(grid.y, max((int)0, (int)(DIV_ROUND_UP(u.y + ys, BLOCK))))  // max_y
    };

    uint n = (rect.w - rect.y) * (rect.z - rect.x);

    if (n == 0)
    {
        depths[idx] = BAD_MARKER;
        areas[idx] = {0, 0};
        return;
    }
    gs_rects[idx] = rect;
    patch_num_per_gs[idx] = n;
}


__global__ void getRanges(
    const int patch_num,
    const uint64_t *__restrict__ patch_keys,
    int2 *__restrict__ patch_range_per_tile)
{
    const int cur_patch = blockIdx.x * blockDim.x + threadIdx.x;

    if (cur_patch >= patch_num)
        return;

    const int prv_patch = cur_patch == 0 ? 0 : cur_patch - 1;

    uint32_t cur_tile = patch_keys[cur_patch] >> 32;
    uint32_t prv_tile = patch_keys[prv_patch] >> 32;

    if (cur_patch == 0)
        patch_range_per_tile[cur_tile].x = 0;
    else if (cur_patch == patch_num - 1)
        patch_range_per_tile[cur_tile].y = patch_num;

    if (prv_tile != cur_tile)
    {
        patch_range_per_tile[prv_tile].y = cur_patch;
        patch_range_per_tile[cur_tile].x = cur_patch;
    }
}

__global__ void draw __launch_bounds__(BLOCK *BLOCK)(
    const int width,
    const int height,
    const int2 *__restrict__ patch_range_per_tile,
    const int *__restrict__ gsid_per_patch,
    const float2 *__restrict__ us,
    const float3 *__restrict__ cinv2ds,
    const float *__restrict__ alphas,
    const float3 *__restrict__ colors,
    float *__restrict__ image,
    int *__restrict__ contrib,
    float *__restrict__ final_tau)

{
    /*
    More details are discussed in section 5 of the forward.pdf
    */
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix = {tile.x * BLOCK + threadIdx.x,
                       tile.y * BLOCK + threadIdx.y};

    const int tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx = width * pix.y + pix.x;

    const bool inside = pix.x < width && pix.y < height;
    const int2 range = patch_range_per_tile[tile_idx];

    const int gs_num = range.y - range.x;

    // not patch for this tile.
    if (gs_num == 0)
        return;

    bool thread_is_finished = !inside;

    __shared__ float2 shared_pos2d[BLOCK_SIZE];
    __shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float shared_alpha[BLOCK_SIZE];
    __shared__ float3 shared_color[BLOCK_SIZE];
    __shared__ int shared_gsid[BLOCK_SIZE];

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
                         false,
                         range,
                         gsid_per_patch,
                         us,
                         cinv2ds,
                         alphas,
                         colors,
                         shared_pos2d,
                         shared_cinv2d,
                         shared_alpha,
                         shared_color,
                         shared_gsid);
            __syncthreads();
        }

        if (thread_is_finished)
            continue;

        // get 2d gaussian info for current tile (pix share the same info within the tile)
        float2 u = shared_pos2d[j];
        float3 cinv = shared_cinv2d[j];
        float alpha = shared_alpha[j];
        float3 color = shared_color[j];
        float2 d = u - pix;

        cont_tmp = cont_tmp + 1;

        // forward.pdf (F.5.1)
        // mahalanobis squared distance for 2d gaussian to this pix
        float maha_dist = max(0.0f, mahaSqDist(cinv, d));

        float alpha_prime = min(0.99f, alpha * exp(-0.5f * maha_dist));
        if (alpha_prime < 0.002f)
            continue;

        // forward.pdf (F.5)
        finial_color += tau * alpha_prime * color;
        cont = cont_tmp; // how many gs contribute to this pixel.

        // forward.pdf (F.5.2)
        tau = tau * (1.f - alpha_prime);

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
    const float3 *__restrict__ cov2ds,
    float *__restrict__ depths,
    float3 *__restrict__ cinv2ds,
    int2 *__restrict__ areas,
    float *__restrict__ dcinv2d_dcov2ds)
{
    // compute inverse of cov2d
    // Determine the drawing area of 2d Gaussian.

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
        return;

    if (depths[i] < MIN_DEPTH)
        return;

    // forward.pdf (F.5.3)
    const float3 cov2d = cov2ds[i];
    const float a = cov2d.x;
    const float b = cov2d.y;
    const float c = cov2d.z;

    // forward.pdf (F.5.3)
    const float det_inv = 1.f / (a * c - b * b);
    if (isnan(det_inv))
    {
        depths[i] = BAD_MARKER;
        return;
    }

    cinv2ds[i] = {det_inv * c, -det_inv * b, det_inv * a};
    areas[i] = {(int)ceil(3 * sqrt(abs(a))), (int)ceil(3 * sqrt(abs(c)))};

    if (dcinv2d_dcov2ds != nullptr)
    {
        // backward.pdf (B.5.3)
        const float det2_inv = det_inv * det_inv;
        dcinv2d_dcov2ds[i * 9 + 0] = -c * c * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 1] = 2 * b * c * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 2] = -a * c * det2_inv + det_inv;
        dcinv2d_dcov2ds[i * 9 + 3] = b * c * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 4] = -2 * b * b * det2_inv - det_inv;
        dcinv2d_dcov2ds[i * 9 + 5] = a * b * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 6] = -a * c * det2_inv + det_inv;
        dcinv2d_dcov2ds[i * 9 + 7] = 2 * a * b * det2_inv;
        dcinv2d_dcov2ds[i * 9 + 8] = -a * a * det2_inv;
    }
}

__global__ void computeCov3D(
    int32_t gs_num,
    const float4 *__restrict__ rots,
    const float3 *__restrict__ scales,
    const float *__restrict__ depths,
    float *__restrict__ cov3ds,
    float *__restrict__ dcov3d_drots,
    float *__restrict__ dcov3d_dscales)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
        return;

    if (depths[i] < MIN_DEPTH)
        return;
    const float4 rot = rots[i];
    const float3 s = scales[i];
    const float w = rot.x;
    const float x = rot.y;
    const float y = rot.z;
    const float z = rot.w;
    const float s0 = s.x;
    const float s1 = s.y;
    const float s2 = s.z;
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float xw = x * w;
    const float yw = y * w;
    const float zw = z * w;

    Matrix<3, 3> R = {
        1.f - 2.f * (yy + zz), 2.f * (xy - zw), 2.f * (xz + yw),
        2.f * (xy + zw), 1.f - 2.f * (xx + zz), 2.f * (yz - xw),
        2.f * (xz - yw), 2.f * (yz + xw), 1.f - 2.f * (xx + yy)};

    Matrix<3, 3> M = {
        R(0, 0) * s0, R(0, 1) * s1, R(0, 2) * s2,
        R(1, 0) * s0, R(1, 1) * s1, R(1, 2) * s2,
        R(2, 0) * s0, R(2, 1) * s1, R(2, 2) * s2};
    // forward.pdf (F.2)
    Matrix<3, 3> Sigma = M * M.transpose();

    cov3ds[i * 6 + 0] = Sigma(0, 0);
    cov3ds[i * 6 + 1] = Sigma(0, 1);
    cov3ds[i * 6 + 2] = Sigma(0, 2);
    cov3ds[i * 6 + 3] = Sigma(1, 1);
    cov3ds[i * 6 + 4] = Sigma(1, 2);
    cov3ds[i * 6 + 5] = Sigma(2, 2);

    if (dcov3d_drots != nullptr && dcov3d_dscales != nullptr)
    {
        Matrix<6, 9> dcov3d_dm = {2 * M(0, 0), 2 * M(0, 1), 2 * M(0, 2), 0, 0, 0, 0, 0, 0,
                                  M(1, 0), M(1, 1), M(1, 2), M(0, 0), M(0, 1), M(0, 2), 0, 0, 0,
                                  M(2, 0), M(2, 1), M(2, 2), 0, 0, 0, M(0, 0), M(0, 1), M(0, 2),
                                  0, 0, 0, 2 * M(1, 0), 2 * M(1, 1), 2 * M(1, 2), 0, 0, 0,
                                  0, 0, 0, M(2, 0), M(2, 1), M(2, 2), M(1, 0), M(1, 1), M(1, 2),
                                  0, 0, 0, 0, 0, 0, 2 * M(2, 0), 2 * M(2, 1), 2 * M(2, 2)};
        Matrix<9, 4> dm_rot = {0, 0, -4 * s0 * y, -4 * s0 * z,
                               -2 * s1 * z, 2 * s1 * y, 2 * s1 * x, -2 * s1 * w,
                               2 * s2 * y, 2 * s2 * z, 2 * s2 * w, 2 * s2 * x,
                               2 * s0 * z, 2 * s0 * y, 2 * s0 * x, 2 * s0 * w,
                               0, -4 * s1 * x, 0, -4 * s1 * z,
                               -2 * s2 * x, -2 * s2 * w, 2 * s2 * z, 2 * s2 * y,
                               -2 * s0 * y, 2 * s0 * z, -2 * s0 * w, 2 * s0 * x,
                               2 * s1 * x, 2 * s1 * w, 2 * s1 * z, 2 * s1 * y,
                               0, -4 * s2 * x, -4 * s2 * y, 0};
        Matrix<9, 3> dm_scale = {R(0, 0), 0, 0,
                                 0, R(0, 1), 0,
                                 0, 0, R(0, 2),
                                 R(1, 0), 0, 0,
                                 0, R(1, 1), 0,
                                 0, 0, R(1, 2),
                                 R(2, 0), 0, 0,
                                 0, R(2, 1), 0,
                                 0, 0, R(2, 2)};
        // backward.pdf (B.2a)
        Matrix<6, 4> dcov3d_drot = dcov3d_dm * dm_rot;
        // backward.pdf (B.2b)
        Matrix<6, 3> dcov3d_dscale = dcov3d_dm * dm_scale;

        // copy to global memory
        const int offset_dr = i * 24;
        for (int j = 0; j < 24; j++)
        {
            dcov3d_drots[offset_dr + j] = dcov3d_drot(j);
        }
        const int offset_ds = i * 18;
        for (int j = 0; j < 18; j++)
        {
            dcov3d_dscales[offset_ds + j] = dcov3d_dscale(j);
        }
    }
}

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
    float *__restrict__ dcov2d_dcov3ds,
    float *__restrict__ dcov2d_dpcs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
        return;

    if(depths[i] < MIN_DEPTH)
        return;

    float3 pc = pcs[i];
    float x = pc.x;
    float y = pc.y;
    const float z = pc.z;
    const float a = cov3ds[i * 6 + 0];
    const float b = cov3ds[i * 6 + 1];
    const float c = cov3ds[i * 6 + 2];
    const float d = cov3ds[i * 6 + 3];
    const float e = cov3ds[i * 6 + 4];
    const float f = cov3ds[i * 6 + 5];

    const float limx = 1.3f * tan_fovx;
    const float limy = 1.3f * tan_fovy;
    x = min(limx, max(-limx, x / z)) * z;
    y = min(limy, max(-limy, y / z)) * z;


    float z2 = z * z;

    Matrix<3, 3> R = {
        Rcw[0], Rcw[1], Rcw[2],
        Rcw[3], Rcw[4], Rcw[5],
        Rcw[6], Rcw[7], Rcw[8]};

    Matrix<2, 3> J = {
        focal_x / z, 0.0f, -(focal_x * x) / z2,
        0.0f, focal_y / z, -(focal_y * y) / z2};

    Matrix<3, 3> Sigma = {
        a, b, c,
        b, d, e,
        c, e, f};

    Matrix<2, 3> M = J * R;

    // forward.pdf (F.3)
    Matrix<2, 2> Sigma_prime = M * Sigma * M.transpose();

    // make sure the cov2d is not too small.
    cov2ds[i] = {Sigma_prime(0, 0) + 0.3f,
                 Sigma_prime(0, 1),
                 Sigma_prime(1, 1) + 0.3f};

    if (dcov2d_dcov3ds != nullptr && dcov2d_dpcs != nullptr)
    {
        // backward.pdf (B.3a)
        Matrix<3, 6> dcov2d_dcov3d = {M(0, 0) * M(0, 0),
                                      2 * M(0, 0) * M(0, 1),
                                      2 * M(0, 0) * M(0, 2),
                                      M(0, 1) * M(0, 1),
                                      2 * M(0, 1) * M(0, 2),
                                      M(0, 2) * M(0, 2),
                                      M(0, 0) * M(1, 0),
                                      M(0, 0) * M(1, 1) + M(0, 1) * M(1, 0),
                                      M(0, 0) * M(1, 2) + M(0, 2) * M(1, 0),
                                      M(0, 1) * M(1, 1),
                                      M(0, 1) * M(1, 2) + M(0, 2) * M(1, 1),
                                      M(0, 2) * M(1, 2),
                                      M(1, 0) * M(1, 0),
                                      2 * M(1, 0) * M(1, 1),
                                      2 * M(1, 0) * M(1, 2),
                                      M(1, 1) * M(1, 1),
                                      2 * M(1, 1) * M(1, 2),
                                      M(1, 2) * M(1, 2)};

        Matrix<3, 6> dcov2d_dm = {2 * a * M(0, 0) + 2 * b * M(0, 1) + 2 * c * M(0, 2),
                                  2 * b * M(0, 0) + 2 * d * M(0, 1) + 2 * e * M(0, 2),
                                  2 * c * M(0, 0) + 2 * e * M(0, 1) + 2 * f * M(0, 2),
                                  0, 0, 0,
                                  a * M(1, 0) + b * M(1, 1) + c * M(1, 2),
                                  b * M(1, 0) + d * M(1, 1) + e * M(1, 2),
                                  c * M(1, 0) + e * M(1, 1) + f * M(1, 2),
                                  a * M(0, 0) + b * M(0, 1) + c * M(0, 2),
                                  b * M(0, 0) + d * M(0, 1) + e * M(0, 2),
                                  c * M(0, 0) + e * M(0, 1) + f * M(0, 2),
                                  0, 0, 0,
                                  2 * a * M(1, 0) + 2 * b * M(1, 1) + 2 * c * M(1, 2),
                                  2 * b * M(1, 0) + 2 * d * M(1, 1) + 2 * e * M(1, 2),
                                  2 * c * M(1, 0) + 2 * e * M(1, 1) + 2 * f * M(1, 2)};

        const float z2_inv = 1.f / (z * z);
        const float z3_inv = z2_inv / z;
        Matrix<6, 3> dm_dpc = {-focal_x * R(2, 0) * z2_inv, 0, -focal_x * R(0, 0) * z2_inv + 2 * focal_x * R(2, 0) * x * z3_inv,
                               -focal_x * R(2, 1) * z2_inv, 0, -focal_x * R(0, 1) * z2_inv + 2 * focal_x * R(2, 1) * x * z3_inv,
                               -focal_x * R(2, 2) * z2_inv, 0, -focal_x * R(0, 2) * z2_inv + 2 * focal_x * R(2, 2) * x * z3_inv,
                               0, -focal_y * R(2, 0) * z2_inv, -focal_y * R(1, 0) * z2_inv + 2 * focal_y * R(2, 0) * y * z3_inv,
                               0, -focal_y * R(2, 1) * z2_inv, -focal_y * R(1, 1) * z2_inv + 2 * focal_y * R(2, 1) * y * z3_inv,
                               0, -focal_y * R(2, 2) * z2_inv, -focal_y * R(1, 2) * z2_inv + 2 * focal_y * R(2, 2) * y * z3_inv};

        // backward.pdf (B.3b)
        Matrix<3, 3> dcov2d_dpc = dcov2d_dm * dm_dpc;

        // copy to global memory
        const int offset_dpc = i * 9;
        for (int j = 0; j < 9; j++)
        {
            dcov2d_dpcs[offset_dpc + j] = dcov2d_dpc(j);
        }
        const int offset_dcov3d = i * 18;
        for (int j = 0; j < 18; j++)
        {
            dcov2d_dcov3ds[offset_dcov3d + j] = dcov2d_dcov3d(j);
        }
    }
}

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
    float *__restrict__ du_dpcs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= gs_num)
        return;

    Matrix<3, 1> pw = {pws[i].x, pws[i].y, pws[i].z};

    Matrix<3, 1> t = {tcw[0].x, tcw[0].y, tcw[0].z};

    Matrix<3, 3> R = {
        Rcw[0], Rcw[1], Rcw[2],
        Rcw[3], Rcw[4], Rcw[5],
        Rcw[6], Rcw[7], Rcw[8]};

    // forward.pdf (F.1.1)
    Matrix<3, 1> pc = R * pw + t;

    const float x = pc(0);
    const float y = pc(1);
    const float z = pc(2);

    if (z < MIN_DEPTH)
    {
        depths[i] = BAD_MARKER;
        return;
    }

    const float z_inv = 1.f / z;
    const float z2_inv = z_inv * z_inv;

    const float x_focal_x = x * focal_x;
    const float y_focal_y = y * focal_y;

    // forward.pdf (F.1.2)
    const float u0 = x_focal_x * z_inv + center_x;
    const float u1 = y_focal_y * z_inv + center_y;

    // make sure the cov2d is not too small.
    us[i] = {u0, u1};
    pcs[i] = {x, y, z};
    depths[i] = z;

    if (du_dpcs != nullptr)
    {
        // backward.pdf (B.1.2)
        du_dpcs[i * 6 + 0] = focal_x * z_inv;
        du_dpcs[i * 6 + 2] = -x_focal_x * z2_inv;
        du_dpcs[i * 6 + 4] = focal_y * z_inv;
        du_dpcs[i * 6 + 5] = -y_focal_y * z2_inv;
    }
}

__global__ void sh2Color(
    int32_t gs_num,
    const float3 *__restrict__ shs,
    const float3 *__restrict__ pws,
    const float3 *__restrict__ twc,
    const int sh_dim3,
    float3 *__restrict__ colors,
    float *__restrict__ dcolor_dshs,
    float *__restrict__ dcolor_dpws)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gs_num)
        return;

    float dc_dsh1, dc_dsh2, dc_dsh3;
    float dc_dsh4, dc_dsh5, dc_dsh6;
    float dc_dsh7, dc_dsh8, dc_dsh9;
    float dc_dsh10, dc_dsh11, dc_dsh12;
    float dc_dsh13, dc_dsh14, dc_dsh15;

    float3 sh1, sh2, sh3;
    float3 sh4, sh5, sh6;
    float3 sh7, sh8, sh9;
    float3 sh10, sh11, sh12;
    float3 sh13, sh14, sh15;

    float d0, d1, d2;
    float normd_inv;
    float x, y, z;
    float xx, yy, zz, xy, xz, yz;

    // level0
    float3 sh0 = shs[i * sh_dim3 + 0];
    float3 color = {0.5f, 0.5f, 0.5f};
    color += SH_C0_0 * sh0;

    if (sh_dim3 > 1)
    {
        // level1
        float3 d = pws[i] - twc[0];
        d0 = d.x;
        d1 = d.y;
        d2 = d.z;
        normd_inv = 1. / sqrt(d0 * d0 + d1 * d1 + d2 * d2);
        float3 dir = d * normd_inv;
        x = dir.x;
        y = dir.y;
        z = dir.z;

        sh1 = shs[i * sh_dim3 + 1];
        sh2 = shs[i * sh_dim3 + 2];
        sh3 = shs[i * sh_dim3 + 3];

        dc_dsh1 = SH_C1_0 * y;
        dc_dsh2 = SH_C1_1 * z;
        dc_dsh3 = SH_C1_2 * x;

        color += dc_dsh1 * sh1 + dc_dsh2 * sh2 + dc_dsh3 * sh3;

        if (sh_dim3 > 4)
        {
            // level2
            xx = x * x;
            yy = y * y;
            zz = z * z;
            xy = x * y;
            yz = y * z;
            xz = x * z;
            sh4 = shs[i * sh_dim3 + 4];
            sh5 = shs[i * sh_dim3 + 5];
            sh6 = shs[i * sh_dim3 + 6];
            sh7 = shs[i * sh_dim3 + 7];
            sh8 = shs[i * sh_dim3 + 8];

            dc_dsh4 = SH_C2_0 * xy;
            dc_dsh5 = SH_C2_1 * yz;
            dc_dsh6 = SH_C2_2 * (2.0f * zz - xx - yy);
            dc_dsh7 = SH_C2_3 * xz;
            dc_dsh8 = SH_C2_4 * (xx - yy);

            color += dc_dsh4 * sh4 + dc_dsh5 * sh5 + dc_dsh6 * sh6 + dc_dsh7 * sh7 + dc_dsh8 * sh8;

            if (sh_dim3 > 9)
            {
                // level3
                sh9 = shs[i * sh_dim3 + 9];
                sh10 = shs[i * sh_dim3 + 10];
                sh11 = shs[i * sh_dim3 + 11];
                sh12 = shs[i * sh_dim3 + 12];
                sh13 = shs[i * sh_dim3 + 13];
                sh14 = shs[i * sh_dim3 + 14];
                sh15 = shs[i * sh_dim3 + 15];

                dc_dsh9 = SH_C3_0 * y * (3.0f * xx - yy);
                dc_dsh10 = SH_C3_1 * xy * z;
                dc_dsh11 = SH_C3_2 * y * (4.0f * zz - xx - yy);
                dc_dsh12 = SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
                dc_dsh13 = SH_C3_4 * x * (4.0f * zz - xx - yy);
                dc_dsh14 = SH_C3_5 * z * (xx - yy);
                dc_dsh15 = SH_C3_6 * x * (xx - 3.0f * yy);

                color += dc_dsh9 * sh9 + dc_dsh10 * sh10 + dc_dsh11 * sh11 + dc_dsh12 * sh12 +
                         dc_dsh13 * sh13 + dc_dsh14 * sh14 + dc_dsh15 * sh15;
            }
        }
    }
    colors[i] = color;

    // calc the jacobians
    if (dcolor_dshs != nullptr && dcolor_dpws != nullptr)
    {
        float3 dc_dr0 = {0, 0, 0};
        float3 dc_dr1 = {0, 0, 0};
        float3 dc_dr2 = {0, 0, 0};
        Matrix<3, 3> dr_dpw = {0, 0, 0, 0, 0, 0, 0, 0, 0};

        dcolor_dshs[sh_dim3 * i + 0] = SH_C0_0;
        if (sh_dim3 > 1)
        {
            const float normd3_inv = normd_inv * normd_inv * normd_inv;
            const float dr_dpw00 = -d0 * d0 * normd3_inv + normd_inv;
            const float dr_dpw11 = -d1 * d1 * normd3_inv + normd_inv;
            const float dr_dpw22 = -d2 * d2 * normd3_inv + normd_inv;
            const float dr_dpw01 = -d0 * d1 * normd3_inv;
            const float dr_dpw02 = -d0 * d2 * normd3_inv;
            const float dr_dpw12 = -d1 * d2 * normd3_inv;
            dr_dpw = {dr_dpw00, dr_dpw01, dr_dpw02, dr_dpw01, dr_dpw11, dr_dpw12, dr_dpw02, dr_dpw12, dr_dpw22};

            dcolor_dshs[sh_dim3 * i + 1] = dc_dsh1;
            dcolor_dshs[sh_dim3 * i + 2] = dc_dsh2;
            dcolor_dshs[sh_dim3 * i + 3] = dc_dsh3;

            dc_dr0 += SH_C1_2 * sh3;
            dc_dr1 += SH_C1_0 * sh1;
            dc_dr2 += SH_C1_1 * sh2;
            if (sh_dim3 > 4)
            {
                dcolor_dshs[sh_dim3 * i + 4] = dc_dsh4;
                dcolor_dshs[sh_dim3 * i + 5] = dc_dsh5;
                dcolor_dshs[sh_dim3 * i + 6] = dc_dsh6;
                dcolor_dshs[sh_dim3 * i + 7] = dc_dsh7;
                dcolor_dshs[sh_dim3 * i + 8] = dc_dsh8;

                dc_dr0 += SH_C2_0 * y * sh4 - SH_C2_2 * 2 * x * sh6 + SH_C2_3 * z * sh7 + SH_C2_4 * 2 * x * sh8;
                dc_dr1 += SH_C2_0 * x * sh4 + SH_C2_1 * z * sh5 - SH_C2_2 * 2.0 * y * sh6 - SH_C2_4 * 2 * y * sh8;
                dc_dr2 += SH_C2_1 * y * sh5 + SH_C2_2 * (4.0 * z) * sh6 + SH_C2_3 * x * sh7;

                if (sh_dim3 > 9)
                {
                    dcolor_dshs[sh_dim3 * i + 9 ] = dc_dsh9;
                    dcolor_dshs[sh_dim3 * i + 10] = dc_dsh10;
                    dcolor_dshs[sh_dim3 * i + 11] = dc_dsh11;
                    dcolor_dshs[sh_dim3 * i + 12] = dc_dsh12;
                    dcolor_dshs[sh_dim3 * i + 13] = dc_dsh13;
                    dcolor_dshs[sh_dim3 * i + 14] = dc_dsh14;
                    dcolor_dshs[sh_dim3 * i + 15] = dc_dsh15;

                    dc_dr0 += 6.0 * SH_C3_0 * sh9 * x * y +
                              SH_C3_1 * sh10 * yz -
                              2 * SH_C3_2 * sh11 * xy -
                              6.0 * SH_C3_3 * sh12 * xz +
                              SH_C3_4 * sh13 * (4.0 * zz - 3.0 * xx - yy) +
                              2 * SH_C3_5 * sh14 * xz +
                              SH_C3_6 * sh15 * (3 * xx - 3 * yy);
                    dc_dr1 += SH_C3_0 * sh9 * (-2 * yy + 3.0 * xx - yy) +
                              SH_C3_1 * sh10 * xz +
                              SH_C3_2 * sh11 * (-xx - yy + 4.0 * zz - 2 * yy) -
                              6.0 * SH_C3_3 * sh12 * yz + SH_C3_4 * sh13 * (-2 * xy) -
                              2 * SH_C3_5 * sh14 * yz -
                              6.0 * SH_C3_6 * sh15 * xy;
                    dc_dr2 += SH_C3_1 * sh10 * xy +
                              8.0 * SH_C3_2 * sh11 * yz +
                              SH_C3_3 * sh12 * (-3.0 * xx - 3.0 * yy + 6.0 * zz) +
                              8.0 * SH_C3_4 * sh13 * xz +
                              SH_C3_5 * sh14 * (xx - yy);
                }
            }
        }
        Matrix<3, 3> dc_dr = {dc_dr0.x, dc_dr1.x, dc_dr2.x,
                              dc_dr0.y, dc_dr1.y, dc_dr2.y,
                              dc_dr0.z, dc_dr1.z, dc_dr2.z};
        Matrix<3, 3> dcolor_dpw = dc_dr * dr_dpw;
        const int offset_dpw = i * 9;
        for (int j = 0; j < 9; j++)
        {
            dcolor_dpws[offset_dpw + j] = dcolor_dpw(j);
        }
    }
}

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
        float *__restrict__ dloss_dcolors)
{
    /*
    More details are discussed in section 5 of the backward.pdf
    */
    const uint2 tile = {blockIdx.x, blockIdx.y};
    const uint2 pix = {tile.x * BLOCK + threadIdx.x,
                       tile.y * BLOCK + threadIdx.y};

    const int tile_idx = tile.y * gridDim.x + tile.x;
    const uint32_t pix_idx = width * pix.y + pix.x;

    const bool inside = pix.x < width && pix.y < height;
    const int2 range = patch_range_per_tile[tile_idx];

    const int gs_num = range.y - range.x;

    // not patch for this tile.
    if (gs_num == 0)
        return;

    bool thread_is_finished = !inside;

    __shared__ float2 shared_pos2d[BLOCK_SIZE];
    __shared__ float3 shared_cinv2d[BLOCK_SIZE];
    __shared__ float shared_alpha[BLOCK_SIZE];
    __shared__ float3 shared_color[BLOCK_SIZE];
    __shared__ int shared_gsid[BLOCK_SIZE];

    float3 gamma_cur2last = {0, 0, 0};

    float3 dloss_dgamma = {0, 0, 0};

    float tau = 0;
    int cont = 0;
    if (inside)
    {
        dloss_dgamma = {dloss_dgammas[0 * height * width + pix_idx],
                        dloss_dgammas[1 * height * width + pix_idx],
                        dloss_dgammas[2 * height * width + pix_idx]};
        tau = final_tau[pix_idx];
        cont = contrib[pix_idx];
    }

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
            fetch2shared(i / BLOCK_SIZE,
                         true,
                         range,
                         gsid_per_patch,
                         us,
                         cinv2ds,
                         alphas,
                         colors,
                         shared_pos2d,
                         shared_cinv2d,
                         shared_alpha,
                         shared_color,
                         shared_gsid);
            __syncthreads();
        }

        // Because we fetch data backwards, we skip the last CONT Gaussian data
        if (i < gs_num - cont)
            continue;

        const float2 u = shared_pos2d[j];
        const float3 cinv2d = shared_cinv2d[j];
        const float alpha = shared_alpha[j];
        const float3 color = shared_color[j];
        const int gs_id = shared_gsid[j];
        const float2 d = u - pix;
        // forward.pdf (eq.F.5.1)
        const float maha_dist = max(0.0f, mahaSqDist(cinv2d, d));
        const float g = exp(-0.5f * maha_dist);
        const float alpha_prime = min(0.99f, alpha * g);

        if (alpha_prime < 0.002f)
            continue;
        // forward.pdf (eq.F.5.2)
        tau = tau / (1.f - alpha_prime);
    
        // backward.pdf (eq.B.5a)
        const float3 dgamma_dalphaprime = tau * (color - gamma_cur2last);
        // backward.pdf (eq.B.5.1a)
        const float dalphaprime_dalpha = g;
        const float dloss_dalphaprime = dot(dloss_dgamma, dgamma_dalphaprime); 
        const float dloss_dalpha = dloss_dalphaprime * dalphaprime_dalpha;
        atomicAdd(&dloss_dalphas[gs_id], dloss_dalpha);

        // backward.pdf (eq.B.5b)
        const float dgamma_dcolor = alpha_prime * tau;
        float3 dloss_dcolor = dloss_dgamma * dgamma_dcolor;
        atomicAdd(&dloss_dcolors[gs_id * 3 + 0], dloss_dcolor.x);
        atomicAdd(&dloss_dcolors[gs_id * 3 + 1], dloss_dcolor.y);
        atomicAdd(&dloss_dcolors[gs_id * 3 + 2], dloss_dcolor.z);
        // backward.pdf (eq.B.5.2b)
        float2 dalphaprime_du = {(-cinv2d.x*d.x - cinv2d.y*d.y) * alpha_prime, 
                                 (-cinv2d.y*d.x - cinv2d.z*d.y) * alpha_prime};
        float2 dloss_du = dloss_dalphaprime * dalphaprime_du;
        atomicAdd(&dloss_dus[gs_id].x, dloss_du.x);
        atomicAdd(&dloss_dus[gs_id].y, dloss_du.y);
        // backward.pdf (eq.B.5.2c)
        float3 dalphaprime_dcinv2d = {-0.5f * alpha_prime * (d.x * d.x),
                                      -1.0f * alpha_prime * (d.x * d.y),
                                      -0.5f * alpha_prime * (d.y * d.y)};
        float3 dloss_dcinv2d = dloss_dalphaprime * dalphaprime_dcinv2d;
        atomicAdd(&dloss_dcinv2ds[gs_id].x, dloss_dcinv2d.x);
        atomicAdd(&dloss_dcinv2ds[gs_id].y, dloss_dcinv2d.y);
        atomicAdd(&dloss_dcinv2ds[gs_id].z, dloss_dcinv2d.z);

        // save gamma_cur2last for next step.
        gamma_cur2last = alpha_prime * color + (1 - alpha_prime) * gamma_cur2last;
    }
}
