/*
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK 16
#define BLOCK_SIZE (BLOCK * BLOCK)
#define DIV_ROUND_UP(X, Y) ((X) + (Y) - 1) / (Y)

#define DEBUG 1

#define CHECK_CUDA(debug) \
if(debug) { \
cudaError_t err = cudaDeviceSynchronize(); \
if (err != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
} \
}


// Spherical harmonics coefficients
#define SH_C0_0 (0.28209479177387814)
#define SH_C1_0 (-0.4886025119029199)
#define SH_C1_1 (0.4886025119029199)
#define SH_C1_2 (-0.4886025119029199)
#define SH_C2_0 (1.0925484305920792)
#define SH_C2_1 (-1.0925484305920792)
#define SH_C2_2 (0.31539156525252005)
#define SH_C2_3 (-1.0925484305920792)
#define SH_C2_4 (0.5462742152960396)
#define SH_C3_0 (-0.5900435899266435)
#define SH_C3_1 (2.890611442640554)
#define SH_C3_2 (-0.4570457994644658)
#define SH_C3_3 (0.3731763325901154)
#define SH_C3_4 (-0.4570457994644658)
#define SH_C3_5 (1.445305721320277)
#define SH_C3_6 (-0.5900435899266435)

inline __device__ float dot(const float3 &a, const float3 &b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __device__ float3 operator*(const float &b, const float3 &a)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __device__ float3 operator*(const float3 &a, const float &b)
{
  return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ void operator+=(float3 &a, const float3 &b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

inline __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float2 operator-(const float2 &a, const uint2 &b)
{
  return make_float2(a.x - (float)b.x, a.y - (float)b.y);
}

inline __device__ float2 operator*(const float &b, const float2 &a)
{
  return make_float2(a.x * b, a.y * b);
}

inline __device__ float mahaSqDist(const float3 &cinv, const float2 &d)
{
  return cinv.x * d.x * d.x + cinv.z * d.y * d.y + 2 * cinv.y * d.x * d.y;
}


#endif
