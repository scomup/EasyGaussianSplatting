#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_


#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK 16
#define BLOCK_SIZE (BLOCK * BLOCK)
#define DIV_ROUND_UP(X, Y) ((X) + (Y) - 1) / (Y)

inline __device__ float3 operator*(const float &b, const float3 &a) 
{
  return make_float3(a.x*b, a.y*b, a.z*b);
}

inline __device__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ float2 operator-(const float2 &a, const uint2 &b)
{
  return make_float2(a.x-(float)b.x, a.y-(float)b.y);
}

inline __device__ void operator+=(float3 &a, const float3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __device__ float mahaSqDist(const float3 &cinv, const float2& d)
{
    return cinv.x * d.x * d.x + cinv.z * d.y * d.y + 2 * cinv.y * d.x * d.y;
}


#endif
