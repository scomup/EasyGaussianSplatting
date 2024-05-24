/* Copyright:
 * This file is part of gsplatcu.
 * (c) Liu Yang
 * For the full license information, please view the LICENSE file.
 */

#include <cuda.h>
#include <cuda_runtime.h>

// Define a struct template for the matrix

template <int Height, int Width>
struct Matrix
{
  float elements[Width * Height]; // Flattened 1D array to store matrix elements

  // Function to get the width of the matrix
  __host__ __device__ int width() const
  {
    return Width;
  }

  // Function to get the height of the matrix
  __host__ __device__ int height() const
  {
    return Height;
  }

  // Function to access elements.
  __host__ __device__ const float &operator()(int row, int col) const
  {
    return elements[row * Width + col];
  }

  // Function to modify elements.
  __host__ __device__ float &operator()(int row, int col)
  {
    return elements[row * Width + col];
  }

  // Function to access elements.
  __host__ __device__ const float &operator()(int n) const
  {
    return elements[n];
  }

  // Function to modify elements.
  __host__ __device__ float &operator()(int n)
  {
    return elements[n];
  }

  template <int OtherWidth>
  __host__ __device__ Matrix<Height, OtherWidth> operator*(const Matrix<Width, OtherWidth> &other) const
  {
    Matrix<Height, OtherWidth> res;
    
    for (int i = 0; i < Height; ++i)
    {
      for (int j = 0; j < OtherWidth; ++j)
      {
        res(i, j) = 0;
        for (int k = 0; k < Width; ++k)
        {
          res(i, j) += (*this)(i, k) * other(k, j);
        }
      }
    }
    return res;
  }

  __host__ __device__ Matrix<Height, Width> operator+(const Matrix<Height, Width> &other) const
  {
    Matrix<Height, Width> res;
    
    for (int i = 0; i < Height; ++i)
    {
      for (int j = 0; j < Width; ++j)
      {
        res(i, j) = (*this)(i, j) + other(i, j);
      }
    }
    return res;
  }

  // Function to transpose the matrix
  __host__ __device__ Matrix<Width, Height> transpose() const
  {
      Matrix<Width, Height> matT;
      for (int i = 0; i < Height; ++i)
      {
          for (int j = 0; j < Width; ++j)
          {
              matT(j, i) = (*this)(i, j);
          }
      }
      return matT;
  }
};