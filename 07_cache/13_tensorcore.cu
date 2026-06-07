#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

__global__ void kernel(int dim_m, int dim_n, int dim_k,
                       half *d_a, half *d_b, float *d_c) {
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int BK = 16;
  constexpr int PAD = 8;
  constexpr int SMEM_M = BM + PAD;
  constexpr int SMEM_N = BN + PAD;

  int block_m = BM * blockIdx.x;
  int block_n = BN * blockIdx.y;

  int tid = threadIdx.x;
  int warp_id = tid / 32;

  int warp_m = warp_id % 4;  // 0..3
  int warp_n = warp_id / 4;  // 0..1

  __shared__ half block_a[BK][SMEM_M];
  __shared__ half block_b[BK][SMEM_N];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];

  #pragma unroll
  for (int r = 0; r < 2; r++) {
    #pragma unroll
    for (int c = 0; c < 4; c++) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  for (int k0 = 0; k0 < dim_k; k0 += BK) {
    for (int idx = tid; idx < BK * BM; idx += blockDim.x) {
      int kk = idx / BM;
      int mm = idx % BM;

      int global_k = k0 + kk;
      int global_m = block_m + mm;

      if (global_m < dim_m && global_k < dim_k) {
        block_a[kk][mm] = d_a[global_k * dim_m + global_m];
      } else {
        block_a[kk][mm] = __float2half(0.0f);
      }
    }

    for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
      int kk = idx / BN;
      int nn = idx % BN;

      int global_k = k0 + kk;
      int global_n = block_n + nn;

      if (global_n < dim_n && global_k < dim_k) {
        block_b[kk][nn] = d_b[global_n * dim_k + global_k];
      } else {
        block_b[kk][nn] = __float2half(0.0f);
      }
    }

    __syncthreads();

    #pragma unroll
    for (int r = 0; r < 2; r++) {
      int row_tile = warp_m * 2 + r;

      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
      wmma::load_matrix_sync(
          a_frag,
          &block_a[0][row_tile * 16],
          SMEM_M
      );

      #pragma unroll
      for (int c = 0; c < 4; c++) {
        int col_tile = warp_n * 4 + c;

        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(
            b_frag,
            &block_b[0][col_tile * 16],
            SMEM_N
        );

        wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int r = 0; r < 2; r++) {
    int row_tile = warp_m * 2 + r;
    int c_m = block_m + row_tile * 16;

    #pragma unroll
    for (int c = 0; c < 4; c++) {
      int col_tile = warp_n * 4 + c;
      int c_n = block_n + col_tile * 16;

      if (c_m < dim_m && c_n < dim_n) {
        wmma::store_matrix_sync(
            &d_c[c_n * dim_m + c_m],
            acc[r][c],
            dim_m,
            wmma::mem_col_major
        );
      }
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;
  float *A, *B, *C, *C2;
  half *A_half, *B_half;
  cudaMallocManaged(&A, m * k * sizeof(float));
  cudaMallocManaged(&B, k * n * sizeof(float));
  cudaMallocManaged(&C, m * n * sizeof(float));
  cudaMallocManaged(&C2, m * n * sizeof(float));
  cudaMallocManaged(&A_half, m * k * sizeof(half));
  cudaMallocManaged(&B_half, k * n * sizeof(half));
  for (int i=0; i<m; i++)
    for (int j=0; j<k; j++)
      A[k*i+j] = drand48();
  for (int i=0; i<k; i++)
    for (int j=0; j<n; j++)
      B[n*i+j] = drand48();

  for (int i = 0; i < m * k; i++)
    A_half[i] = __float2half(A[i]);
  for (int i = 0; i < k * n; i++)
    B_half[i] = __float2half(B[i]);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++)
    for (int j=0; j<m; j++)
      C[m*i+j] = C2[m*i+j] = 0;
  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
		 CUBLAS_OP_N,
		 CUBLAS_OP_N,
		 m,
		 n,
		 k,
		 &alpha,
		 A, CUDA_R_32F, m,
		 B, CUDA_R_32F, k,
		 &beta,
		 C, CUDA_R_32F, m,
		 CUBLAS_COMPUTE_32F_FAST_16F,
		 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }
  auto toc = chrono::steady_clock::now();
  int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;
//   int tile = 64;
//   dim3 block = dim3(tile);
//   dim3 grid = dim3((m+tile-1)/tile, (n+tile-1)/tile);

  int tile_m = 128;
  int tile_n = 128;
  dim3 block = dim3(256);
  dim3 grid = dim3((m + tile_m - 1) / tile_m,
                 (n + tile_n - 1) / tile_n);
  for (int i = 0; i < Nt+2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    kernel<<< grid, block >>>(m,
			      n,
			      k,
			      A_half,
			      B_half,
			      C2);
    cudaDeviceSynchronize();
  }
  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
  printf("CUBLAS: %.2f Gflops, MYKERNEL: %.2f Gflops\n", cublas_flops, cutlass_flops);
  double err = 0;
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      err += fabs(C[m*i+j] - C2[m*i+j]);
    }
  }
  printf("error: %lf\n", err/n/m);
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cudaFree(A_half);
  cudaFree(B_half);
  cublasDestroy(cublas_handle);
}