#include <cstdio>
#include <cstdlib>

__global__ void init_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= range) return;
  bucket[i] = 0;
}

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void rebuild_key(int *key, int *bucket, int n, int range) {
  int value = blockIdx.x * blockDim.x + threadIdx.x;
  if (value >= range) return;

  int start = 0;
  for (int i = 0; i < value; i++) {
    start += bucket[i];
  }

  for (int i = 0; i < bucket[value]; i++) {
    if (start + i < n) {
      key[start + i] = value;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  int *bucket;

  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int threads = 256;
  init_bucket<<<(range + threads - 1) / threads, threads>>>(bucket, range);
  cudaDeviceSynchronize();

  count_bucket<<<(n + threads - 1) / threads, threads>>>(key, bucket, n);
  cudaDeviceSynchronize();

  rebuild_key<<<(range + threads - 1) / threads, threads>>>(key, bucket, n, range);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
