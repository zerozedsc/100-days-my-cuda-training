#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Size of vectors
#define N (1 << 24)  // 16M elements
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// CUDA error checking
#define CHECK_CUDA_ERROR(call) {                                             \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                    \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}

// 1. Warp Shuffle Reduction (Optimized)
__device__ float warpReduceSum(float sum) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum; // The final sum will be in thread 0
}

__global__ void dotProductWarpShuffle(float *a, float *b, float *c) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    float sum = 0.0f;

    // Compute partial sums
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }

    // Perform warp-level reduction
    sum = warpReduceSum(sum);

    // First thread of each warp adds the sum
    if (lane == 0) {
        atomicAdd(c, sum);
    }
}

// 2. Global Memory Reduction
__global__ void dotProductGlobal(float *a, float *b, float *c, float *temp) {
    __shared__ float sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        temp[blockIdx.x] = sdata[0];
    }
}

// 3. Shared Memory Reduction
__global__ void dotProductShared(float *a, float *b, float *c) {
    __shared__ float sharedSum[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }

    sharedSum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(c, sharedSum[0]);
    }
}

// 4. CPU Dot Product
float dotProductCPU(float *a, float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    float *h_a, *h_b;
    float *d_a, *d_b, *d_c, *d_temp;
    float h_c, cpu_result;
    
    // Allocate host memory
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Determine grid dimensions
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    numBlocks = numBlocks > 1024 ? 1024 : numBlocks;

    printf("Vector size: %d\n", N);
    printf("Block size: %d\n", BLOCK_SIZE);
    printf("Number of blocks: %d\n", numBlocks);
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // 1. Test CPU Reduction
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_result = dotProductCPU(h_a, h_b, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    printf("CPU Result: %f, Time: %f ms\n", cpu_result, cpu_duration.count());

    // 2. Test Warp Shuffle Reduction
    h_c = 0.0f;
    CHECK_CUDA_ERROR(cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    dotProductWarpShuffle<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    float warpTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&warpTime, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Warp Shuffle Result: %f, Time: %f ms\n", h_c, warpTime);

    // 3. Test Global Memory Reduction
    h_c = 0.0f;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_temp, numBlocks * sizeof(float)));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    dotProductGlobal<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c, d_temp);
    
    // Sum the partial results from d_temp on CPU
    float* temp_result = (float*)malloc(numBlocks * sizeof(float));
    CHECK_CUDA_ERROR(cudaMemcpy(temp_result, d_temp, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    h_c = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        h_c += temp_result[i];
    }
    free(temp_result);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float globalTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&globalTime, start, stop));
    printf("Global Memory Result: %f, Time: %f ms\n", h_c, globalTime);

    // 4. Test Shared Memory Reduction
    h_c = 0.0f;
    CHECK_CUDA_ERROR(cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    dotProductShared<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, d_c);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float sharedTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&sharedTime, start, stop));
    CHECK_CUDA_ERROR(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Shared Memory Result: %f, Time: %f ms\n", h_c, sharedTime);

    // Compute and print speed comparisons
    printf("\nSpeed Comparisons:\n");

    printf("CPU vs. Warp Shuffle: %.2fx\n", cpu_duration.count() / warpTime);
    printf("CPU vs. Global Memory: %.2fx\n", cpu_duration.count() / globalTime);
    printf("CPU vs. Shared Memory: %.2fx\n", cpu_duration.count() / sharedTime);

    printf("Warp Shuffle vs. Global Memory: %.2fx\n", warpTime / globalTime);
    printf("Warp Shuffle vs. Shared Memory: %.2fx\n", warpTime / sharedTime);

    printf("Global Memory vs. Shared Memory: %.2fx\n", globalTime / sharedTime);

    // Free memory
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
