#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define N (1 << 20)  // 1,048,576 elements
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

// CPU Prefix Sum (Exclusive Scan)
void prefixSumCPU(int* input, int* output, int n) {
    output[0] = 0;
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// Parallel Prefix Sum (Hillis-Steele for Global Memory)
__global__ void prefixSumGlobal(int* d_in, int* d_out, int n) {
    __shared__ int temp[1024];  // Use shared memory buffering
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= n) return;
    temp[threadIdx.x] = d_in[idx];
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = (threadIdx.x >= offset) ? temp[threadIdx.x - offset] : 0;
        __syncthreads();
        temp[threadIdx.x] += val;
        __syncthreads();
    }

    d_out[idx] = temp[threadIdx.x];
}

// Optimized CUDA Prefix Sum (Shared Memory - Hillis-Steele)
__global__ void prefixSumShared(int* d_in, int* d_out, int n) {
    __shared__ int temp[1024];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) temp[threadIdx.x] = d_in[idx];
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = (threadIdx.x >= offset) ? temp[threadIdx.x - offset] : 0;
        __syncthreads();
        temp[threadIdx.x] += val;
        __syncthreads();
    }

    if (idx < n) d_out[idx] = temp[threadIdx.x];
}

// Warp Shuffle Prefix Sum (Optimized for Larger Inputs)
__device__ int warpPrefixSum(int val, int lane) {
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

__global__ void prefixSumWarpShuffle(int* d_in, int* d_out, int n) {
    __shared__ int warpSums[WARP_SIZE];  
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    if (idx >= n) return;

    int val = d_in[idx];
    val = warpPrefixSum(val, lane);

    if (lane == WARP_SIZE - 1) {
        warpSums[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        int warpScan = (lane < (blockDim.x / WARP_SIZE)) ? warpSums[lane] : 0;
        warpScan = warpPrefixSum(warpScan, lane);
        warpSums[lane] = warpScan;
    }
    __syncthreads();

    if (warpId > 0) {
        val += warpSums[warpId - 1];
    }
    d_out[idx] = val;
}

// Main Function
int main() {
    int *h_in, *h_outCPU, *h_outGPU;
    int *d_in, *d_out;
    
    h_in = (int*)malloc(N * sizeof(int));
    h_outCPU = (int*)malloc(N * sizeof(int));
    h_outGPU = (int*)malloc(N * sizeof(int));

    srand(time(NULL));

    for (int i = 0; i < N; i++) {
        h_in[i] = rand() % 100;
    }

    // CPU Prefix Sum
    auto startScanCPU = std::chrono::high_resolution_clock::now();
    prefixSumCPU(h_in, h_outCPU, N);
    auto endScanCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::chrono::milliseconds::period> prefixSumTimeCPU = endScanCPU - startScanCPU;

    // Allocate GPU memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, N * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, N * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));

    // Determine optimal block size
    int minGridSize, blockSize;
    CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, prefixSumGlobal, 0, N));
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Run GPU Kernel (Global Memory)
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    prefixSumGlobal<<<numBlocks, blockSize>>>(d_in, d_out, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_outGPU, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    float timeGlobal;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeGlobal, start, stop));

    // Run GPU Kernel (Shared Memory)
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    prefixSumShared<<<numBlocks, blockSize>>>(d_in, d_out, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_outGPU, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    float timeShared;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeShared, start, stop));

    // Run GPU Kernel (Warp Shuffle)
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    prefixSumWarpShuffle<<<numBlocks, blockSize>>>(d_in, d_out, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaMemcpy(h_outGPU, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    float timeWarp;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&timeWarp, start, stop));

    // Print Results
    printf("\nExecution Times [N=%d]:\n", N);
    printf("Prefix Sum (CPU): %.5f ms\n", prefixSumTimeCPU.count());
    printf("Prefix Sum (Global Memory - Hillis-Steele): %.5f ms\n", timeGlobal);
    printf("Prefix Sum (Shared Memory - Hillis-Steele): %.5f ms\n", timeShared);
    printf("Prefix Sum (Warp Shuffle - Multi-Warp Optimized): %.5f ms\n", timeWarp);

    // Free memory
    free(h_in);
    free(h_outCPU);
    free(h_outGPU);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
