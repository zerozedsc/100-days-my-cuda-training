#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <float.h>  // For FLT_MAX

#define N (1 << 24)  // 16M elements
#define WARP_SIZE 32
#define NUM_TRIALS 20  // Number of times each kernel is run to get an average

// CUDA error checking
#define CHECK_CUDA_ERROR(call) {                                             \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__,     \
                cudaGetErrorString(err));                                    \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}

// Warp Shuffle Reduction (XOR-based, optimized)
__device__ float warpReduceXor(float sum) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    return sum;  // The final sum will be in lane 0
}

// Hierarchical Reduction with XOR Shuffle (Minimized atomicAdd)
__global__ void dotProductHierarchical(float *a, float *b, float *c) {
    extern __shared__ float sharedSum[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;

    float sum = 0.0f;
    for (int i = gid; i < N; i += gridDim.x * blockDim.x) {
        sum += a[i] * b[i];
    }

    // Warp reduction (no shared memory needed for this step)
    sum = warpReduceXor(sum);

    // Store only the warp sums in shared memory
    if (lane == 0) {
        sharedSum[warpId] = sum;
    }
    __syncthreads();

    // Final reduction using first warp
    if (warpId == 0) {
        sum = (tid < (blockDim.x / WARP_SIZE)) ? sharedSum[lane] : 0.0f;
        sum = warpReduceXor(sum);

        if (tid == 0) {
            atomicAdd(c, sum);  // Only one atomic operation per block
        }
    }
}

// CPU Reduction
float dotProductCPU(float *a, float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    float *h_a, *h_b;
    float *d_a, *d_b, *d_c;
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

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // CPU Reduction
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_result = dotProductCPU(h_a, h_b, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    
    printf("CPU Result: %f, Time: %f ms\n\n", cpu_result, cpu_duration.count());

    // Test different block sizes and select the best one
    int blockSizes[] = {128, 192, 256, 320, 384, 512, 640, 768, 1024};  // More block sizes for better tuning
    float bestTime = FLT_MAX;
    int bestBlockSize = 0;

    for (int i = 0; i < 9; i++) {
        int BLOCK_SIZE = blockSizes[i];
        int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        numBlocks = numBlocks > 1024 ? 1024 : numBlocks;

        int minGridSize = 0, maxBlockSize = 0;
        CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, 
            dotProductHierarchical, 
            (BLOCK_SIZE / WARP_SIZE) * sizeof(float), 
            0));
        
        printf("Testing Block Size: %d\n", BLOCK_SIZE);
        printf("Number of blocks: %d\n", numBlocks);
        printf("Max Block Size: %d\n", maxBlockSize);
        printf("Min Grid Size: %d\n\n", minGridSize);
        
        // Run multiple times and average
        float totalTime = 0.0f;
        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            h_c = 0.0f;
            CHECK_CUDA_ERROR(cudaMemcpy(d_c, &h_c, sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaEventRecord(start));
            dotProductHierarchical<<<numBlocks, BLOCK_SIZE, (BLOCK_SIZE / WARP_SIZE) * sizeof(float)>>>(d_a, d_b, d_c);
            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

            float hierarchicalTime;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&hierarchicalTime, start, stop));
            totalTime += hierarchicalTime;
        }

        float avgTime = totalTime / NUM_TRIALS;
        CHECK_CUDA_ERROR(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));

        printf("Hierarchical Reduction Result: %f, Avg Time: %f ms\n", h_c, avgTime);
        printf("CPU vs. Hierarchical: %.2fx speedup\n\n", cpu_duration.count() / avgTime);

        // Check if this is the best-performing block size
        if (avgTime < bestTime) {
            bestTime = avgTime;
            bestBlockSize = BLOCK_SIZE;
        }
    }

    printf("Optimal Block Size: %d (Execution Time: %f ms)\n", bestBlockSize, bestTime);

    // Free memory
    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
