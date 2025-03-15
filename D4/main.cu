#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel for coalesced global memory access
__global__ void coalescedGlobalMemoryAccess(float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Coalesced access: threads in a warp access contiguous memory
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel for uncoalesced global memory access
__global__ void uncoalescedGlobalMemoryAccess(float *input, float *output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Uncoalesced access: strided access pattern
        int stride = 32; // Warp size
        int accessIdx = (idx * stride) % n;
        output[idx] = input[accessIdx] * 2.0f;
    }
}

// Kernel for optimized shared memory usage
__global__ void optimizedSharedMemory(float *input, float *output, int n)
{
    extern __shared__ float sharedData[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Coalesced load from global memory into shared memory
    if (idx < n) {
        sharedData[tid] = input[idx];
    }
    __syncthreads();
    
    // Process data in shared memory
    if (idx < n) {
        sharedData[tid] = sharedData[tid] * 2.0f;
    }
    __syncthreads();
    
    // Coalesced write back to global memory
    if (idx < n) {
        output[idx] = sharedData[tid];
    }
}

// Kernel with bank conflicts in shared memory
__global__ void bankConflictSharedMemory(float *input, float *output, int n)
{
    extern __shared__ float sharedData[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Coalesced load from global memory into shared memory
    if (idx < n) {
        sharedData[tid] = input[idx];
    }
    __syncthreads();
    
    // Create bank conflicts by accessing with stride 32 (assuming 32 banks)
    if (idx < n) {
        int bankConflictIdx = (tid * 32) % blockDim.x;
        float val = sharedData[bankConflictIdx];
        sharedData[tid] = val * 2.0f;
    }
    __syncthreads();
    
    // Write back to global memory
    if (idx < n) {
        output[idx] = sharedData[tid];
    }
}

// Utility function to measure kernel execution time
float measureKernelTime(void (*timing_func)(float*, float*, int, dim3, dim3, size_t),
                       float* d_input, float* d_output, int n, 
                       dim3 gridSize, dim3 blockSize, size_t sharedMemSize) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warm-up run
    timing_func(d_input, d_output, n, gridSize, blockSize, sharedMemSize);
    
    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    timing_func(d_input, d_output, n, gridSize, blockSize, sharedMemSize);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

// Wrapper functions for the kernels
void runCoalescedGlobal(float* d_input, float* d_output, int n, 
                       dim3 gridSize, dim3 blockSize, size_t sharedMemSize) {
    coalescedGlobalMemoryAccess<<<gridSize, blockSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runUncoalescedGlobal(float* d_input, float* d_output, int n, 
                         dim3 gridSize, dim3 blockSize, size_t sharedMemSize) {
    uncoalescedGlobalMemoryAccess<<<gridSize, blockSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runOptimizedShared(float* d_input, float* d_output, int n, 
                       dim3 gridSize, dim3 blockSize, size_t sharedMemSize) {
    optimizedSharedMemory<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void runBankConflict(float* d_input, float* d_output, int n, 
                    dim3 gridSize, dim3 blockSize, size_t sharedMemSize) {
    bankConflictSharedMemory<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(void)
{
    // Problem size
    int n = 1 << 22; // 4M elements
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
    
    // Launch configuration
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);
    
    printf("Memory access pattern performance comparison:\n");
    printf("Array size: %d elements (%zu bytes)\n", n, bytes);
    printf("Block size: %d, Grid size: %d\n\n", blockSize, gridSize);
    
    // Run and time each kernel
    float timeCoalesced = measureKernelTime(runCoalescedGlobal, d_input, d_output, n, 
                                          gridSize, blockSize, 0);
    printf("Coalesced global memory access time:      %.3f ms\n", timeCoalesced);
    
    float timeUncoalesced = measureKernelTime(runUncoalescedGlobal, d_input, d_output, n, 
                                            gridSize, blockSize, 0);
    printf("Uncoalesced global memory access time:    %.3f ms\n", timeUncoalesced);
    
    float timeOptShared = measureKernelTime(runOptimizedShared, d_input, d_output, n, 
                                          gridSize, blockSize, sharedMemSize);
    printf("Optimized shared memory access time:      %.3f ms\n", timeOptShared);
    
    float timeBankConflict = measureKernelTime(runBankConflict, d_input, d_output, n, 
                                             gridSize, blockSize, sharedMemSize);
    printf("Bank conflict shared memory access time:  %.3f ms\n", timeBankConflict);
    
    // Performance comparison
    printf("\nPerformance comparison (relative to coalesced):\n");
    printf("Uncoalesced vs Coalesced: %.2fx slower\n", timeUncoalesced / timeCoalesced);
    printf("Optimized Shared vs Coalesced: %.2fx %s\n", 
           fabs(timeOptShared / timeCoalesced),
           timeOptShared < timeCoalesced ? "faster" : "slower");
    printf("Bank Conflict vs Optimized Shared: %.2fx slower\n", 
           timeBankConflict / timeOptShared);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);
    
    return 0;
}