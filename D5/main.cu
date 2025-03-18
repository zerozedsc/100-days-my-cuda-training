#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// CPU reduction
void reductionCPU(float *input, float *output, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    *output = sum;
}

// GPU reduction with global memory
__global__ void reductionGlobal(float *input, float *output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // First round reduction
    float sum = 0.0f;
    for (int i = tid; i < size; i += stride) {
        sum += input[i];
    }
    
    // Store per-thread results
    input[tid] = sum;
    __syncthreads();
    
    // Only thread 0 does the final reduction
    if (tid == 0) {
        sum = 0.0f;
        for (int i = 0; i < blockDim.x * gridDim.x && i < size; i++) {
            sum += input[i];
        }
        *output = sum;
    }
}

// GPU reduction with global memory (optimized)
__global__ void reductionGlobalOptimized(float *input, float *output, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    float sum = 0.0f;
    
    // Load elements and reduce in a local variable
    for (int i = idx; i < size; i += stride) {
        sum += input[i];  
    }

    // Store the partial sum in global memory
    __shared__ float sharedSum[256];  // Assuming max 256 threads per block
    int tid = threadIdx.x;
    sharedSum[tid] = sum;
    __syncthreads();

    // Reduce within a block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }

    // Store final result from each block in global memory
    if (tid == 0) {
        output[blockIdx.x] = sharedSum[0];
    }
}

// GPU reduction with shared memory
__global__ void reductionShared(float *input, float *output, int size) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (i < size) ? input[i] : 0.0f;
    __syncthreads();
    
    // Do reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int size = 1 << 24; // 16M elements
    size_t bytes = size * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(sizeof(float));
    float *h_output_cpu = (float*)malloc(sizeof(float));
    float *h_output_global = (float*)malloc(sizeof(float));
    float *h_output_global_opt = (float*)malloc(sizeof(float));
    float *h_output_shared = (float*)malloc(sizeof(float));
    
    // Initialize data
    for (int i = 0; i < size; i++) {
        h_input[i] = 1.0f;
    }
    
    // Allocate device memory
    float *d_input, *d_output, *d_intermediate;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // CPU reduction
    auto cpu_start = std::chrono::high_resolution_clock::now();
    reductionCPU(h_input, h_output_cpu, size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;
    
    // GPU reduction with global memory
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    auto global_start = std::chrono::high_resolution_clock::now();
    reductionGlobal<<<gridSize, blockSize>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    auto global_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> global_ms = global_end - global_start;
    
    cudaMemcpy(h_output_global, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // GPU reduction with global memory (optimized)
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice); // Reset input
    cudaMalloc(&d_intermediate, gridSize * sizeof(float));
    
    auto global_opt_start = std::chrono::high_resolution_clock::now();
    reductionGlobalOptimized<<<gridSize, blockSize>>>(d_input, d_intermediate, size);
    reductionGlobalOptimized<<<1, blockSize>>>(d_intermediate, d_output, gridSize);
    cudaDeviceSynchronize();
    auto global_opt_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> global_opt_ms = global_opt_end - global_opt_start;
    
    cudaMemcpy(h_output_global_opt, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // GPU reduction with shared memory
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice); // Reset input
    
    auto shared_start = std::chrono::high_resolution_clock::now();
    reductionShared<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_intermediate, size);
    
    // Second reduction for the intermediate results
    int blocks_stage2 = 1;
    reductionShared<<<1, gridSize, gridSize * sizeof(float)>>>(d_intermediate, d_output, gridSize);
    cudaDeviceSynchronize();
    auto shared_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> shared_ms = shared_end - shared_start;
    
    cudaMemcpy(h_output_shared, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Array size: %d\n", size);
    printf("CPU Reduction:                 %f (result: %f)\n", cpu_ms.count(), *h_output_cpu);
    printf("GPU Reduction (Global):        %f (result: %f)\n", global_ms.count(), *h_output_global);
    printf("GPU Reduction (Global Opt):    %f (result: %f)\n", global_opt_ms.count(), *h_output_global_opt);
    printf("GPU Reduction (Shared):        %f (result: %f)\n", shared_ms.count(), *h_output_shared);
    printf("Speedup (Global vs CPU):       %fx\n", cpu_ms.count() / global_ms.count());
    printf("Speedup (Global Opt vs CPU):   %fx\n", cpu_ms.count() / global_opt_ms.count());
    printf("Speedup (Shared vs CPU):       %fx\n", cpu_ms.count() / shared_ms.count());
    printf("Speedup (Global Opt vs Global):%fx\n", global_ms.count() / global_opt_ms.count());
    printf("Speedup (Shared vs Global):    %fx\n", global_ms.count() / shared_ms.count());
    
    // Free memory
    free(h_input);
    free(h_output);
    free(h_output_cpu);
    free(h_output_global);
    free(h_output_global_opt);
    free(h_output_shared);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_intermediate);
    
    return 0;
}