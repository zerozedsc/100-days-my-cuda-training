#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Size of the matrix
#define N 1024
#define BLOCK_SIZE 16

// For constant memory - reduced to fit within 64KB limit
#define CONST_MEM_SIZE 16384  // 16K floats (64KB / 4 bytes)
__constant__ float d_B[CONST_MEM_SIZE];

// Function to initialize matrices
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// Function to print the result matrix (only a small part for verification)
void printResult(float *matrix, const char* name) {
    printf("%s (showing top-left 4x4):\n", name);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.4f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Global memory kernel
__global__ void matrixAddGlobal(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

// Shared memory kernel
__global__ void matrixAddShared(float *A, float *B, float *C, int n) { 
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        // Load data to shared memory
        s_A[threadIdx.y][threadIdx.x] = A[row * n + col];
        s_B[threadIdx.y][threadIdx.x] = B[row * n + col];
        
        // Ensure all threads have loaded data
        __syncthreads();
        
        // Perform addition
        C[row * n + col] = s_A[threadIdx.y][threadIdx.x] + s_B[threadIdx.y][threadIdx.x];
    }
}

// Modified constant memory kernel
__global__ void matrixAddConstant(float *A, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        int idx = row * n + col;
        // Use modulo to wrap around within the available constant memory
        int constIdx = idx % CONST_MEM_SIZE;
        C[idx] = A[idx] + d_B[constIdx];
    }
}
// Function to test global memory
void testGlobalMemory(float *h_A, float *h_B, float *h_C) {
    float *d_A, *d_B, *d_C;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));
    
    // Copy host memory to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    
    // Start timing
    cudaEventRecord(start);
    
    // Launch kernel
    matrixAddGlobal<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing and result
    printf("Global Memory Matrix Addition: %f ms\n", milliseconds);
    printResult(h_C, "Global Memory Result");
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Function to test shared memory
void testSharedMemory(float *h_A, float *h_B, float *h_C) {
    float *d_A, *d_B, *d_C;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));
    
    // Copy host memory to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    
    // Start timing
    cudaEventRecord(start);
    
    // Launch kernel
    matrixAddShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing and result
    printf("Shared Memory Matrix Addition: %f ms\n", milliseconds);
    printResult(h_C, "Shared Memory Result");
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Modified function to test constant memory
void testConstantMemory(float *h_A, float *h_B, float *h_C) {
    float *d_A, *d_C;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * N * sizeof(float));
    
    // Copy host memory to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Only copy what fits in constant memory (16384 floats)
    cudaMemcpyToSymbol(d_B, h_B, CONST_MEM_SIZE * sizeof(float));
    
    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);
    
    // Start timing
    cudaEventRecord(start);
    
    // Launch kernel
    matrixAddConstant<<<dimGrid, dimBlock>>>(d_A, d_C, N);
    
    // Check for errors after kernel launch (helpful for diagnosis)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing and result
    printf("Constant Memory Matrix Addition: %f ms\n", milliseconds);
    printResult(h_C, "Constant Memory Result");
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Allocate host memory
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C = (float*)malloc(N * N * sizeof(float));
    
    // Initialize matrices
    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);
    
    // Test with different memory types
    printf("Testing matrix addition with different CUDA memory types...\n\n");
    
    testGlobalMemory(h_A, h_B, h_C);
    testSharedMemory(h_A, h_B, h_C);
    testConstantMemory(h_A, h_B, h_C);
    
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}


// Error: ptxas error   : File uses too much global constant data (0x400000 bytes, 0x10000 max)
// Build failed with exit code 255