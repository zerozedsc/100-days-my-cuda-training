#include "share.h"
#include <chrono>

using namespace std;

// Kernel(CUDA) function to perform arithmetic operation
__global__ void arithmetic_operation(int a, int b, char op, int *result) {
    switch(op) {
        case '+':
            *result = a + b;
            break;
        case '-':
            *result = a - b;
            break;
        case '*':
            *result = a * b;
            break;
        case '/':
            *result = b != 0 ? a / b : 0; // Prevent division by zero
            break;
    }
}

void cuda_arithmetic(int a, int b, char op, int *result) {
    int *d_result;
    cudaMalloc(&d_result, sizeof(int));
    
    // Launch kernel with single thread
    arithmetic_operation<<<1, 1>>>(a, b, op, d_result);
    
    // Copy result back to host
    cudaMemcpy(result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_result);
}

// Kernel for matrix arithmetic operations
__global__ void matrix_operation_kernel(int *a, int *b, int *c, int rows, int cols, char op) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        switch(op) {
            case '+':
                c[idx] = a[idx] + b[idx];
                break;
            case '-':
                c[idx] = a[idx] - b[idx];
                break;
            case '*':
                c[idx] = a[idx] * b[idx];
                break;
            case '/':
                c[idx] = b[idx] != 0 ? a[idx] / b[idx] : 0; // Prevent division by zero
                break;
        }
    }
}

void cuda_matrix_operation(int *h_a, int *h_b, int *h_c, int rows, int cols, char op) {
    int size = rows * cols * sizeof(int);
    int *d_a, *d_b, *d_c;
    
    // Allocate memory on device
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    
    // Copy inputs to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, 
                 (rows + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    matrix_operation_kernel<<<gridDim, blockDim>>>(d_a, d_b, d_c, rows, cols, op);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void runMatrixOperationCuda() {
    int rows = 2048;
    int cols = 2048;
    int size = rows * cols;
    char operations[] = {'+', '-', '*', '/'};
    
    // Allocate host memory
    int *h_a = (int*)malloc(size * sizeof(int));
    int *h_b = (int*)malloc(size * sizeof(int));
    int *h_c = (int*)malloc(size * sizeof(int));
    
    // Initialize matrices
    for (int i = 0; i < size; i++) {
        h_a[i] = rand() % 100 + 1;  // Ensure non-zero values
        h_b[i] = rand() % 100 + 1;  // Ensure non-zero values
    }
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (char op : operations) {
        float milliseconds = 0;
        
        // Record start event
        cudaEventRecord(start);
        
        // Perform matrix operation
        cuda_matrix_operation(h_a, h_b, h_c, rows, cols, op);
        
        // Record stop event and synchronize
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // Calculate elapsed time
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("(CUDA) Matrix Operation %c (%dx%d):\n", op, rows, cols);
        printf("Execution time: %.6f ms\n", milliseconds);
        
        // Verify result (just a few elements)
        printf("Verification (first 5 elements):\n");
        for (int i = 0; i < 5; i++) {
            printf("%d %c %d = %d\n", h_a[i], op, h_b[i], h_c[i]);
        }
        printf("\n");
    }
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void runNormalCalcCuda(){
    printf("Running in CUDA\n");

    int a = 10, b = 5, result;
    char op = '+';

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Record the start event
    cudaEventRecord(start);
    
    // Execute the arithmetic operation
    cuda_arithmetic(a, b, op, &result);
    
    // Record the stop event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("(CUDA) %d %c %d = %d\n", a, op, b, result);
    printf("Execution time: %.6f ms\n\n", milliseconds);
    
    // Destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


// CPU function to perform arithmetic operation
void cpu_arithmetic(int a, int b, char op, int *result) {
    switch(op) {
        case '+':
            *result = a + b;
            break;
        case '-':
            *result = a - b;
            break;
        case '*':
            *result = a * b;
            break;
        case '/':
            *result = b != 0 ? a / b : 0; // Prevent division by zero
            break;
    }
}

// CPU function to perform matrix arithmetic operation
void cpu_matrix_operation(int *a, int *b, int *c, int rows, int cols, char op) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            switch(op) {
                case '+':
                    c[idx] = a[idx] + b[idx];
                    break;
                case '-':
                    c[idx] = a[idx] - b[idx];
                    break;
                case '*':
                    c[idx] = a[idx] * b[idx];
                    break;
                case '/':
                    c[idx] = b[idx] != 0 ? a[idx] / b[idx] : 0; // Prevent division by zero
                    break;
            }
        }
    }
}

void runMatrixOperationCpu() {
    int rows = 2048;
    int cols = 2048;
    int size = rows * cols;
    char operations[] = {'+', '-', '*', '/'};
    
    // Allocate host memory
    int *h_a = (int*)malloc(size * sizeof(int));
    int *h_b = (int*)malloc(size * sizeof(int));
    int *h_c = (int*)malloc(size * sizeof(int));
    
    // Initialize matrices
    for (int i = 0; i < size; i++) {
        h_a[i] = rand() % 100 + 1;  // Ensure non-zero values
        h_b[i] = rand() % 100 + 1;  // Ensure non-zero values
    }
    
    for (char op : operations) {
        // Use high-resolution timer from chrono
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform matrix operation
        cpu_matrix_operation(h_a, h_b, h_c, rows, cols, op);
        
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        float milliseconds = duration.count() / 1000.0f;
        
        printf("(CPU) Matrix Operation %c (%dx%d):\n", op, rows, cols);
        printf("Execution time: %.6f ms\n", milliseconds);
        
        // Verify result (just a few elements)
        printf("Verification (first 5 elements):\n");
        for (int i = 0; i < 5; i++) {
            printf("%d %c %d = %d\n", h_a[i], op, h_b[i], h_c[i]);
        }
        printf("\n");
    }
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
}

void runNormalCalcCpu() {
    printf("Running in CPU\n");

    int a = 10, b = 5, result;
    char op = '+';
    
    // Use high-resolution timer from chrono
    auto start = std::chrono::high_resolution_clock::now();
    
    cpu_arithmetic(a, b, op, &result);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    float milliseconds = duration.count() / 1000.0f;
    
    printf("(CPU) %d %c %d = %d\n", a, op, b, result);
    printf("Execution time: %.6f ms\n\n", milliseconds);
}


int main()
{
    querySystemInfo();

    runNormalCalcCuda();
    runNormalCalcCpu();
    runMatrixOperationCuda();
    runMatrixOperationCpu();
    
    return 0;
}