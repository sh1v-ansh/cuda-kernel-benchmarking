#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main () {
    int N = 1 << 24; // We are going to add vectors with a length of 16 million (floats)
    size_t bytes = N * sizeof(float);

    /** 
    ----------- HOW IT WORKS -----------
    
    1) Allocate memory on the CPU and initialize
    2) Allocate memory on the GPU
    3) Copy contents from CPU to GPU
    4) Launch operation
    5) Copy GPU to CPU
    6) Cleanup on the CPU and GPU
    */

    // STEP 1 - Allocating and initializing on the CPU

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // STEP 2 - Allocating on the GPU

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // STEP 3 - Copy CPU -> GPU

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // STEP 4 - Launch the operation

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vectorAdd <<< blocks, threads >>> (d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // STEP 5 - Copy GPU -> CPU

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // STEP 5.5 - Verification and bandwidth check

    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 3.0f) { correct = false; break; }
    }
    printf("Result: %s\n", correct ? "CORRECT" : "WRONG");

    float gb = (3.0f * bytes) / 1e9f;
    printf("Data moved: %.2f GB\n", gb);

    // STEP 6 - Cleanup GPU and CPU mallocs

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}