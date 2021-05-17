//
// Created by kindr on 2021/5/17.
//

#include "matrixMultiplication.cuh"
#include "../../common/arrayHelper.cuh"

__global__
void matrixMulGPU(const int *A, const int *B, int *C, size_t m, size_t n, size_t k) {
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
//    printf("C[%u][%u]=%d\n", row, col, sum);
}

void matrixMulCPU(const int *A, const int *B, int *C, size_t m, size_t n, size_t k) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < k; col++) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                sum += A[row * n + i] * B[i * k + col];
            }
            C[row * k + col] = sum;
        }
    }
}

void matrixMultiplication(size_t m, size_t n, size_t k, size_t nThread) {

    int *A, *B, *cpuResult, *gpuResult;
    size_t nElement_A = m * n;
    size_t nElement_B = n * k;
    size_t nElement_C = m * k;
    A = static_cast<int *>(malloc(nElement_A * sizeof(int)));
    B = static_cast<int *>(malloc(nElement_B * sizeof(int)));
    cpuResult = static_cast<int *>(malloc(nElement_C * sizeof(int)));
    gpuResult = static_cast<int *>(malloc(nElement_C * sizeof(int)));
    randomInitArray(A, nElement_A);
    randomInitArray(B, nElement_B);

    matrixMulCPU(A, B, cpuResult, m, n, k);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, nElement_A * sizeof(int));
    cudaMalloc(&d_B, nElement_B * sizeof(int));
    cudaMalloc(&d_C, nElement_C * sizeof(int));

    cudaMemcpy(d_A, A, nElement_A * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, nElement_B * sizeof(int), cudaMemcpyHostToDevice);

    size_t mBlock = (m + nThread - 1) / nThread;
    size_t kBlock = (k + nThread - 1) / nThread;
    dim3 grid = dim3(kBlock, mBlock);
    dim3 block = dim3(nThread, nThread);
    matrixMulGPU<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    cudaMemcpy(gpuResult, d_C, nElement_C * sizeof(int), cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

//    printArrayMatrix(A, m, n);
//    printf("-----------------\n");
//    printArrayMatrix(B, n, k);
//    printf("-----------------\n");
//    printArrayMatrix(cpuResult, m, k);
//    printf("-----------------\n");
//    printArrayMatrix(gpuResult, m, k);
//    printf("-----------------\n");

    bool isSame = isArraySame(cpuResult, gpuResult, nElement_C);
    printf("isSame?: %s", isSame ? "true" : "false");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(cpuResult);
    free(gpuResult);
}
