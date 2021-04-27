#include "vectorAdd.cuh"
#include "../common/utils.cuh"
#include "../common/vectorHelper.cuh"

#include <iostream>

void addVectorOnCPU(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C) {
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = A[i] + B[i];
    }
}

__global__
void addVectorOnGPU(const float *A, const float *B, float *C, size_t N) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

void performVectorAdd(size_t nElement, size_t nThread) {
    auto A = std::vector<float>(nElement);
    auto B = std::vector<float>(nElement);
    auto cpuResult = std::vector<float>(nElement);
    auto gpuResult = std::vector<float>(nElement);

    randomInitVector(A);
    randomInitVector(B);

    std::cout << "addVectorOnCPU:";
    TIME([&]() {
        addVectorOnCPU(A, B, cpuResult);
    });

    size_t nBytes = nElement * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_B, nBytes);
    cudaMalloc(&d_C, nBytes);
    CHECK(cudaGetLastError())

    std::cout << "addVectorOnGPU:";
    TIME([&]() {
        cudaMemcpy(d_A, A.data(), nBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.data(), nBytes, cudaMemcpyHostToDevice);
        CHECK(cudaGetLastError())

        size_t nBlock = (nElement + nThread - 1) / nThread;
        addVectorOnGPU<<<nBlock, nThread>>>(d_A, d_B, d_C, nElement);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError())

        cudaMemcpy(gpuResult.data(), d_C, nBytes, cudaMemcpyDeviceToHost);
        CHECK(cudaGetLastError())
    });

    std::cout << std::boolalpha << "Is same?: "
              << isFloatVectorSame(cpuResult, gpuResult)
              << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
