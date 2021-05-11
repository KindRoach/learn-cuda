//
// Created by kindr on 2021/5/10.
//

#include <iostream>
#include "vectorAddMultiStream.cuh"
#include "../../common/arrayHelper.cuh"
#include "../../common/utils.cuh"

const int times = 300;

__global__
void addVectorOnGPU_1M(const float *A, const float *B, float *C, size_t N) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < times; ++i) if (idx < N) C[idx] = A[idx] + B[idx];
}


void vectorAddMultiStream(size_t nElement, size_t nThread) {
    size_t nBytes = nElement * sizeof(float);

    float *A, *B, *cpuResult, *gpuResult;
    cudaHostAlloc(&A, nBytes, cudaHostAllocDefault);
    cudaHostAlloc(&B, nBytes, cudaHostAllocDefault);
    cudaHostAlloc(&cpuResult, nBytes, cudaHostAllocDefault);
    cudaHostAlloc(&gpuResult, nBytes, cudaHostAllocDefault);


    randomInitArray(A, nElement);
    randomInitArray(B, nElement);

    addArrayOnCPU(A, B, cpuResult, nElement);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, nBytes);
    cudaMalloc(&d_B, nBytes);
    cudaMalloc(&d_C, nBytes);
    CHECK(cudaGetLastError())


    const int nStream = 4;
    size_t nStreamBytes = nStream * sizeof(cudaStream_t);
    auto *stream = static_cast<cudaStream_t *>(malloc(nStreamBytes));
    for (int i = 0; i < nStream; i++) {
        cudaStreamCreate(&stream[i]);
    }
    CHECK(cudaGetLastError());

    size_t nElementPerStream = nElement / nStream;
    size_t nBytesPerStream = nElementPerStream * sizeof(float);
    size_t nBlockPerStream = (nElementPerStream + nThread - 1) / nThread;
    for (int i = 0; i < nStream; i++) {
        size_t offset = i * nElementPerStream;
        cudaMemcpyAsync(&d_A[offset], &A[offset], nBytesPerStream, cudaMemcpyHostToDevice, stream[i]);
        cudaMemcpyAsync(&d_B[offset], &B[offset], nBytesPerStream, cudaMemcpyHostToDevice, stream[i]);
        addVectorOnGPU_1M<<< nBlockPerStream, nThread, 0, stream[i] >>>(
                &d_A[offset], &d_B[offset], &d_C[offset], nElementPerStream);
        cudaMemcpyAsync(&gpuResult[offset], &d_C[offset], nBytesPerStream, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    for (int i = 0; i < nStream; i++) {
        cudaStreamDestroy(stream[i]);
    }

    std::cout << std::boolalpha << "Is same?: "
              << isFloatArraySame(cpuResult, gpuResult, nElement, 1e-8)
              << std::endl;

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(cpuResult);
    cudaFreeHost(gpuResult);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
