//
// Created by kindr on 2021/5/3.
//

#ifndef LEARNCUDA_ARRAYHELPER_CUH
#define LEARNCUDA_ARRAYHELPER_CUH

#include <curand_kernel.h>
#include <iostream>
#include "vectorHelper.cuh"
#include "utils.cuh"
#include "randomHelper.cuh"

void addArrayOnCPU(const float *A, const float *B, float *C, const size_t N);

bool isFloatArraySame(const float *A, const float *B, const size_t N, float error);

template<typename T>
bool isArraySame(const T *A, const T *B, const size_t N) {
    for (int i = 0; i < N; ++i) {
        if (A[i] != B[i]) {
            return false;
        }
    }
    return true;
}

template<typename T>
void printArrayMatrix(const T *A, const size_t M, const size_t N) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            std::cout << A[row * N + col] << "\t";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void randomInitArray(T *vec, const size_t N) {
    const int threadNum = 1024;

    curandState *d_random;
    cudaMalloc(&d_random, threadNum * sizeof(curandState));
    init_random<<<1, threadNum>>>(d_random);
    CHECK(cudaGetLastError())

    T *d_vec;
    size_t nBytes = N * sizeof(T);
    cudaMalloc(&d_vec, nBytes);
    CHECK(cudaGetLastError())

    gpu_random<<<1, threadNum>>>(d_random, d_vec, N);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    cudaMemcpy(vec, d_vec, nBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_vec);
    cudaFree(d_random);
    CHECK(cudaGetLastError())
}

#endif //LEARNCUDA_ARRAYHELPER_CUH
