//
// Created by kindr on 2021/5/3.
//

#include "arrayHelper.cuh"
#include "vectorHelper.cuh"
#include "utils.cuh"

void addArrayOnCPU(const float *A, const float *B, float *C, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

bool isFloatArraySame(const float *A, const float *B, const size_t N, float error) {
    for (int i = 0; i < N; ++i) {
        if (!isFloatSame(A[i], B[i], error)) {
            return false;
        }
    }
    return true;
}

void randomInitArray(float *vec, const size_t N) {
    const int threadNum = 1024;

    curandState *d_random;
    cudaMalloc(&d_random, threadNum * sizeof(curandState));
    init_random<<<1, threadNum>>>(d_random);
    CHECK(cudaGetLastError())

    float *d_vec;
    size_t nBytes = N * sizeof(float);
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
