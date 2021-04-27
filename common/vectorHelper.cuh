//
// Created by kindr on 2021/4/26.
//

#ifndef LEARNCUDA_VECTORHELPER_CUH
#define LEARNCUDA_VECTORHELPER_CUH

#include <vector>
#include <curand_kernel.h>
#include "utils.cuh"

bool isFloatSame(float a, float b);

bool isFloatVectorSame(const std::vector<float> &A, const std::vector<float> &B);

__global__ void init_random(curandState *state);

__global__ void gpu_random(curandState *states, float *d_vec, size_t N);

__global__ void gpu_random(curandState *states, int *d_vec, size_t N);

template<typename T>
void randomInitVector(std::vector<T> &vec) {
    const int threadNum = 1024;

    curandState *d_random;
    cudaMalloc(&d_random, threadNum * sizeof(curandState));
    init_random<<<1, threadNum>>>(d_random);
    CHECK(cudaGetLastError())

    T *d_vec;
    size_t nBytes = vec.size() * sizeof(T);
    cudaMalloc(&d_vec, nBytes);
    CHECK(cudaGetLastError())

    gpu_random<<<1, threadNum>>>(d_random, d_vec, vec.size());
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError())

    cudaMemcpy(vec.data(), d_vec, nBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_vec);
    cudaFree(d_random);
    CHECK(cudaGetLastError())
}

#endif //LEARNCUDA_VECTORHELPER_CUH
