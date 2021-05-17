//
// Created by kindr on 2021/5/17.
//

#include <vector>
#include "utils.cuh"
#include "vectorHelper.cuh"
#include "randomHelper.cuh"

__global__ void init_random(curandState *state) {
    auto idx = threadIdx.x;
    curand_init(idx, idx, 0, &state[idx]);
}

__global__ void gpu_random(curandState *states, float *d_vec, size_t N) {
    for (size_t i = 0; i < N; i += blockDim.x) {
        size_t idx = i + threadIdx.x;
        if (idx < N) {
            float uniform = curand_uniform(&states[threadIdx.x]);
            d_vec[idx] = (uniform - 0.5f) * 2;
        }
    }
}

__global__ void gpu_random(curandState *states, int *d_vec, const size_t N) {
    for (size_t i = 0; i < N; i += blockDim.x) {
        size_t idx = i + threadIdx.x;
        if (idx < N) {
            float uniform = curand_uniform(&states[threadIdx.x]);
            d_vec[idx] = static_cast<int>((uniform - 0.5) * 10);
        }
    }
}