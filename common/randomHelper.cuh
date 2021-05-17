//
// Created by kindr on 2021/5/17.
//

#ifndef LEARNCUDA_RANDOMHELPER_CUH
#define LEARNCUDA_RANDOMHELPER_CUH

__global__ void init_random(curandState *state);

__global__ void gpu_random(curandState *states, float *d_vec, size_t N);

__global__ void gpu_random(curandState *states, int *d_vec, size_t N);

#include <vector>
#include <curand_kernel.h>

#endif //LEARNCUDA_RANDOMHELPER_CUH
