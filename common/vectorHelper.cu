#include "vectorHelper.cuh"

#include <vector>

const float error = 1e-4;

bool isFloatSame(float a, float b) {
    return a - b < error && b - a < error;
}

bool isFloatVectorSame(const std::vector<float> &A, const std::vector<float> &B) {
    for (int i = 0; i < A.size(); ++i) {
        if (!isFloatSame(A[i], B[i])) {
            return false;
        }
    }
    return true;
}

__global__ void init_random(curandState *state) {
    auto idx = threadIdx.x;
    curand_init(idx, idx, 0, &state[idx]);
}

__global__ void gpu_random(curandState *states, float *d_vec, const size_t N) {
    for (size_t i = 0; i < N; i += blockDim.x) {
        size_t idx = i + threadIdx.x;
        if (idx < N) {
            d_vec[idx] = curand_uniform(&states[threadIdx.x]);
        }
    }
}

__global__ void gpu_random(curandState *states, int *d_vec, const size_t N) {
    for (size_t i = 0; i < N; i += blockDim.x) {
        size_t idx = i + threadIdx.x;
        if (idx < N) {
            float uniform = curand_uniform(&states[threadIdx.x]);
            d_vec[idx] = static_cast<int>((uniform - 0.5) * 1e4);
        }
    }
}
