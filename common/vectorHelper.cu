#include "vectorHelper.cuh"
#include "utils.cuh"

#include <vector>
#include <curand_kernel.h>

bool isFloatSame(float a, float b, float error) {
    return a - b < error && b - a < error;
}

bool isFloatVectorSame(const std::vector<float> &A, const std::vector<float> &B, float error) {
    for (int i = 0; i < A.size(); ++i) {
        if (!isFloatSame(A[i], B[i], error)) {
            return false;
        }
    }
    return true;
}

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
            d_vec[idx] = static_cast<int>((uniform - 0.5) * 1e4);
        }
    }
}

void randomInitVector(std::vector<float> &vec) {
    const int threadNum = 1024;

    curandState *d_random;
    cudaMalloc(&d_random, threadNum * sizeof(curandState));
    init_random<<<1, threadNum>>>(d_random);
    CHECK(cudaGetLastError())

    float *d_vec;
    size_t nBytes = vec.size() * sizeof(float);
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
