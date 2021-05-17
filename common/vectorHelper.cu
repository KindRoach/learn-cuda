#include "vectorHelper.cuh"
#include "utils.cuh"
#include "randomHelper.cuh"

#include <vector>

void addVectorOnCPU(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C) {
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = A[i] + B[i];
    }
}

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
