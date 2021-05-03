//
// Created by kindr on 2021/4/29.
//

#include "zeroCopyMemory.cuh"
#include "../../common/utils.cuh"
#include <cstdio>
#include <vector>

__global__
void addOne(float *vec, size_t N) {
    unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) vec[idx] = vec[idx] + 1.f;
}

void zeroCopyMemory(size_t nElement, size_t nThread) {
    float *vec;
    size_t nBytes = nElement * sizeof(float);
    cudaHostAlloc(&vec, nBytes, cudaHostAllocMapped);
    CHECK(cudaGetLastError());
    memset(vec, 0, nBytes);

    size_t nBlock = (nElement + nThread - 1) / nThread;
    addOne<<<nBlock, nThread>>>(vec, nElement);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    bool isSame = true;
    for (size_t i = 0; i < nElement; ++i) {
        if (vec[i] != 1.f) {
            isSame = false;
        }
    }

    printf("isSame?: %s", isSame ? "true" : "false");
    cudaFreeHost(vec);
}
