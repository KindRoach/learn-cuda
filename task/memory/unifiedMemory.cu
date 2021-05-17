//
// Created by kindr on 2021/4/30.
//

#include "unifiedMemory.cuh"
#include "../../common/utils.cuh"
#include "zeroCopyMemory.cuh"


void unifiedMemory(size_t nElement, size_t nThread) {
    float *vec;
    size_t nBytes = nElement * sizeof(float);
    cudaMallocManaged(&vec, nBytes);
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

    cudaFree(vec);
}
