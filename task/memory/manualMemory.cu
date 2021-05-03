//
// Created by kindr on 2021/4/28.
//

#include "manualMemory.cuh"
#include "../../common/utils.cuh"
#include "zeroCopyMemory.cuh"

#include <cstdio>

void manualMemory(size_t nElement, size_t nThread) {

    size_t nBytes = nElement * sizeof(float);
    auto *vec = (float *) malloc(nBytes);
    memset(vec, 0, nBytes);

    float *d_vec;
    cudaMalloc(&d_vec, nBytes);
    CHECK(cudaGetLastError());

    cudaMemcpy(d_vec, vec, nBytes, cudaMemcpyHostToDevice);
    CHECK(cudaGetLastError());

    size_t nBlock = (nElement + nThread - 1) / nThread;
    addOne<<<nBlock, nThread>>>(d_vec, nElement);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    cudaMemcpy(vec, d_vec, nBytes, cudaMemcpyDeviceToHost);
    CHECK(cudaGetLastError());

    bool isSame = true;
    for (size_t i = 0; i < nElement; ++i) {
        if (vec[i] != 1.f) {
            isSame = false;
        }
    }

    printf("isSame?: %s", isSame ? "true" : "false");

    cudaFree(d_vec);
    free(vec);
}
