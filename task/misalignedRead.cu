//
// Created by kindr on 2021/5/2.
//

#include "misalignedRead.cuh"
#include "../common/utils.cuh"

__global__
void misalignedAddOne(float *vec, size_t N, const size_t offset) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < N) vec[idx] = vec[idx] + 1.f;
}

void misalignedRead(size_t nElement, size_t nThread, const size_t offset) {
    float *vec;
    size_t nBytes = nElement * sizeof(float);
    cudaMallocManaged(&vec, nBytes, cudaMemAttachGlobal);
    CHECK(cudaGetLastError());
    memset(vec, 0, nBytes);

    size_t nBlock = (nElement + nThread - 1) / nThread;
    misalignedAddOne<<<nBlock, nThread>>>(vec, nElement, offset);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    bool isSame = true;
    for (size_t i = offset; i < nElement; ++i) {
        if (vec[i] != 1.f) {
            isSame = false;
        }
    }

    printf("isSame?: %s", isSame ? "true" : "false");

    cudaFreeHost(vec);

}
