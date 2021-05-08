//
// Created by kindr on 2021/5/8.
//

#include "multiKernelConcurrent.cuh"
#include "../../common/utils.cuh"
#include "../memory/zeroCopyMemory.cuh"
#include <cstdio>

void multiKernelConcurrent(size_t nElement, size_t nThread) {
    float *vec, *d_vec;
    size_t nBytes = nElement * sizeof(float);
    vec = static_cast<float *>(malloc(nBytes));
    cudaMalloc(&d_vec, nBytes);
    cudaMemset(d_vec, 0, nBytes);
    CHECK(cudaGetLastError());

    int n_stream = 16;
    const size_t &nStreamBytes = n_stream * sizeof(cudaStream_t);
    auto *stream = static_cast<cudaStream_t *>(malloc(nStreamBytes));
    for (int i = 0; i < n_stream; i++) {
        cudaStreamCreate(&stream[i]);
    }
    CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    CHECK(cudaGetLastError());

    size_t nBlock = (nElement + nThread - 1) / nThread;
//    for (int i = 0; i < n_stream; i++) {
//        addOne<<<nBlock, nThread, 0, stream[i]>>>(vec, nElement);
//        addOne<<<nBlock, nThread, 0, stream[i]>>>(vec, nElement);
//        addOne<<<nBlock, nThread, 0, stream[i]>>>(vec, nElement);
//        addOne<<<nBlock, nThread, 0, stream[i]>>>(vec, nElement);
//    }

    for (int i = 0; i < n_stream; i++) {
        addOne<<<nBlock, nThread>>>(d_vec, nElement);
        addOne<<<nBlock, nThread>>>(d_vec, nElement);
        addOne<<<nBlock, nThread>>>(d_vec, nElement);
        addOne<<<nBlock, nThread>>>(d_vec, nElement);
    }
    CHECK(cudaGetLastError());


    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    CHECK(cudaGetLastError());

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("elapsed time:%f ms\n", elapsed_time);

    for (int i = 0; i < n_stream; i++) {
        cudaStreamDestroy(stream[i]);
    }


    cudaMemcpy(vec, d_vec, nBytes, cudaMemcpyDeviceToHost);
    bool isSame = true;
    for (size_t i = 0; i < nElement; ++i) {
        if (vec[i] != 4.f * n_stream) {
            isSame = false;
        }
    }

    printf("isSame?: %s", isSame ? "true" : "false");

    cudaFreeHost(vec);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(stream);
}
