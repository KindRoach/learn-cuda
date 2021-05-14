//
// Created by kindr on 2021/5/10.
//

#include "syncStreamWithEvent.cuh"
#include "../../common/utils.cuh"
#include "multiKernelConcurrent.cuh"

const int N = 1 << 25;

void syncStreamWithEvent() {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaEvent_t e;
    cudaEventCreate(&e);

    math_kernel1<<<1, 1, 0, s1>>>(N);
    math_kernel1<<<1, 1, 0, s1>>>(N);
    math_kernel1<<<1, 1, 0, s2>>>(N);

    cudaEventRecord(e, s1);
    cudaStreamWaitEvent(s2, e);

    math_kernel1<<<1, 1, 0, s2>>>(N);

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    cudaEventDestroy(e);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}
