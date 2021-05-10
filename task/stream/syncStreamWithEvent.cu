//
// Created by kindr on 2021/5/10.
//

#include "syncStreamWithEvent.cuh"
#include "../../common/utils.cuh"

const int N = 1 << 25;

__global__
void kernel_func() {
    double sum = 0;
    for (int i = 0; i < N; i++) sum += tan(0.1) * tan(0.1);
    printf("sum=%g\n", sum);
}

void syncStreamWithEvent() {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaEvent_t e;
    cudaEventCreate(&e);

    kernel_func<<<1, 1, 0, s1>>>();
    kernel_func<<<1, 1, 0, s1>>>();
    kernel_func<<<1, 1, 0, s2>>>();

    cudaEventRecord(e, s1);
    cudaStreamWaitEvent(s2, e);

    kernel_func<<<1, 1, 0, s2>>>();

    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    cudaEventDestroy(e);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}
