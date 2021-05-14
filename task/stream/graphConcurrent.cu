//
// Created by kindr on 2021/5/12.
//

#include "graphConcurrent.cuh"
#include "multiKernelConcurrent.cuh"

const int N = 1 << 25;

void graphConcurrent() {
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    // 开始捕获流操作
    cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);

    math_kernel1<<<1, 1, 0, s1>>>(N);

    cudaEvent_t e1, e2;
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);

    cudaEventRecord(e1, s1);
    cudaStreamWaitEvent(s2, e1);

    math_kernel2<<<1, 1, 0, s1>>>(N);
    math_kernel2<<<1, 1, 0, s2>>>(N);

    cudaEventRecord(e2, s2);
    cudaStreamWaitEvent(s1, e2);

    math_kernel1<<<1, 1, 0, s1>>>(N);

    // 捕获结束
    cudaGraph_t graph;
    cudaStreamEndCapture(s1, &graph);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    for (int i = 0; i < 2; i++) {
        cudaGraphLaunch(graphExec, nullptr);
    }

    cudaDeviceSynchronize();
}
