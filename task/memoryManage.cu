//
// Created by kindr on 2021/4/28.
//

#include "memoryManage.cuh"

#include <cstdio>

__device__ float devData;

__global__ void checkGlobalVariable() {
    printf("Device: the value of the global variable is %f\n", devData);
    devData += 2.0f;
}

void memoryManage() {

    float value = 3.14f;
    cudaMemcpyToSymbol(devData, &value, sizeof(float));
    printf("Host: copied %f to the global variable\n", value);

    checkGlobalVariable <<<1, 1>>>();
    cudaDeviceSynchronize();

    cudaMemcpyFromSymbol(&value, devData, sizeof(float));
    printf("Host: the value changed by the kernel to %f\n", value);
    cudaDeviceReset();
}
