//
// Created by kindr on 2021/4/28.
//

#include "nestHelloWorld.cuh"

#include "cstdio"

__global__ void childKernel(unsigned ptid, unsigned pbid) {
    printf("Hello from child (%d,%d), parent is (%d,%d)\n",
           blockIdx.x, threadIdx.x, pbid, ptid);
}

__global__ void parentKernel() {
    childKernel<<<2, 2>>>(blockIdx.x, threadIdx.x);
}

void nestSayHello() {

    parentKernel<<<2, 2>>>();

}
