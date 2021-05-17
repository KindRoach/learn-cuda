//
// Created by kindr on 2021/5/16.
//

#include "floatPrecision.cuh"
#include "stdio.h"
#include "cuda_fp16.h"

const int n = 1 << 25;

__global__
void halfAdd() {
    half x = 0;
    for (int i = 0; i < n; i++) {
        x += 0.125;
    }
    printf("%f\n", __half2float(x));
}

__global__
void floatAdd() {
    float x = 0;
    for (int i = 0; i < n; i++) {
        x += 0.125;
    }
    printf("%f\n", x);
}

__global__
void doubleAdd() {
    double x = 0;
    for (int i = 0; i < n; i++) {
        x += 0.125;
    }
    printf("%f\n", x);
}

void floatPrecision() {
    halfAdd<<<1, 1>>>();
    floatAdd<<<1, 1>>>();
    doubleAdd<<<1, 1>>>();
    cudaDeviceSynchronize();
}
