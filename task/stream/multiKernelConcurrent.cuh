//
// Created by kindr on 2021/5/8.
//

#ifndef LEARNCUDA_MULTIKERNELCONCURRENT_CUH
#define LEARNCUDA_MULTIKERNELCONCURRENT_CUH

__global__ void math_kernel1(int n);

__global__ void math_kernel2(int n);

void multiKernelConcurrent();

#endif //LEARNCUDA_MULTIKERNELCONCURRENT_CUH
