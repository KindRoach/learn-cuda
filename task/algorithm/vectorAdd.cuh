//
// Created by kindr on 2021/4/26.
//

#ifndef LEARNCUDA_VECTORADD_CUH
#define LEARNCUDA_VECTORADD_CUH

#include <vector>

__global__ void addVectorOnGPU(const float *A, const float *B, float *C, size_t N);

void performVectorAdd(size_t nElement, size_t nThread);

#endif //LEARNCUDA_VECTORADD_CUH
