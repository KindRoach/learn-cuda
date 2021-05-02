//
// Created by kindr on 2021/4/29.
//

#ifndef LEARNCUDA_ZEROCOPYMEMORY_CUH
#define LEARNCUDA_ZEROCOPYMEMORY_CUH

__global__ void addOne(float *vec, size_t N);

void zeroCopyMemory(size_t nElement, size_t nThread);

#endif //LEARNCUDA_ZEROCOPYMEMORY_CUH
