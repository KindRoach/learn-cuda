//
// Created by kindr on 2021/5/3.
//

#ifndef LEARNCUDA_ARRAYHELPER_CUH
#define LEARNCUDA_ARRAYHELPER_CUH

void addArrayOnCPU(const float *A, const float *B, float *C, const size_t N);

bool isFloatArraySame(const float *A, const float *B, const size_t N, float error);

void randomInitArray(float *vec, const size_t N);

#endif //LEARNCUDA_ARRAYHELPER_CUH
