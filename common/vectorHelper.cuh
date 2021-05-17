//
// Created by kindr on 2021/4/26.
//

#ifndef LEARNCUDA_VECTORHELPER_CUH
#define LEARNCUDA_VECTORHELPER_CUH

#include <vector>
#include <curand_kernel.h>

void addVectorOnCPU(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C);

bool isFloatSame(float a, float b, float error);

bool isFloatVectorSame(const std::vector<float> &A, const std::vector<float> &B, float error);

void randomInitVector(std::vector<float> &vec);

#endif //LEARNCUDA_VECTORHELPER_CUH
