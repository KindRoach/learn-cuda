//
// Created by kindr on 2021/5/3.
//

#include "arrayHelper.cuh"

void addArrayOnCPU(const float *A, const float *B, float *C, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

bool isFloatArraySame(const float *A, const float *B, const size_t N, float error) {
    for (int i = 0; i < N; ++i) {
        if (!isFloatSame(A[i], B[i], error)) {
            return false;
        }
    }
    return true;
}
