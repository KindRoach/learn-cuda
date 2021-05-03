//
// Created by kindr on 2021/4/28.
//

#include "pinnedMemory.cuh"

#include "../../common/utils.cuh"
#include <cstdio>

bool profileCopies(float *h_a, float *h_b, float *d, unsigned int n) {
    unsigned int bytes = n * sizeof(float);
    CHECK(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        if (h_a[i] != h_b[i]) {
            return false;
        }
    }
    return true;
}

void pinnedMemory(size_t nElements) {
    size_t bytes = nElements * sizeof(float);

    // 可分页内存
    float *h_aPageable, *h_bPageable;
    h_aPageable = (float *) malloc(bytes);
    h_bPageable = (float *) malloc(bytes);
    for (size_t i = 0; i < nElements; ++i) {
        h_aPageable[i] = static_cast<float>(i);
    }

    // 固定内存
    float *h_aPinned, *h_bPinned;
    CHECK(cudaMallocHost((void **) &h_aPinned, bytes));
    CHECK(cudaMallocHost((void **) &h_bPinned, bytes));
    memcpy(h_aPinned, h_aPageable, bytes);
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    float *d_a;
    CHECK(cudaMalloc((void **) &d_a, bytes));

    TIME([&]() {
        bool isSame = profileCopies(h_aPageable, h_bPageable, d_a, nElements);
        printf("Pageable isSame: %s --- ", isSame ? "true" : "false");
    });

    TIME([&]() {
        bool isSame = profileCopies(h_aPinned, h_bPinned, d_a, nElements);
        printf("Pinned isSame: %s --- ", isSame ? "true" : "false");
    });

    // cleanup
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
}
