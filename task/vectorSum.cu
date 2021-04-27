#include <numeric>
#include <iostream>
#include "../common/utils.cuh"
#include "../common/vectorHelper.cuh"

__global__ void sumVectorOnGPU(float *vec) {
    unsigned offset = blockDim.x * blockIdx.x;
    for (unsigned i = 1; i < blockDim.x; i <<= 1) {
        if (threadIdx.x % (i << 1) == 0) {
            unsigned idx = offset + threadIdx.x;
            vec[idx] = vec[idx] + vec[idx + i];
        }
        __syncthreads();
    }
}

void performVectorSum(size_t nElement, size_t nThread) {
    auto vec = std::vector<float>(nElement);
    randomInitVector(vec);

    // padding 0
    size_t nBlock = (nElement + nThread - 1) / nThread;
    nElement = nBlock * nThread;
    for (size_t i = vec.size(); i < nElement; ++i) {
        vec.emplace_back(0.f);
    }

    float cpuResult = 0;
    std::cout << "sumVectorOnCPU:";
    TIME([&]() {
        cpuResult = std::accumulate(vec.begin(), vec.end(), 0.f);
    });

    size_t nBytes = nElement * sizeof(float);

    float *d_vec;
    cudaMalloc(&d_vec, nBytes);
    CHECK(cudaGetLastError())

    float gpuResult = 0;
    auto gpuOutput = std::vector<float>(nElement);

    std::cout << "sumVectorOnGPU:";
    TIME([&]() {
        cudaMemcpy(d_vec, vec.data(), nBytes, cudaMemcpyHostToDevice);
        sumVectorOnGPU<<<nBlock, nThread>>>(d_vec);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError())

        cudaMemcpy(gpuOutput.data(), d_vec, nBytes, cudaMemcpyDeviceToHost);
        CHECK(cudaGetLastError())

        for (size_t i = 0; i < nElement; i += nThread) {
            gpuResult += gpuOutput[i];
        }
    });

    printf("Is same?: %s (%f==%f)\n",
           isFloatSame(cpuResult, gpuResult, 1e-1) ? "true" : "false",
           cpuResult, gpuResult);

    cudaFree(d_vec);
}


