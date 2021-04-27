#include <numeric>
#include <iostream>
#include "../common/utils.cuh"
#include "../common/vectorHelper.cuh"

__global__ void sumVectorOnGPU_Ver1(float *vec) {
    unsigned offset = blockDim.x * blockIdx.x;
    for (unsigned i = 1; i < blockDim.x; i <<= 1) {
        if (threadIdx.x % (i << 1) == 0) {
            unsigned idx = offset + threadIdx.x;
            vec[idx] = vec[idx] + vec[idx + i];
        }
        __syncthreads();
    }
}

__global__ void sumVectorOnGPU_Ver2(float *vec) {
    unsigned offset = blockDim.x * blockIdx.x;
    for (unsigned i = 1; i < blockDim.x; i <<= 1) {
        if (threadIdx.x < blockDim.x / (i << 1)) {
            unsigned idx = offset + threadIdx.x * (i << 1);
            vec[idx] = vec[idx] + vec[idx + i];
        }
        __syncthreads();
    }
}

__global__ void sumVectorOnGPU_Ver3(float *vec) {
    unsigned offset = blockDim.x * blockIdx.x;
    for (unsigned i = blockDim.x >> 1; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            unsigned idx = offset + threadIdx.x;
            vec[idx] = vec[idx] + vec[idx + i];
        }
        __syncthreads();
    }
}

void performVectorSum(size_t nElement, size_t nThread) {
    auto vec = std::vector<float>(nElement);
    randomInitVector(vec);

    float cpuResult = 0;
    std::cout << "sumVectorOnCPU:";
    TIME([&]() {
        cpuResult = std::accumulate(vec.begin(), vec.end(), 0.f);
    });

    // padding 0
    size_t nBlock = (nElement + nThread - 1) / nThread;
    nElement = nBlock * nThread;
    auto gpuOutput = std::vector<float>(nElement);
    for (size_t i = vec.size(); i < nElement; ++i) {
        vec.emplace_back(0.f);
    }

    size_t nBytes = nElement * sizeof(float);

    float *d_vec;
    cudaMalloc(&d_vec, nBytes);
    CHECK(cudaGetLastError())

    float gpuResult_ver1 = 0;
    std::cout << "sumVectorOnGPU_Ver1:";
    TIME([&]() {
        cudaMemcpy(d_vec, vec.data(), nBytes, cudaMemcpyHostToDevice);
        sumVectorOnGPU_Ver1<<<nBlock, nThread>>>(d_vec);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError())

        cudaMemcpy(gpuOutput.data(), d_vec, nBytes, cudaMemcpyDeviceToHost);
        CHECK(cudaGetLastError())

        gpuResult_ver1 = 0;
        for (size_t i = 0; i < nElement; i += nThread) {
            gpuResult_ver1 += gpuOutput[i];
        }
    });

    float gpuResult_ver2 = 0;
    std::cout << "sumVectorOnGPU_Ver2:";
    TIME([&]() {
        cudaMemcpy(d_vec, vec.data(), nBytes, cudaMemcpyHostToDevice);
        sumVectorOnGPU_Ver2<<<nBlock, nThread>>>(d_vec);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError())

        cudaMemcpy(gpuOutput.data(), d_vec, nBytes, cudaMemcpyDeviceToHost);
        CHECK(cudaGetLastError())

        gpuResult_ver2 = 0;
        for (size_t i = 0; i < nElement; i += nThread) {
            gpuResult_ver2 += gpuOutput[i];
        }

    });

    float gpuResult_ver3 = 0;
    std::cout << "sumVectorOnGPU_Ver2:";
    TIME([&]() {
        cudaMemcpy(d_vec, vec.data(), nBytes, cudaMemcpyHostToDevice);
        sumVectorOnGPU_Ver3<<<nBlock, nThread>>>(d_vec);
        cudaDeviceSynchronize();
        CHECK(cudaGetLastError())

        cudaMemcpy(gpuOutput.data(), d_vec, nBytes, cudaMemcpyDeviceToHost);
        CHECK(cudaGetLastError())

        gpuResult_ver3 = 0;
        for (size_t i = 0; i < nElement; i += nThread) {
            gpuResult_ver3 += gpuOutput[i];
        }

    });

    printf("cpuResult: %f\ngpuResult_ver1: %f\ngpuResult_ver2: %f\ngpuResult_ver3: %f\n",
           cpuResult, gpuResult_ver1, gpuResult_ver2, gpuResult_ver3);

    cudaFree(d_vec);
}


