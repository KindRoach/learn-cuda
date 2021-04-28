#include <numeric>
#include <iostream>
#include "../common/utils.cuh"
#include "../common/vectorHelper.cuh"

__global__ void sumVectorOnGPU_Ver1(float *vec, float *res) {
    float *vec_i = vec + blockDim.x * 2 * blockIdx.x;
    for (unsigned i = 1; i <= blockDim.x; i *= 2) {
        if (threadIdx.x % i == 0) {
            unsigned idx = threadIdx.x * 2;
            vec_i[idx] = vec_i[idx] + vec_i[idx + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) res[blockIdx.x] = vec_i[0];
}

__global__ void sumVectorOnGPU_Ver2(float *vec, float *res) {
    float *vec_i = vec + blockDim.x * 2 * blockIdx.x;
    for (unsigned i = 1; i <= blockDim.x; i *= 2) {
        if (threadIdx.x < blockDim.x / i) {
            unsigned idx = threadIdx.x * (i << 1);
            vec_i[idx] = vec_i[idx] + vec_i[idx + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) res[blockIdx.x] = vec_i[0];
}

__global__ void sumVectorOnGPU_Ver3(float *vec, float *res) {
    float *vec_i = vec + blockDim.x * 2 * blockIdx.x;
    for (unsigned i = blockDim.x; i > 0; i /= 2) {
        if (threadIdx.x < i) {
            unsigned idx = threadIdx.x;
            vec_i[idx] = vec_i[idx] + vec_i[idx + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) res[blockIdx.x] = vec_i[0];
}


void performVectorSum(size_t nElement, size_t nThread) {
    auto vec = std::vector<float>(nElement);
    randomInitVector(vec);

    float cpuResult = 0;
    std::cout << "sumVectorOnCPU:";
    TIME([&]() {
        cpuResult = std::accumulate(vec.begin(), vec.end(), 0.f);
        printf(" %f --- ", cpuResult);
    });


    // padding 0 for GPU
    size_t nBlock = (nElement + nThread * 2 - 1) / (nThread * 2);
    nElement = nBlock * nThread * 2;
    for (size_t i = vec.size(); i < nElement; ++i) {
        vec.emplace_back(0.f);
    }

    float *d_vec, *d_result;
    cudaMalloc(&d_vec, nElement * sizeof(float));
    cudaMalloc(&d_result, nBlock * sizeof(float));
    CHECK(cudaGetLastError())

    auto ver1_func = [&]() { sumVectorOnGPU_Ver1<<<nBlock, nThread>>>(d_vec, d_result); };
    auto ver2_func = [&]() { sumVectorOnGPU_Ver2<<<nBlock, nThread>>>(d_vec, d_result); };
    auto ver3_func = [&]() { sumVectorOnGPU_Ver3<<<nBlock, nThread>>>(d_vec, d_result); };
    auto gpuOutput = std::vector<float>(nBlock);

    auto sumVector = [&](auto kernel) {
        TIME([&]() {
            printf("sumVectorOnGPU:");

            cudaMemcpy(d_vec, vec.data(), nElement * sizeof(float), cudaMemcpyHostToDevice);
            kernel();
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError())

            cudaMemcpy(gpuOutput.data(), d_result, nBlock * sizeof(float), cudaMemcpyDeviceToHost);
            CHECK(cudaGetLastError())

            float gpuResult = 0;
            for (size_t i = 0; i < nBlock; ++i) {
                gpuResult += gpuOutput[i];
            }

            printf(" %f --- ", gpuResult);
        });
    };

    sumVector(ver1_func);
    sumVector(ver2_func);
    sumVector(ver3_func);

    cudaFree(d_vec);
    cudaFree(d_result);
}


