//
// Created by kindr on 2021/5/3.
//

#include <vector>
#include <cuda_runtime_api.h>
#include <string>
#include "matrixTranspose.cuh"
#include "../../common/arrayHelper.cuh"
#include "../../common/utils.cuh"

__global__ void copyRow(float *out, const float *in, const size_t nx, const size_t ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        unsigned int idx = iy * nx + ix;
        out[idx] = in[idx];
    }
}

__global__ void copyCol(float *out, const float *in, const size_t nx, const size_t ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        unsigned int idx = iy * nx + ix;
        out[idx] = in[idx];
    }
}

__global__ void transformNaiveRow(float *out, const float *in, const size_t nx, const size_t ny) {
    unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx_row = ix + iy * nx;
    unsigned int idx_col = ix * ny + iy;
    if (ix < nx && iy < ny) {
        out[idx_col] = in[idx_row];
    }
}

__global__ void transformNaiveCol(float *out, const float *in, const size_t nx, const size_t ny) {
    unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx_row = ix + iy * nx;
    unsigned int idx_col = ix * ny + iy;
    if (ix < nx && iy < ny) {
        out[idx_row] = in[idx_col];
    }
}

__global__ void transformNaiveRowDiagonal(float *out, const float *in, const size_t nx, const size_t ny) {
    unsigned int block_y = blockIdx.x;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int ix = threadIdx.x + blockDim.x * block_x;
    unsigned int iy = threadIdx.y + blockDim.y * block_y;
    unsigned int idx_row = ix + iy * nx;
    unsigned int idx_col = ix * ny + iy;
    if (ix < nx && iy < ny) {
        out[idx_col] = in[idx_row];
    }
}

__global__ void transformNaiveColDiagonal(float *out, const float *in, const size_t nx, const size_t ny) {
    unsigned int block_y = blockIdx.x;
    unsigned int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    unsigned int ix = threadIdx.x + blockDim.x * block_x;
    unsigned int iy = threadIdx.y + blockDim.y * block_y;
    unsigned int idx_row = ix + iy * nx;
    unsigned int idx_col = ix * ny + iy;
    if (ix < nx && iy < ny) {
        out[idx_row] = in[idx_col];
    }
}

void matrixTranspose(size_t n, size_t nThread) {

    float *InMatrix, *outMatrix;
    size_t nElement = n * n;
    size_t nBytes = nElement * sizeof(float);
    cudaMallocManaged(&InMatrix, nBytes, cudaMemAttachGlobal);
    cudaMallocManaged(&outMatrix, nBytes, cudaMemAttachGlobal);
    randomInitArray(InMatrix, nElement);

    size_t nBlock = (n + nThread - 1) / nThread;

    auto transpose = [&](auto kernel, const std::string &tag) {
        TIME([&]() {
            printf("%s: ", tag.data());
            kernel();
            cudaDeviceSynchronize();
            CHECK(cudaGetLastError())
            memset(outMatrix, 0, nBytes);
        });
    };


    const dim3 &dimGrid = dim3(nBlock, nBlock);
    const dim3 &dimBlock = dim3(nThread, nThread);

    auto f_copyRow = [&]() { copyRow<<<dimGrid, dimBlock>>>(outMatrix, InMatrix, n, n); };
    auto f_copyCol = [&]() { copyCol<<<dimGrid, dimBlock>>>(outMatrix, InMatrix, n, n); };
    transpose(f_copyRow, "copyRow");
    transpose(f_copyCol, "copyCol");

    auto f_transRow = [&]() { transformNaiveRow<<<dimGrid, dimBlock>>>(outMatrix, InMatrix, n, n); };
    auto f_transCol = [&]() { transformNaiveCol<<<dimGrid, dimBlock>>>(outMatrix, InMatrix, n, n); };
    transpose(f_transRow, "transRow");
    transpose(f_transCol, "transCol");

    auto f_transRowDia = [&]() { transformNaiveRowDiagonal<<<dimGrid, dimBlock>>>(outMatrix, InMatrix, n, n); };
    auto f_transColDia = [&]() { transformNaiveColDiagonal<<<dimGrid, dimBlock>>>(outMatrix, InMatrix, n, n); };
    transpose(f_transRowDia, "transRowDia");
    transpose(f_transColDia, "transColDia");
}
