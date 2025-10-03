#include <cstdio>
#include <iostream>

__device__ int get_laneid(int tid_in_block)
{
    return tid_in_block % warpSize;
}

__device__ int get_warpid(int tid_in_block)
{
    return tid_in_block / warpSize;
}

__global__ void print_mapping_1d()
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int tid_in_block = threadIdx.x;
    int warpId = get_warpid(tid_in_block);
    int laneId = get_laneid(tid_in_block);

    printf("gridDim=%d blockDim=%d; global_id=%d; block_id=%d; thread_id=%d; warpId=%d; laneId=%d\n",
           gridDim.x, blockDim.x,
           global_id, block_id, thread_id,
           warpId, laneId);
}

__global__ void print_mapping_2d()
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    int block_x = blockIdx.x;
    int block_y = blockIdx.y;

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = get_warpid(tid_in_block);
    int laneId = get_laneid(tid_in_block);

    printf("gridDim=(%d,%d) blockDim=(%d,%d); "
           "global_id=(%d,%d); block_id=(%d,%d); thread_id=(%d,%d); warpId=%d; laneId=%d\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y,
           global_x, global_y,
           block_x, block_y,
           thread_x, thread_y,
           warpId, laneId);
}

int main()
{
    // 1D example
    print_mapping_1d<<<2, 64>>>();
    cudaDeviceSynchronize();

    std::cout << "=========================\n";

    // 2D example
    dim3 grid(2, 2);
    dim3 block(8, 8);
    print_mapping_2d<<<grid, block>>>();
    cudaDeviceSynchronize();
}
