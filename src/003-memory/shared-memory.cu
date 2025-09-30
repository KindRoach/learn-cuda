#include "util/util.cuh"

template<typename T, size_t BLOCK_SIZE>
__global__ void test_shared_memory_kernel() {
    __shared__ T slm[BLOCK_SIZE];

    size_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_id = threadIdx.x;

    size_t slm_read_id = local_id;
    size_t slm_write_id = BLOCK_SIZE - local_id - 1;

    slm[slm_read_id] = static_cast<T>(global_id);
    __syncthreads();

    printf("global id = %llu, local id = %llu, slm[%llu]=%d\n",
           global_id, local_id, slm_write_id, static_cast<int>(slm[slm_write_id]));
}

int main() {
    constexpr size_t block_size = 4;
    test_shared_memory_kernel<float, block_size><<<2, block_size>>>();
    cuda_check(cudaDeviceSynchronize());
}
