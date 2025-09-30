#include "util/util.cuh"

__global__ void hello_from_gpu() {
    printf("Hello from GPU at (Block:%d, Thread:%d)\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    hello_from_gpu<<<2, 5>>>();
    cuda_check(cudaDeviceSynchronize());
}
