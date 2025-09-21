#include <cstdio>

__global__ void helloFromGPU() {
    printf("Hello from GPU at (Block:%d, Thread:%d)\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    helloFromGPU<<<2, 5>>>();
    cudaDeviceSynchronize();
}
