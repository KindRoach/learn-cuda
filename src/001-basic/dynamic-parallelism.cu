#include <cstdio>

__global__ void child_kernel(unsigned ptid, unsigned pbid) {
    printf("  Hello from child (block %d, thread %d), launched by parent (block %d, thread %d)\n", blockIdx.x,
           threadIdx.x, pbid, ptid);
}

__global__ void parent_kernel() {
    printf("Parent kernel (block %d, thread %d) launching child kernel...\n",
           blockIdx.x, threadIdx.x);
    child_kernel<<<1, 2>>>(threadIdx.x, blockIdx.x);
}

int main() {
    parent_kernel<<<1, 2>>>();
    cudaDeviceSynchronize();
}
