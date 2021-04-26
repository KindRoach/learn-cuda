#include <iostream>

__global__
void say_hello() {
    printf("Hello from GPU at thread:%d.\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU.\n");
    say_hello<<<1, 32>>>();
    cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}
