#include "cpp-bench-utils/utils.hpp"

template<typename T>
__global__ void write_kernel(T *device_ptr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    device_ptr[i] = static_cast<T>(i);
}

template<typename T>
__global__ void read_kernel(T *device_ptr) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Index: %llu, Value: %d\n", i, static_cast<int>(device_ptr[i]));
}

void checkPointerType(void *ptr) {
    cudaPointerAttributes attr{};
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);

    if (err != cudaSuccess) {
        std::cout << "Error or not a CUDA-registered pointer: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    switch (attr.type) {
        case cudaMemoryTypeUnregistered:
            std::cout << "Unregistered / Unknown Memory (possibly regular CPU memory)\n";
            break;
        case cudaMemoryTypeHost:
            std::cout << "Host Memory: Pinned or Mapped/Zero-copy (allocated via cudaHostAlloc)\n";
            break;
        case cudaMemoryTypeDevice:
            std::cout << "Device memory (allocated via cudaMalloc)\n";
            break;
        case cudaMemoryTypeManaged:
            std::cout << "Unified Managed Memory (allocated via cudaMallocManaged)\n";
            break;
        default:
            std::cout << "Unknown memory type\n";
            break;
    }
}

template<typename T>
void test_mem(T *ptr, size_t size) {
    std::cout << "test_mem:" << ptr << "\n";
    checkPointerType(ptr);

    write_kernel<<<1, size>>>(ptr);
    read_kernel<<<1, size>>>(ptr);
    cbu::cuda_check(cudaDeviceSynchronize());
    std::cout << "\n";
}

int main() {
    using namespace cbu;
    using dtype = float;
    size_t size = 16;
    size_t nBytes = size * sizeof(dtype);

    dtype *p1, *p2, *p3;

    // normal device memory
    cuda_check(cudaMalloc(&p1, nBytes));

    // mapped memory
    cuda_check(cudaHostAlloc(&p2, nBytes, cudaHostAllocMapped));

    // mannaged memory
    cuda_check(cudaMallocManaged(&p3, nBytes));

    // test
    test_mem<dtype>(p1, size);
    test_mem<dtype>(p2, size);
    test_mem<dtype>(p3, size);

    // free
    cuda_check(cudaFree(p1));
    cuda_check(cudaFreeHost(p2));
    cuda_check(cudaFree(p3));
}
