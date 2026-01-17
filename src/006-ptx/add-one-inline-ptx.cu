//
// Created by kindr on 1/17/2026.
//

#include "cpp-bench-utils/utils.hpp"

#include <thrust/device_vector.h>

__global__ void add_one_ptx_kernel(int* A, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        int* ptr = A + idx;
        int val;

        // A[ptr] += 1 using PTX inline assembly
        asm volatile(
            "ld.global.u32 %1, [%0];\n\t"
            "add.u32 %1, %1, 1;\n\t"
            "st.global.u32 [%0], %1;\n\t"
            :
            : "l"(ptr), "r"(val)
            : "memory"
        );
    }
}

template <typename T>
void add_one_ref(std::vector<T>& A)
{
    for (T& a : A) a += 1;
}

int main()
{
    using namespace cbu;

    using dtype = int;
    using d_vec = thrust::device_vector<dtype>;

    constexpr uint16_t block_size = 256;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    // Host vector
    std::vector<dtype> h_A(size);
    random_fill(h_A, static_cast<dtype>(0), static_cast<dtype>(1000));

    // Device vector
    d_vec d_A = h_A;
    dtype* d_ptr = thrust::raw_pointer_cast(d_A.data());

    // kernel launcher
    std::function<void()> launch_kernel = [&]
    {
        add_one_ptx_kernel<<< (size + block_size - 1) / block_size, block_size >>>(d_ptr, size);
    };

    // Reference impl
    add_one_ref(h_A);

    // accuracy validate
    launch_kernel();
    cuda_check(cudaGetLastError());
    cuda_acc_check(h_A, d_A);

    // benchmark
    benchmark_func_by_time(secs, [&]()
    {
        launch_kernel();
        cuda_check(cudaDeviceSynchronize());
    }, BenchmarkOptions{
        .total_mem_bytes = size * sizeof(dtype) * 2,
        .total_flop = size,
    });
}
