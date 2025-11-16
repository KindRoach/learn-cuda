#include <numeric>
#include <thrust/inner_product.h>

#include "cpp-bench-utils/utils.hpp"

template <typename T>
void vector_dot_ref(const std::vector<T>& a, const std::vector<T>& b, std::vector<T>& out)
{
    out[0] = std::inner_product(a.begin(), a.end(), b.begin(), static_cast<T>(0));
}

template <typename T>
void vector_dot_thrust(thrust::device_vector<T>& a, thrust::device_vector<T>& b, thrust::device_vector<T>& out)
{
    out[0] = thrust::inner_product(a.begin(), a.end(), b.begin(), static_cast<T>(0));
}

template <typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
__global__ void vector_dot_block_reduce_multi_ele_kernel(T* a, T* b, T* out)
{
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t idx = blockIdx.x * blockDim.x * THREAD_SIZE + threadIdx.x;

    T thread_sum = 0;
    for (size_t i = 0; i < BLOCK_SIZE * THREAD_SIZE; i += BLOCK_SIZE)
    {
        thread_sum += a[idx + i] * b[idx + i];
    }

    T block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

template <typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
void vector_dot_block_reduce_multi_ele(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& out)
{
    out[0] = 0;

    size_t size = a.size();
    cbu::check_divisible(size, BLOCK_SIZE * THREAD_SIZE,
                         "Global size must be divisible by BLOCK_SIZE * THREAD_SIZE");

    size_t grid_size = size / (BLOCK_SIZE * THREAD_SIZE);
    thrust::fill(out.begin(), out.end(), T{0});
    vector_dot_block_reduce_multi_ele_kernel<T, BLOCK_SIZE, THREAD_SIZE><<<grid_size, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(out.data())
    );
}

int main()
{
    using namespace cbu;
    using dtype = int;
    using d_vec = thrust::device_vector<dtype>;
    constexpr size_t block_size = 256;
    constexpr size_t thread_size = 4;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> a(size), b(size), out_ref(1);
    random_fill(a);
    random_fill(b);

    std::cout << "vector_dot_ref:\n";
    BenchmarkOptions opt{
        .total_mem_bytes = size * sizeof(dtype) * 2,
        .total_flop = size * 2,
    };
    benchmark_func_by_time(secs, [&] { vector_dot_ref(a, b, out_ref); }, opt);

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec d_out(1);

    using func_t = std::function<void(d_vec&, d_vec&, d_vec&)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"vector_dot_thrust", vector_dot_thrust<dtype>},
        {"vector_dot_block_reduce_multi_ele", vector_dot_block_reduce_multi_ele<dtype, block_size, thread_size>}
    };

    for (auto [func_name, func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_out.begin(), d_out.end(), dtype{0});
        benchmark_func_by_time(secs, [&]()
        {
            func(d_a, d_b, d_out);
            cuda_check(cudaDeviceSynchronize());
        }, opt);

        cuda_acc_check(out_ref, d_out);
    }
}
