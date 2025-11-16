#include <numeric>

#include "cpp-bench-utils/utils.hpp"

template <typename T>
void vector_sum_ref(const std::vector<T>& vec, std::vector<T>& out)
{
    out[0] = std::accumulate(vec.begin(), vec.end(), T{0});
}

template <typename T>
void vector_sum_thrust_reduction(thrust::device_vector<T>& vec, thrust::device_vector<T>& out)
{
    out[0] = thrust::reduce(vec.begin(), vec.end(), T{0}, thrust::plus<T>());
}

template <typename T>
__global__ void vector_sum_atomic_kernel(T* vec, T* out)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(out, vec[idx]);
}

template <typename T, size_t BLOCK_SIZE>
void vector_sum_atomic(thrust::device_vector<T>& vec, thrust::device_vector<T>& out)
{
    out[0] = 0;

    size_t size = vec.size();
    cbu::check_divisible(size, BLOCK_SIZE, "Global size must be divisible by BLOCK_SIZE");

    size_t grid_size = size / BLOCK_SIZE;
    thrust::fill(out.begin(), out.end(), T{0});
    vector_sum_atomic_kernel<T><<<grid_size, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(vec.data()),
        thrust::raw_pointer_cast(out.data())
    );
}

template <typename T, size_t BLOCK_SIZE>
__global__ void vector_sum_block_reduce_kernel(T* vec, T* out)
{
    __shared__ T shared_data[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;

    shared_data[tid] = vec[idx];
    __syncthreads();
    for (size_t stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, shared_data[0]);
    }
}

template <typename T, size_t BLOCK_SIZE>
__global__ void vector_sum_block_reduce_cub_kernel(T* vec, T* out)
{
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    T block_sum = BlockReduce(temp_storage).Sum(vec[idx]);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

template <typename T, size_t BLOCK_SIZE, bool USE_CUB>
void vector_sum_block_reduce(thrust::device_vector<T>& vec, thrust::device_vector<T>& out)
{
    out[0] = 0;

    size_t size = vec.size();
    cbu::check_divisible(size, BLOCK_SIZE, "Global size must be divisible by BLOCK_SIZE");

    size_t grid_size = size / BLOCK_SIZE;
    thrust::fill(out.begin(), out.end(), T{0});
    if constexpr (USE_CUB)
    {
        vector_sum_block_reduce_cub_kernel<T, BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(vec.data()),
            thrust::raw_pointer_cast(out.data())
        );
    }
    else
    {
        vector_sum_block_reduce_kernel<T, BLOCK_SIZE><<<grid_size, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(vec.data()),
            thrust::raw_pointer_cast(out.data())
        );
    }
}

template <typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
__global__ void vector_sum_block_reduce_multi_ele_kernel(T* vec, T* out)
{
    __shared__ T shared_data[BLOCK_SIZE];

    size_t idx = blockIdx.x * blockDim.x * THREAD_SIZE + threadIdx.x;

    T thread_sum = 0;
    for (size_t i = 0; i < BLOCK_SIZE * THREAD_SIZE; i += BLOCK_SIZE)
    {
        thread_sum += vec[idx + i];
    }

    size_t tid = threadIdx.x;
    shared_data[tid] = thread_sum;
    __syncthreads();
    for (size_t stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(out, shared_data[0]);
    }
}

template <typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
__global__ void vector_sum_block_reduce_multi_ele_cub_kernel(T* vec, T* out)
{
    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t idx = blockIdx.x * blockDim.x * THREAD_SIZE + threadIdx.x;

    T thread_sum = 0;
    for (size_t i = 0; i < BLOCK_SIZE * THREAD_SIZE; i += BLOCK_SIZE)
    {
        thread_sum += vec[idx + i];
    }

    T block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0)
    {
        atomicAdd(out, block_sum);
    }
}

template <typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE, bool USE_CUB>
void vector_sum_block_reduce_multi_ele(thrust::device_vector<T>& vec, thrust::device_vector<T>& out)
{
    out[0] = 0;

    size_t size = vec.size();
    cbu::check_divisible(size, BLOCK_SIZE * THREAD_SIZE,
                         "Global size must be divisible by BLOCK_SIZE * THREAD_SIZE");

    size_t grid_size = size / (BLOCK_SIZE * THREAD_SIZE);
    thrust::fill(out.begin(), out.end(), T{0});
    if constexpr (USE_CUB)
    {
        vector_sum_block_reduce_multi_ele_cub_kernel<T, BLOCK_SIZE, THREAD_SIZE><<<grid_size, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(vec.data()),
            thrust::raw_pointer_cast(out.data())
        );
    }
    else
    {
        vector_sum_block_reduce_multi_ele_kernel<T, BLOCK_SIZE, THREAD_SIZE><<<grid_size, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(vec.data()),
            thrust::raw_pointer_cast(out.data())
        );
    }
}

int main()
{
    using namespace cbu;
    using dtype = int;
    constexpr uint16_t block_size = 256;
    constexpr uint8_t thread_size = 4;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> vec(size), out_ref(1);
    random_fill(vec);

    std::cout << "vector_sum_ref:\n";
    BenchmarkOptions opt{
        .total_mem_bytes = size * sizeof(dtype),
        .total_flop = size - 1
    };
    benchmark_func_by_time(secs, [&] { vector_sum_ref(vec, out_ref); }, opt);

    thrust::device_vector<dtype> d_vec = vec;
    thrust::device_vector<dtype> d_out(1);

    using func_t = std::function<void(thrust::device_vector<dtype>&, thrust::device_vector<dtype>&)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"vector_sum_thrust_reduction", vector_sum_thrust_reduction<dtype>},
        {"vector_sum_atomic", vector_sum_atomic<dtype, block_size>},
        {"vector_sum_block_reduce", vector_sum_block_reduce<dtype, block_size, false>},
        {"vector_sum_block_reduce_cub", vector_sum_block_reduce<dtype, block_size, true>},
        {
            "vector_sum_block_reduce_multi_ele",
            vector_sum_block_reduce_multi_ele<dtype, block_size, thread_size, false>
        },
        {
            "vector_sum_block_reduce_multi_ele_cub",
            vector_sum_block_reduce_multi_ele<dtype, block_size, thread_size, true>
        }
    };

    for (auto [func_name, func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_out.begin(), d_out.end(), dtype{0});
        benchmark_func_by_time(secs, [&]()
        {
            func(d_vec, d_out);
            cuda_check(cudaDeviceSynchronize());
        }, opt);

        cuda_acc_check(out_ref, d_out);
    }
}
