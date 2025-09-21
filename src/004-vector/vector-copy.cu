#include "util/bench.hpp"
#include "util/validate.hpp"
#include "util/vector.hpp"
#include "util/cuda-util.cuh"

template<typename T>
__global__ void vector_copy_naive_kernel(T *src, T *dst) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];
}

template<typename T, size_t BLOCK_SIZE>
void vector_copy_naive(
    thrust::device_vector<T> &src,
    thrust::device_vector<T> &out
) {
    size_t size = src.size();
    check_divisible(size, BLOCK_SIZE, "Global size must be divisible by BLOCK_SIZE");
    size_t blocksPerGrid = size / BLOCK_SIZE;
    vector_copy_naive_kernel<T><<<blocksPerGrid, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(src.data()),
        thrust::raw_pointer_cast(out.data())
    );
}

template<typename T, typename vec_t, size_t BLOCK_SIZE>
void vector_copy_naive_vec(
    thrust::device_vector<T> &src,
    thrust::device_vector<T> &out
) {
    size_t size = src.size();
    size_t vec_size = sizeof(vec_t) / sizeof(T);
    check_divisible(size, BLOCK_SIZE * vec_size, "Global size must be divisible by BLOCK_SIZE * vec_size");
    size_t blocksPerGrid = size / (BLOCK_SIZE * vec_size);
    vector_copy_naive_kernel<vec_t><<<blocksPerGrid, BLOCK_SIZE>>>(
        reinterpret_cast<vec_t *>(thrust::raw_pointer_cast(src.data())),
        reinterpret_cast<vec_t *>(thrust::raw_pointer_cast(out.data()))
    );
}

template<typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
__global__ void vector_copy_multi_ele_kernel(T *src, T *dst) {
    size_t offset = blockIdx.x * blockDim.x * THREAD_SIZE + threadIdx.x;
    for (size_t i = 0; i < BLOCK_SIZE * THREAD_SIZE; i += BLOCK_SIZE) {
        dst[offset + i] = src[offset + i];
    }
}

template<typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
void vector_copy_multi_ele(thrust::device_vector<T> &src, thrust::device_vector<T> &out) {
    size_t size = src.size();
    check_divisible(size, BLOCK_SIZE * THREAD_SIZE, "Global size must be divisible by BLOCK_SIZE * THREAD_SIZE");
    size_t blocksPerGrid = size / (BLOCK_SIZE * THREAD_SIZE);
    vector_copy_multi_ele_kernel<T, BLOCK_SIZE, THREAD_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(src.data()),
        thrust::raw_pointer_cast(out.data())
    );
}


int main() {
    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;
    constexpr uint16_t block_size = 256;
    constexpr uint16_t thread_size = 4;

    size_t secs = 10;
    size_t loop = 1000;
    size_t size = 1000 * 1024 * 1024; // 1G elements

    std::vector<dtype> vec(size);
    random_fill(vec);

    d_vec d_src = vec;
    d_vec d_dst(size);

    using func_t = std::function<void(d_vec &, d_vec &)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"vector_copy_naive", vector_copy_naive<dtype, block_size>},
        {"vector_copy_naive_vec", vector_copy_naive_vec<dtype, float4, block_size>},
        {"vector_copy_multi_ele", vector_copy_multi_ele<dtype, block_size, thread_size>}
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        thrust::fill(d_dst.begin(), d_dst.end(), 0);
        benchmark_func_by_time(secs, [&]() {
            func(d_src, d_dst);
        });
        cuda_acc_check(vec, d_dst);
    }
}
