#include "util/util.cuh"

template<typename T>
void vector_add_ref(std::vector<T> &a, std::vector<T> &b, std::vector<T> &c) {
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] + b[i];
    }
}

template<typename T>
__global__ void vector_add_naive_kernel(T *a, T *b, T *c) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

template<typename T, size_t BLOCK_SIZE>
void vector_add_naive(
    thrust::device_vector<T> &a,
    thrust::device_vector<T> &b,
    thrust::device_vector<T> &c
) {
    size_t size = a.size();
    check_divisible(size, BLOCK_SIZE, "Global size must be divisible by BLOCK_SIZE");
    size_t blocksPerGrid = size / BLOCK_SIZE;
    vector_add_naive_kernel<T><<<blocksPerGrid, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data())
    );
}

template<typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
__global__ void vector_add_multi_ele_kernel(T *a, T *b, T *c) {
    size_t offset = blockIdx.x * blockDim.x * THREAD_SIZE + threadIdx.x;
    for (size_t i = 0; i < THREAD_SIZE; i++) {
        size_t idx = offset + i * BLOCK_SIZE;
        c[idx] = a[idx] + b[idx];
    }
}

template<typename T, size_t BLOCK_SIZE, size_t THREAD_SIZE>
void vector_add_multi_ele(
    thrust::device_vector<T> &a,
    thrust::device_vector<T> &b,
    thrust::device_vector<T> &c
) {
    size_t size = a.size();
    check_divisible(size, BLOCK_SIZE * THREAD_SIZE, "Global size must be divisible by BLOCK_SIZE * THREAD_SIZE");
    size_t blocksPerGrid = size / (BLOCK_SIZE * THREAD_SIZE);
    vector_add_multi_ele_kernel<T, BLOCK_SIZE, THREAD_SIZE><<<blocksPerGrid, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data())
    );
}

int main() {
    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;
    constexpr uint16_t block_size = 256;
    constexpr uint16_t thread_size = 32;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    // Initialize two input vectors and one output vector
    std::vector<dtype> vec_a(size), vec_b(size), vec_c(size);
    random_fill(vec_a);
    random_fill(vec_b);

    std::cout << "vector_add_ref:\n";
    benchmark_func_by_time(secs, [&]() {
        vector_add_ref(vec_a, vec_b, vec_c);
    });

    d_vec d_a = vec_a;
    d_vec d_b = vec_b;
    d_vec d_c(size);

    using func_t = std::function<void(d_vec &, d_vec &, d_vec &)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"vector_add_naive", vector_add_naive<dtype, block_size>},
        {"vector_add_multi_ele", vector_add_multi_ele<dtype, block_size, thread_size>}
    };

    for (auto [func_name, func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        thrust::fill(d_c.begin(), d_c.end(), 0);
        benchmark_func_by_time(secs, [&]() {
            func(d_a, d_b, d_c);
            cuda_check(cudaDeviceSynchronize());
        });
        cuda_acc_check(vec_c, d_c);
    }
}
