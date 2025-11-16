#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "cpp-bench-utils/utils.hpp"

// A matrix: [m, n] in row-major or col-major
// b vector: [n]
// o = A x b^T vector : [m]

template <typename T, cbu::matrix_layout a_layout>
void matrix_vector_multiply_ref(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c, size_t m, size_t n)
{
    using namespace cbu;
    size_t ld = a_layout == matrix_layout::row_major ? n : m;
    for (size_t i = 0; i < m; i++)
    {
        T sum = 0;
        for (size_t k = 0; k < n; k++)
        {
            if constexpr (a_layout == matrix_layout::row_major)
            {
                sum += mat(a.data(), ld, i, k) * b[k];
            }
            else
            {
                sum += mat(a.data(), ld, k, i) * b[k];
            }
        }
        c[i] = sum;
    }
}

template <typename T, cbu::matrix_layout a_layout>
__global__ void matrix_vector_multiply_naive_kernel(T* a, T* b, T* c, size_t m, size_t n)
{
    using namespace cbu;
    size_t ld = a_layout == matrix_layout::row_major ? n : m;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;
    for (size_t k = 0; k < n; k++)
    {
        if constexpr (a_layout == matrix_layout::row_major)
        {
            sum += mat(a, ld, i, k) * b[k];
        }
        else
        {
            sum += mat(a, ld, k, i) * b[k];
        }
    }
    c[i] = sum;
}

template <typename T, cbu::matrix_layout b_layout, size_t BLOCK_SIZE>
void matrix_vector_multiply_naive(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    matrix_vector_multiply_naive_kernel<T, b_layout><<<m / BLOCK_SIZE, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n
    );
}

template <typename T, cbu::matrix_layout a_layout>
__global__ void matrix_vector_multiply_row_split_warp_kernel(T* a, T* b, T* c, size_t m, size_t n)
{
    using namespace cbu;
    size_t ld = a_layout == matrix_layout::row_major ? n : m;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0;
    for (size_t k = 0; k < n; k += WARP_SIZE)
    {
        if constexpr (a_layout == matrix_layout::row_major)
        {
            sum += mat(a, ld, i, k + threadIdx.x) * b[k + threadIdx.x];
        }
        else
        {
            sum += mat(a, ld, k + threadIdx.x, i) * b[k + threadIdx.x];
        }
    }

    namespace cg = cooperative_groups;
    cg::thread_block tb = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(tb);
    T warp_sum = cg::reduce(warp, sum, cg::plus<T>());

    if (threadIdx.x == 0)
    {
        c[i] = warp_sum;
    }
}

template <typename T, cbu::matrix_layout b_layout, size_t BLOCK_WARP_NUM>
void matrix_vector_multiply_row_split_warp(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n)
{
    using namespace cbu;
    check_divisible(m, BLOCK_WARP_NUM, "M must be divisible by BLOCK_WARP_NUM");
    check_divisible(n, WARP_SIZE, "N must be divisible by WARP_SIZE");

    dim3 block_range = dim3(WARP_SIZE, BLOCK_WARP_NUM);
    dim3 grid_range = dim3(1, m / BLOCK_WARP_NUM);

    matrix_vector_multiply_row_split_warp_kernel<T, b_layout><<<grid_range, block_range>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n
    );
}

template <typename T, cbu::matrix_layout a_layout, size_t BLOCK_SIZE>
__global__ void matrix_vector_multiply_row_split_tile_kernel(T* a, T* b, T* c, size_t m, size_t n)
{
    using namespace cbu;
    __shared__ T tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // avoid bank conflict for b in col_major.

    size_t ld = a_layout == matrix_layout::row_major ? n : m;
    size_t block_offset_y = blockIdx.y * BLOCK_SIZE;

    T sum = 0;
    for (size_t k = 0; k < n; k += BLOCK_SIZE)
    {
        if constexpr (a_layout == matrix_layout::row_major)
        {
            tile[threadIdx.y][threadIdx.x] = mat(a, ld, block_offset_y + threadIdx.y, k + threadIdx.x);
        }
        else
        {
            // transpose a tile in shared memory, equivalent to:
            // tile_b[threadIdx.y][threadIdx.x] = mat(b, ldb, k + threadIdx.x, block_offset_y + threadIdx.y);
            tile[threadIdx.x][threadIdx.y] = mat(a, ld, k + threadIdx.y, block_offset_y + threadIdx.x);
        }

        __syncthreads();
        sum += tile[threadIdx.y][threadIdx.x] * b[k + threadIdx.x];
        __syncthreads();
    }

    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T block_sum = BlockReduce(temp_storage).Sum(sum);

    namespace cg = cooperative_groups;
    cg::thread_block tb = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(tb);
    T warp_sum = cg::reduce(warp, sum, cg::plus<T>());

    if (threadIdx.x == 0)
    {
        c[block_offset_y + threadIdx.y] = warp_sum;
    }
}

template <typename T, cbu::matrix_layout b_layout, size_t BLOCK_SIZE>
void matrix_vector_multiply_row_split_tile(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");

    dim3 block_range = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_range = dim3(1, m / BLOCK_SIZE);

    matrix_vector_multiply_row_split_tile_kernel<T, b_layout, BLOCK_SIZE><<<grid_range, block_range>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n
    );
}


template <typename T, cbu::matrix_layout a_layout, size_t BLOCK_SIZE>
__global__ void matrix_vector_multiply_row_split_block_kernel(T* a, T* b, T* c, size_t m, size_t n)
{
    using namespace cbu;
    size_t ld = a_layout == matrix_layout::row_major ? n : m;
    size_t i = blockIdx.x;
    size_t ele_per_block = n / (BLOCK_SIZE / WARP_SIZE);
    size_t warp_start = (threadIdx.x / WARP_SIZE) * ele_per_block;
    size_t warp_end = warp_start + ele_per_block;
    size_t lane_id = threadIdx.x % WARP_SIZE;

    T sum = 0;
    for (size_t k = warp_start; k < warp_end; k += WARP_SIZE)
    {
        if constexpr (a_layout == matrix_layout::row_major)
        {
            sum += mat(a, ld, i, k + lane_id) * b[k + lane_id];
        }
        else
        {
            sum += mat(a, ld, k + lane_id, i) * b[k + lane_id];
        }
    }

    typedef cub::BlockReduce<T, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    T block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0)
    {
        c[i] = block_sum;
    }
}

template <typename T, cbu::matrix_layout b_layout, size_t BLOCK_SIZE>
void matrix_vector_multiply_row_split_block(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n)
{
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");
    matrix_vector_multiply_row_split_block_kernel<T, b_layout, BLOCK_SIZE><<<m, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n
    );
}


template <cbu::matrix_layout a_layout>
void test_matrix_multiply()
{
    using namespace cbu;
    std::string a_major = a_layout == matrix_layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix a in " << a_major << " --------------\n";

    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;

    size_t secs = 10;
    size_t m = 512 * 1024, n = 1024; // 1G FLOPs

    std::vector<dtype> a(m * n), b(n), c(m);
    random_fill(a);
    random_fill(b);

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec d_c(c.size());

    std::cout << "matrix_vector_multiply_ref:\n";
    BenchmarkOptions opt{
        .total_mem_bytes = (m * n + n + m) * sizeof(dtype),
        .total_flop = 2 * m * n,
    };
    benchmark_func_by_time(secs, [&]()
    {
        matrix_vector_multiply_ref<dtype, a_layout>(a, b, c, m, n);
    }, opt);

    using func_t = std::function<void(d_vec&, d_vec&, d_vec&, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"matrix_vector_multiply_naive", matrix_vector_multiply_naive<dtype, a_layout, 256>},
        {"matrix_vector_multiply_row_split_warp", matrix_vector_multiply_row_split_warp<dtype, a_layout, 32>},
        {"matrix_vector_multiply_row_split_tile", matrix_vector_multiply_row_split_tile<dtype, a_layout, 32>},
        {"matrix_vector_multiply_row_split_block", matrix_vector_multiply_row_split_block<dtype, a_layout, 256>},
    };

    for (auto [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        thrust::fill(d_c.begin(), d_c.end(), 0);
        benchmark_func_by_time(secs, [&]()
        {
            func(d_a, d_b, d_c, m, n);
            cuda_check(cudaDeviceSynchronize());
        }, opt);
        cuda_acc_check(c, d_c);
    }
}


int main()
{
    test_matrix_multiply<cbu::matrix_layout::row_major>();
    test_matrix_multiply<cbu::matrix_layout::col_major>();
}
