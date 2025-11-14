#include "cpp-bench-utils/utils.hpp"

// A : [m,k] in row-major
// B : [k,n] in row-major or col-major
// C = A x B : [m,n] in row-major

template <typename T, cbu::matrix_layout b_layout>
__global__ void matrix_multiply_naive_kernel(T* a, T* b, T* c, size_t m, size_t n, size_t k)
{
    using namespace cbu;
    size_t lda = k, ldb = b_layout == matrix_layout::row_major ? n : k, ldc = n;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0;
    for (size_t i = 0; i < k; i++)
    {
        if constexpr (b_layout == matrix_layout::row_major)
        {
            sum += mat(a, lda, y, i) * mat(b, ldb, i, x);
        }
        else
        {
            sum += mat(a, lda, y, i) * mat(b, ldb, x, i);
        }
    }
    cbu::mat(c, ldc, y, x) = sum;
}

template <typename T, cbu::matrix_layout b_layout, size_t BLOCK_SIZE>
void matrix_multiply_naive(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n, size_t k)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");
    dim3 grid_range = dim3(n / BLOCK_SIZE, m / BLOCK_SIZE);
    dim3 block_range = dim3(BLOCK_SIZE, BLOCK_SIZE);
    matrix_multiply_naive_kernel<T, b_layout><<<grid_range, block_range>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n, k
    );
}

template <typename T, cbu::matrix_layout b_layout, size_t BLOCK_SIZE>
__global__ void matrix_multiply_tile_kernel(T* a, T* b, T* c, size_t m, size_t n, size_t k)
{
    using namespace cbu;
    __shared__ T tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T tile_b[BLOCK_SIZE][BLOCK_SIZE + 1]; // avoid bank conflict for b in col_major.

    size_t lda = k, ldb = b_layout == matrix_layout::row_major ? n : k, ldc = n;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0;
    for (size_t k_i = 0; k_i < k; k_i += BLOCK_SIZE)
    {
        // load A tile
        tile_a[threadIdx.y][threadIdx.x] = mat(a, lda, y, k_i + threadIdx.x);

        // load B tile
        if constexpr (b_layout == matrix_layout::row_major)
        {
            tile_b[threadIdx.y][threadIdx.x] = mat(b, ldb, k_i + threadIdx.y, x);
        }
        else
        {
            // Diagonal block mapping, equivalent to:
            // tile_b[threadIdx.y][threadIdx.x] = mat(b, ldb, x, k_i + threadIdx.y);
            tile_b[threadIdx.x][threadIdx.y] = mat(b, ldb, blockIdx.x * blockDim.x + threadIdx.y, k_i + threadIdx.x);
        }

        __syncthreads();

        for (size_t j = 0; j < BLOCK_SIZE; j++)
        {
            sum += tile_a[threadIdx.y][j] * tile_b[j][threadIdx.x];
        }

        __syncthreads();
    }
    mat(c, ldc, y, x) = sum;
}

template <typename T, cbu::matrix_layout b_layout, size_t BLOCK_SIZE>
void matrix_multiply_tile(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n, size_t k)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");
    dim3 grid_range = dim3(n / BLOCK_SIZE, m / BLOCK_SIZE);
    dim3 block_range = dim3(BLOCK_SIZE, BLOCK_SIZE);
    matrix_multiply_tile_kernel<T, b_layout, BLOCK_SIZE><<<grid_range, block_range>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n, k
    );
}

template <cbu::matrix_layout b_layout>
void test_matrix_multiply()
{
    using namespace cbu;
    std::string b_major = b_layout == matrix_layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix b in " << b_major << " --------------\n";

    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;
    constexpr size_t block_size = 32;

    size_t secs = 10;
    size_t m = 2 * 1024, n = 512, k = 1024;

    std::vector<dtype> a(m * k), b(k * n), c(m * n);
    random_fill(a);
    random_fill(b);

    std::cout << "benchmark_func_by_time:\n";
    benchmark_func_by_time(secs, [&]
    {
        matrix_multiply_ref<dtype, b_layout>(a, b, c, m, n, k);
    });

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec d_c(c.size());

    using func_t = std::function<void(d_vec&, d_vec&, d_vec&, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"matrix_multiply_naive", matrix_multiply_naive<dtype, b_layout, block_size>},
        {"matrix_multiply_tile", matrix_multiply_tile<dtype, b_layout, block_size>},
    };

    for (auto [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_c.begin(), d_c.end(), 0);
        benchmark_func_by_time(secs, [&]()
        {
            func(d_a, d_b, d_c, m, n, k);
            cuda_check(cudaDeviceSynchronize());
        });
        cuda_acc_check(c, d_c);
    }
}

int main()
{
    test_matrix_multiply<cbu::matrix_layout::row_major>();
    test_matrix_multiply<cbu::matrix_layout::col_major>();
}
