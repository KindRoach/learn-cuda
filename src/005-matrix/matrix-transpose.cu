#include "cpp-bench-utils/utils.hpp"

// In  : [m,n] in row-major
// Out : [n,m] in row-major

template <typename T>
void matrix_transpose_ref(std::vector<T>& in, std::vector<T>& out, size_t m, size_t n)
{
    size_t ld_in = n, ld_out = m;
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            cbu::mat(out.data(), ld_out, j, i) = cbu::mat(in.data(), ld_in, i, j);
        }
    }
}

template <typename T>
__global__ void matrix_transpose_naive_read_continue_kernel(T* in, T* out, size_t m, size_t n)
{
    size_t ldin = n, ldout = m;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    cbu::mat(out, ldout, x, y) = cbu::mat(in, ldin, y, x);
}

template <typename T, size_t BLOCK_SIZE>
void matrix_transpose_naive_read_continue(
    thrust::device_vector<T>& in,
    thrust::device_vector<T>& out,
    size_t m, size_t n)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");
    dim3 grid_range = dim3(n / BLOCK_SIZE, m / BLOCK_SIZE);
    dim3 block_range = dim3(BLOCK_SIZE, BLOCK_SIZE);
    matrix_transpose_naive_read_continue_kernel<T><<<grid_range, block_range>>>(
        thrust::raw_pointer_cast(in.data()),
        thrust::raw_pointer_cast(out.data()),
        m, n
    );
}

template <typename T>
__global__ void matrix_transpose_naive_write_continue_kernel(T* in, T* out, size_t m, size_t n)
{
    size_t ldin = n, ldout = m;
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    cbu::mat(out, ldout, y, x) = cbu::mat(in, ldin, x, y);
}

template <typename T, size_t BLOCK_SIZE>
void matrix_transpose_naive_write_continue(
    thrust::device_vector<T>& in,
    thrust::device_vector<T>& out,
    size_t m, size_t n)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");
    dim3 grid_range = dim3(m / BLOCK_SIZE, n / BLOCK_SIZE);
    dim3 block_range = dim3(BLOCK_SIZE, BLOCK_SIZE);
    matrix_transpose_naive_write_continue_kernel<T><<<grid_range, block_range>>>(
        thrust::raw_pointer_cast(in.data()),
        thrust::raw_pointer_cast(out.data()),
        m, n
    );
}

template <typename T, int BLOCK_SIZE>
__global__ void matrix_transpose_diagonal_mapping_kernel(T* in, T* out, size_t m, size_t n)
{
    __shared__ T tile[BLOCK_SIZE][BLOCK_SIZE + 1]; // avoid bank conflict

    size_t ldin = n, ldout = m;
    size_t x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    size_t y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = cbu::mat(in, ldin, y, x);

    __syncthreads();

    // Diagonal block mapping
    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;
    cbu::mat(out, ldout, y, x) = tile[threadIdx.x][threadIdx.y];
}

template <typename T, size_t BLOCK_SIZE>
void matrix_transpose_diagonal_mapping(
    thrust::device_vector<T>& in,
    thrust::device_vector<T>& out,
    size_t m, size_t n)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");
    dim3 grid_range = dim3(n / BLOCK_SIZE, m / BLOCK_SIZE);
    dim3 block_range = dim3(BLOCK_SIZE, BLOCK_SIZE);
    matrix_transpose_diagonal_mapping_kernel<T, BLOCK_SIZE><<<grid_range, block_range>>>(
        thrust::raw_pointer_cast(in.data()),
        thrust::raw_pointer_cast(out.data()),
        m, n
    );
}

int main()
{
    using namespace cbu;
    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;
    constexpr size_t block_size = 32;

    size_t secs = 10;
    size_t m = 20 * 1024, n = 5 * 1024; // 100M elements

    size_t size = m * n;
    std::vector<dtype> src(size), dst(size);
    random_fill(src);

    std::cout << "matrix_transpose_ref:\n";
    benchmark_func_by_time(secs, [&] { matrix_transpose_ref(src, dst, m, n); });

    d_vec d_src = src;
    d_vec d_dst(size);

    using func_t = std::function<void(d_vec&, d_vec&, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {
            "matrix_transpose_naive_read_continue",
            matrix_transpose_naive_read_continue<dtype, block_size>
        },
        {
            "matrix_transpose_naive_write_continue",
            matrix_transpose_naive_write_continue<dtype, block_size>
        },
        {
            "matrix_transpose_diagonal_mapping",
            matrix_transpose_diagonal_mapping<dtype, block_size>
        },
    };

    for (auto [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        fill(d_dst.begin(), d_dst.end(), 0);
        benchmark_func_by_time(secs, [&]()
        {
            func(d_src, d_dst, m, n);
            cuda_check(cudaDeviceSynchronize());
        });
        cuda_acc_check(dst, d_dst);
    }
}
