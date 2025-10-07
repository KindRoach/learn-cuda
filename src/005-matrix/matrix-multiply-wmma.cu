#include <mma.h>

#include "util/util.cuh"

// A : [m,k] in row-major
// B : [k,n] in row-major or col-major
// C = A x B : [m,n] in row-major

template <typename dtype, typename acc_type, layout b_layout>
void matrix_multiply_ref(
    std::vector<dtype>& a,
    std::vector<dtype>& b,
    std::vector<acc_type>& c,
    size_t m, size_t n, size_t k)
{
    size_t lda = k, ldb = b_layout == layout::row_major ? n : k, ldc = n;
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            acc_type sum = 0;
            for (size_t p = 0; p < k; p++)
            {
                acc_type a_ele = static_cast<acc_type>(mat(a.data(), lda, i, p));
                acc_type b_ele = static_cast<acc_type>(mat(b.data(), ldb,
                                                           b_layout == layout::row_major ? p : j,
                                                           b_layout == layout::row_major ? j : p));
                sum += a_ele * b_ele;
            }
            mat(c.data(), ldc, i, j) = sum;
        }
    }
}

template <typename dtype, typename acc_type, layout b_layout, size_t WM, size_t WN, size_t WK>
__global__ void matrix_multiply_wmma_kernel(dtype* a, dtype* b, acc_type* c, int m, int n, int k)
{
    size_t lda = k, ldb = b_layout == layout::row_major ? n : k, ldc = n;
    size_t warp_x = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    size_t warp_y = blockIdx.y * blockDim.y + threadIdx.y;

    using b_wmma_layout = std::conditional_t<
        b_layout == layout::row_major,
        nvcuda::wmma::row_major,
        nvcuda::wmma::col_major>;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WM, WN, WK, dtype, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WM, WN, WK, dtype, b_wmma_layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, acc_type> c_frag;

    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    for (int k_i = 0; k_i < k; k_i += WK)
    {
        dtype* tile_a = mat_ptr(a, lda, warp_y * WM, k_i);
        dtype* tile_b =
            b_layout == layout::row_major
                ? mat_ptr(b, ldb, k_i, warp_x * WN)
                : mat_ptr(b, ldb, warp_x * WN, k_i);

        nvcuda::wmma::load_matrix_sync(a_frag, tile_a, lda);
        nvcuda::wmma::load_matrix_sync(b_frag, tile_b, ldb);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    acc_type* tile_c = mat_ptr(c, ldc, warp_y * WM, warp_x * WN);
    nvcuda::wmma::store_matrix_sync(tile_c, c_frag, ldc, nvcuda::wmma::mem_row_major);
}

template <typename dtype, typename acc_type, layout b_layout, size_t BLOCK_WRAP_NUM, size_t WM, size_t WN, size_t WK>
void matrix_multiply_wmma(
    thrust::device_vector<dtype>& a,
    thrust::device_vector<dtype>& b,
    thrust::device_vector<acc_type>& c,
    size_t m, size_t n, size_t k)
{
    check_divisible(m, WM * BLOCK_WRAP_NUM, "M must be divisible by WM * WG_WRAP_NUM");
    check_divisible(n, WN * BLOCK_WRAP_NUM, "N must be divisible by WN * WG_WRAP_NUM");
    check_divisible(k, WK, "K must be divisible by WK");

    dim3 block(BLOCK_WRAP_NUM * WARP_SIZE, BLOCK_WRAP_NUM);
    dim3 grid(n / (BLOCK_WRAP_NUM * WN), m / (BLOCK_WRAP_NUM * WM));

    matrix_multiply_wmma_kernel<dtype, acc_type, b_layout, WM, WN, WK><<<grid, block>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n, k);
}

template <layout b_layout>
void test_matrix_multiply()
{
    std::string b_major = b_layout == layout::row_major ? "row major" : "col major";
    std::cout << "-------------- matrix b in " << b_major << " --------------\n";

    using dtype = half;
    using acc_type = float;
    using d_vec = thrust::device_vector<dtype>;
    using d_vec_acc = thrust::device_vector<acc_type>;

    size_t secs = 10;
    size_t m = 2 * 1024, n = 512, k = 1024;

    std::vector<dtype> a(m * k), b(k * n);
    std::vector<acc_type> c(m * n);
    random_fill(a);
    random_fill(b);

    std::cout << "matrix_transpose_ref:\n";
    benchmark_func_by_time(secs, [&]
    {
        matrix_multiply_ref<dtype, acc_type, b_layout>(a, b, c, m, n, k);
    });

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec_acc d_c(c.size());

    using func_t = std::function<void(d_vec&, d_vec&, d_vec_acc&, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> funcs{
        {"matrix_multiply_wmma", matrix_multiply_wmma<dtype, acc_type, b_layout, 4, 16, 16, 16>}
    };

    for (auto [func_name,func] : funcs)
    {
        std::cout << "\n" << func_name << ":\n";
        thrust::fill(d_c.begin(), d_c.end(), 0);
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
    test_matrix_multiply<layout::row_major>();
    test_matrix_multiply<layout::col_major>();
}
