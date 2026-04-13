#include "cpp-bench-utils/utils.hpp"

// Research: K-block scaling in tiled GEMM
//
// GEMM: C = A x B
// A : [m, k] row-major, lda = k
// B : [k, n] row-major, ldb = n
// C : [m, n] row-major, ldc = n

// ------------------------------------------------------------
// Naive kernel: each thread computes one C element, no tiling
// ------------------------------------------------------------
template <typename T>
__global__ void gemm_naive_kernel(const T* a, const T* b, T* c, size_t m, size_t n, size_t k)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    T sum = 0;
    for (size_t i = 0; i < k; i++)
    {
        sum += cbu::mat(a, k, row, i) * cbu::mat(b, n, i, col);
    }
    cbu::mat(c, n, row, col) = sum;
}

template <typename T, size_t BLOCK_SIZE>
void gemm_naive(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n, size_t k)
{
    cbu::check_divisible(m, BLOCK_SIZE, "M must be divisible by BLOCK_SIZE");
    cbu::check_divisible(n, BLOCK_SIZE, "N must be divisible by BLOCK_SIZE");
    dim3 grid(n / BLOCK_SIZE, m / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    gemm_naive_kernel<T><<<grid, block>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n, k
    );
}

// ------------------------------------------------------------
// Blocked kernel: two levels of tiling.
//   Block tile  (BM x BN x BK): the thread block's share of C/A/B,
//                                cached in shared memory.
//   Thread tile (TM x TN):      each thread's register accumulator.
//
// Thread layout: dim3(BN/TN, BM/TM) — fewer threads, more work each.
//   threadIdx.x = col tile index [0, BN/TN)
//   threadIdx.y = row tile index [0, BM/TM)
//
// Shared memory:
//   smem_a [BM][BK]  — current A tile
//   smem_b [BK][BN]  — current B tile
//   (C lives in registers — no smem_c)
//
// Inner loop: outer-product over BK steps.
//   For each step j, load a column fragment of A (TM elements) and
//   a row fragment of B (TN elements) into registers, then compute
//   the TM x TN outer product. Each smem_a element is reused TN
//   times and each smem_b element TM times — that is the arithmetic
//   intensity gain over the 1x1 blocked kernel.
// ------------------------------------------------------------
template <typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_blocked_kernel(const T* a, const T* b, T* c, size_t m, size_t n, size_t k)
{
    static_assert(BM % TM == 0, "BM must be divisible by TM");
    static_assert(BN % TN == 0, "BN must be divisible by TN");

    __shared__ T smem_a[BM][BK];
    __shared__ T smem_b[BK][BN];

    const size_t tx = threadIdx.x;              // col tile index [0, BN/TN)
    const size_t ty = threadIdx.y;              // row tile index [0, BM/TM)
    const size_t tid = ty * (BN / TN) + tx;
    constexpr size_t n_threads = (BM / TM) * (BN / TN);

    const size_t row_base = blockIdx.y * BM;
    const size_t col_base = blockIdx.x * BN;

    // Register accumulator — stays in registers across the K loop.
    T acc[TM][TN] = {};

    for (size_t k_i = 0; k_i < k; k_i += BK)
    {
        // Load A tile [BM x BK] collaboratively.
        for (size_t i = tid; i < BM * BK; i += n_threads)
        {
            size_t row = i / BK;
            size_t col = i % BK;
            smem_a[row][col] = cbu::mat(a, k, row_base + row, k_i + col);
        }

        // Load B tile [BK x BN] collaboratively.
        for (size_t i = tid; i < BK * BN; i += n_threads)
        {
            size_t row = i / BN;
            size_t col = i % BN;
            smem_b[row][col] = cbu::mat(b, n, k_i + row, col_base + col);
        }

        __syncthreads();

        // Outer-product over the BK dimension.
        for (size_t j = 0; j < BK; j++)
        {
            // Load A column fragment (TM elements) into registers.
            T frag_a[TM];
            for (size_t ti = 0; ti < TM; ti++)
                frag_a[ti] = smem_a[ty * TM + ti][j];

            // Load B row fragment (TN elements) into registers.
            T frag_b[TN];
            for (size_t tj = 0; tj < TN; tj++)
                frag_b[tj] = smem_b[j][tx * TN + tj];

            // TM x TN outer product — TM*TN independent FMAs.
            for (size_t ti = 0; ti < TM; ti++)
                for (size_t tj = 0; tj < TN; tj++)
                    acc[ti][tj] += frag_a[ti] * frag_b[tj];
        }

        __syncthreads();
    }

    // Write TM x TN results back to global memory.
    for (size_t ti = 0; ti < TM; ti++)
        for (size_t tj = 0; tj < TN; tj++)
            cbu::mat(c, n, row_base + ty * TM + ti, col_base + tx * TN + tj) = acc[ti][tj];
}

template <typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
void gemm_blocked(
    thrust::device_vector<T>& a,
    thrust::device_vector<T>& b,
    thrust::device_vector<T>& c,
    size_t m, size_t n, size_t k)
{
    cbu::check_divisible(m, BM, "M must be divisible by BM");
    cbu::check_divisible(n, BN, "N must be divisible by BN");
    cbu::check_divisible(k, BK, "K must be divisible by BK");
    dim3 grid(n / BN, m / BM);
    dim3 block(BN / TN, BM / TM);
    gemm_blocked_kernel<T, BM, BN, BK, TM, TN><<<grid, block>>>(
        thrust::raw_pointer_cast(a.data()),
        thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(c.data()),
        m, n, k
    );
}

// ------------------------------------------------------------
// Benchmark harness
// ------------------------------------------------------------
void test_gemm()
{
    using namespace cbu;

    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;
    constexpr size_t block_size = 32;

    size_t bench_secs = 10;
    size_t m = 4096, n = 4096, k = 4096;  // k=4096=2^12 allows BK up to 512

    std::vector<dtype> h_a(m * k), h_b(k * n), h_c_ref(m * n);
    random_fill(h_a);
    random_fill(h_b);

    BenchmarkOptions opt{
        .total_mem_bytes = (m * k + k * n + m * n) * sizeof(dtype),
        .total_flop = 2 * m * n * k,
    };

    d_vec d_a = h_a;
    d_vec d_b = h_b;
    d_vec d_c(m * n);

    // GPU reference result using gemm_naive
    std::cout << "=== gemm_naive (reference) ===\n";
    thrust::fill(d_c.begin(), d_c.end(), static_cast<dtype>(0));
    benchmark_func_by_time(bench_secs, [&]
    {
        gemm_naive<dtype, block_size>(d_a, d_b, d_c, m, n, k);
        cuda_check(cudaDeviceSynchronize());
    }, opt);
    thrust::copy(d_c.begin(), d_c.end(), h_c_ref.begin());

    using func_t = std::function<void(d_vec&, d_vec&, d_vec&, size_t, size_t, size_t)>;
    std::vector<std::tuple<std::string, func_t>> kernels{
        // BM=BN=64: smem = (64+64)*BK*4 bytes, max BK=96 → 64. 256 threads/block.
        {"gemm_blocked<64,64,  1,4,4>",   gemm_blocked<dtype, 64, 64,   1, 4, 4>},
        {"gemm_blocked<64,64,  2,4,4>",   gemm_blocked<dtype, 64, 64,   2, 4, 4>},
        {"gemm_blocked<64,64,  4,4,4>",   gemm_blocked<dtype, 64, 64,   4, 4, 4>},
        {"gemm_blocked<64,64,  8,4,4>",   gemm_blocked<dtype, 64, 64,   8, 4, 4>},
        {"gemm_blocked<64,64, 16,4,4>",   gemm_blocked<dtype, 64, 64,  16, 4, 4>},
        {"gemm_blocked<64,64, 32,4,4>",   gemm_blocked<dtype, 64, 64,  32, 4, 4>},
        {"gemm_blocked<64,64, 64,4,4>",   gemm_blocked<dtype, 64, 64,  64, 4, 4>},
        
        // BM=BN=32: smem = (32+32)*BK*4 bytes, max BK=192 → 128. 64 threads/block.
        {"gemm_blocked<32,32,  1,4,4>",   gemm_blocked<dtype, 32, 32,   1, 4, 4>},
        {"gemm_blocked<32,32,  2,4,4>",   gemm_blocked<dtype, 32, 32,   2, 4, 4>},
        {"gemm_blocked<32,32,  4,4,4>",   gemm_blocked<dtype, 32, 32,   4, 4, 4>},
        {"gemm_blocked<32,32,  8,4,4>",   gemm_blocked<dtype, 32, 32,   8, 4, 4>},
        {"gemm_blocked<32,32, 16,4,4>",   gemm_blocked<dtype, 32, 32,  16, 4, 4>},
        {"gemm_blocked<32,32, 32,4,4>",   gemm_blocked<dtype, 32, 32,  32, 4, 4>},
        {"gemm_blocked<32,32, 64,4,4>",   gemm_blocked<dtype, 32, 32,  64, 4, 4>},
        {"gemm_blocked<32,32,128,4,4>",   gemm_blocked<dtype, 32, 32, 128, 4, 4>},
    };

    for (auto& [name, func] : kernels)
    {
        std::cout << "\n=== " << name << " ===\n";
        thrust::fill(d_c.begin(), d_c.end(), static_cast<dtype>(0));
        benchmark_func_by_time(bench_secs, [&]
        {
            func(d_a, d_b, d_c, m, n, k);
            cuda_check(cudaDeviceSynchronize());
        }, opt);
        cuda_acc_check(h_c_ref, d_c);
    }
}

int main()
{
    test_gemm();
}
