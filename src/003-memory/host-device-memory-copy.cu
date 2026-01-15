#include "cpp-bench-utils/utils.hpp"

void naive_copy(const void* src, void* dst, size_t bytes, cudaMemcpyKind kind)
{
    cbu::cuda_check(cudaMemcpy(dst, src, bytes, kind));
}


int main()
{
    using namespace cbu;
    using dtype = float;

    size_t secs = 10;
    size_t size = 1000 * 1024 * 1024; // 1G elements
    size_t bytes = size * sizeof(dtype);

    std::vector<dtype> h_vec(size);
    random_fill(h_vec);

    dtype* h_pinned;
    cuda_check(cudaMallocHost(&h_pinned, bytes));
    std::copy(h_vec.begin(), h_vec.end(), h_pinned);

    thrust::device_vector<dtype> d_vec(size);

    std::vector<std::tuple<std::string, const void*, void*, cudaMemcpyKind>> funcs{
        {"normal host -> device", h_vec.data(), raw_pointer_cast(d_vec.data()), cudaMemcpyHostToDevice},
        {"pinned host -> device", h_pinned, raw_pointer_cast(d_vec.data()), cudaMemcpyHostToDevice},
        {"device -> normal host", raw_pointer_cast(d_vec.data()), h_vec.data(), cudaMemcpyDeviceToHost},
        {"device -> pinned host", raw_pointer_cast(d_vec.data()), h_pinned, cudaMemcpyDeviceToHost},
    };

    for (auto [memory_name, src_ptr, dst_ptr, kind] : funcs)
    {
        std::cout << "\n" << memory_name << ":\n";
        benchmark_func_by_time(
            secs,
            [&]()
            {
                naive_copy(src_ptr, dst_ptr, bytes, kind);
            }, {
                .total_mem_bytes = sizeof(dtype) * size
            }
        );
    }

    cuda_check(cudaFreeHost(h_pinned));
}
