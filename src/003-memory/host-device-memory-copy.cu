#include "cpp-bench-utils/utils.hpp"

void naive_copy(void *src, void *dst, size_t bytes) {
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

int main() {
    using namespace cbu;
    using dtype = float;

    size_t secs = 10;
    size_t size = 1000 * 1024 * 1024; // 1G elements
    size_t bytes = size * sizeof(dtype);

    std::vector<dtype> h_vec(size);
    random_fill(h_vec);

    dtype *h_pinned;
    cuda_check(cudaMallocHost(&h_pinned, bytes));
    std::copy(h_vec.begin(), h_vec.end(), h_pinned);

    thrust::device_vector<dtype> d_vec(size);

    using func_t = std::function<void()>;
    std::vector<std::tuple<std::string, dtype *> > funcs{
        {"normal host memory", h_vec.data()},
        {"pinned host memory", h_pinned}
    };

    for (auto [memory_name,src_ptr]: funcs) {
        std::cout << "\n" << memory_name << ":\n";
        fill(d_vec.begin(), d_vec.end(), 0);
        benchmark_func_by_time(secs, [&]() {
            naive_copy(src_ptr, raw_pointer_cast(d_vec.data()), bytes);
        });
        cuda_acc_check(h_vec, d_vec);
    }
}
