#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include "util/util.cuh"

template<typename T>
void vector_dot_ref(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &out) {
    out[0] = std::inner_product(a.begin(), a.end(), b.begin(), static_cast<T>(0));
}

template<typename T>
void vector_sum_thrust(thrust::device_vector<T> &a, thrust::device_vector<T> &b, thrust::device_vector<T> &out) {
    out[0] = thrust::inner_product(a.begin(), a.end(), b.begin(), static_cast<T>(0));
}

int main() {
    using dtype = int;
    using d_vec = thrust::device_vector<dtype>;

    size_t secs = 10;
    size_t size = 100 * 1024 * 1024; // 100M elements

    std::vector<dtype> a(size), b(size), out_ref(1);
    random_fill(a);
    random_fill(b);

    std::cout << "vector_dot_ref:\n";
    benchmark_func_by_time(secs, [&] { vector_dot_ref(a, b, out_ref); });

    d_vec d_a = a;
    d_vec d_b = b;
    d_vec d_out(1);

    using func_t = std::function<void(d_vec &, d_vec &, d_vec &)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"vector_sum_thrust", vector_sum_thrust<dtype>},
    };

    for (auto [func_name, func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        thrust::fill(d_out.begin(), d_out.end(), dtype{0});
        benchmark_func_by_time(secs, [&]() {
            func(d_a, d_b, d_out);
            cuda_check(cudaDeviceSynchronize());
        });

        cuda_acc_check(out_ref, d_out);
    }
}
