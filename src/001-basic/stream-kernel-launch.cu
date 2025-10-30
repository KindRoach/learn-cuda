#include "cpp-bench-utils/utils.hpp"

__global__ void hello_from_kernel(size_t kernel_id) {
    printf(
        "Kernel %llu: Hello from (Block:%d, Thread:%d)\n",
        kernel_id, blockIdx.x, threadIdx.x
    );
}

void single_stream(size_t n_kernel) {
    using namespace cbu;
    std::cout << "Run with single stream:\n";

    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream));

    for (int i = 0; i < n_kernel; i++) {
        hello_from_kernel<<<1, 1, 0, stream>>>(i);
    }

    cuda_check(cudaStreamSynchronize(stream));
    cuda_check(cudaStreamDestroy(stream));

    std::cout << "\n";
}

void multi_stream(size_t n_kernel) {
    using namespace cbu;
    std::cout << "Run with multi stream:\n";

    std::vector<cudaStream_t> streams(n_kernel);
    for (auto &stream: streams) {
        cuda_check(cudaStreamCreate(&stream));
    }

    for (int i = 0; i < n_kernel; i++) {
        hello_from_kernel<<<1, 1, 0, streams[i]>>>(i);
    }

    for (auto &stream: streams) {
        cuda_check(cudaStreamSynchronize(stream));
    }

    for (auto &stream: streams) {
        cuda_check(cudaStreamDestroy(stream));
    }

    std::cout << "\n";
}


int main() {
    size_t n_kernel = 100;
    single_stream(n_kernel);
    multi_stream(n_kernel);
}
