#include "util/util.cuh"

template<typename T>
__global__ void copy_data(T *src, T *dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    dst[i] = src[i];
}

template<typename T>
__global__ void add_one(T *data) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] += 1;
}

template<typename T, size_t BLOCK_SIZE>
void without_graph(thrust::device_vector<T> &data, thrust::device_vector<T> &out, size_t n_kernel) {
    size_t size = data.size();
    check_divisible(size, BLOCK_SIZE, "Global size must be divisible by BLOCK_SIZE");
    size_t grid_size = size / BLOCK_SIZE;

    copy_data<<<grid_size, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(out.data())
    );

    for (size_t i = 0; i < n_kernel; ++i) {
        add_one<<<grid_size, BLOCK_SIZE>>>(thrust::raw_pointer_cast(out.data()));
    }
}

template<typename T, size_t BLOCK_SIZE>
void with_graph(thrust::device_vector<T> &data, thrust::device_vector<T> &out, size_t n_kernel) {
    size_t size = data.size();
    check_divisible(size, BLOCK_SIZE, "Global size must be divisible by BLOCK_SIZE");
    size_t grid_size = size / BLOCK_SIZE;

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;

    cudaStreamCreate(&stream);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // grap start here
    copy_data<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        thrust::raw_pointer_cast(data.data()),
        thrust::raw_pointer_cast(out.data())
    );

    for (size_t i = 0; i < n_kernel; ++i) {
        add_one<<<grid_size, BLOCK_SIZE, 0, stream>>>(thrust::raw_pointer_cast(out.data()));
    }
    // grap end here

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}


int main() {
    using dtype = float;
    using d_vec = thrust::device_vector<dtype>;
    constexpr uint16_t block_size = 256;

    size_t secs = 10;
    size_t n_kernel = 1024;

    std::vector<dtype> h_data(block_size);
    random_fill(h_data);

    std::vector<dtype> h_out = h_data;
    for (size_t i = 0; i < block_size; ++i) {
        for (size_t j = 0; j < n_kernel; ++j) {
            h_out[i] += 1;
        }
    }

    d_vec d_data = h_data;
    d_vec d_out(block_size);

    using func_t = std::function<void(d_vec &, d_vec &, size_t)>;
    std::vector<std::tuple<std::string, func_t> > funcs{
        {"without_graph", without_graph<dtype, block_size>},
        {"with_graph", with_graph<dtype, block_size>},
    };

    for (auto [func_name,func]: funcs) {
        std::cout << "\n" << func_name << ":\n";
        benchmark_func_by_time(secs, [&]() {
            func(d_data, d_out, n_kernel);
            cuda_check(cudaDeviceSynchronize());
        });
        cuda_acc_check(h_out, d_out);
    }
}
