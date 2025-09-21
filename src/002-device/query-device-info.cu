#include <iostream>

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        std::cout << "cudaGetDeviceCount returned " << static_cast<int>(error_id)
                << " -> " << cudaGetErrorString(error_id) << std::endl;
        std::cout << "No CUDA devices found." << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "\nDevice " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  CUDA Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Registers per Block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Threads Dimensions: ["
                << deviceProp.maxThreadsDim[0] << ", "
                << deviceProp.maxThreadsDim[1] << ", "
                << deviceProp.maxThreadsDim[2] << "]" << std::endl;
        std::cout << "  Max Grid Size: ["
                << deviceProp.maxGridSize[0] << ", "
                << deviceProp.maxGridSize[1] << ", "
                << deviceProp.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << "-bit" << std::endl;
        std::cout << "  Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
    }

    return 0;
}
