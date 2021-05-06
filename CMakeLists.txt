cmake_minimum_required(VERSION 3.19)
project(LearnCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(LearnCuda main.cu
        common/utils.cuh common/utils.cu
        common/vectorHelper.cuh common/vectorHelper.cu
        common/deviceInfo.cu common/deviceInfo.cuh
        task/nestHelloWorld.cu task/nestHelloWorld.cuh
        task/algorithm/vectorAdd.cuh task/algorithm/vectorAdd.cu
        task/algorithm/vectorSum.cuh task/algorithm/vectorSum.cu
        task/memory/manualMemory.cu task/memory/manualMemory.cuh
        task/memory/pinnedMemory.cu task/memory/pinnedMemory.cuh
        task/memory/zeroCopyMemory.cu task/memory/zeroCopyMemory.cuh
        task/memory/unifiedMemory.cu task/memory/unifiedMemory.cuh
        task/memory/misalignedRead.cu task/memory/misalignedRead.cuh
        task/algorithm/matrixTranspose.cu task/algorithm/matrixTranspose.cuh common/arrayHelper.cu common/arrayHelper.cuh)

set_target_properties(
        LearnCuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)