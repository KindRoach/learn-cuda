cmake_minimum_required(VERSION 3.19)
project(LearnCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(LearnCuda main.cu
        common/utils.cuh common/utils.cu
        common/vectorHelper.cuh common/vectorHelper.cu
        task/vectorAdd.cuh task/vectorAdd.cu
        task/vectorSum.cuh task/vectorSum.cu
        common/deviceInfo.cu common/deviceInfo.cuh)

set_target_properties(
        LearnCuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)