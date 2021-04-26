cmake_minimum_required(VERSION 3.19)
project(LearnCuda CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(LearnCuda main.cu)

set_target_properties(
        LearnCuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)