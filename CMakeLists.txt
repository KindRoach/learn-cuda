cmake_minimum_required(VERSION 3.25)

project(Learn-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(add_cuda_target target_name)
    add_executable(${target_name} ${ARGN})
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)
endfunction()

# Define root source directory and gather all .cu files recursively
set(src_root "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB_RECURSE sources "${src_root}/*.cu")

# For each .cu source file, create an individual executable
# Preserves directory structure relative to src/
foreach (file_path ${sources})
    # Extract executable name (from filename without extension)
    get_filename_component(target_name "${file_path}" NAME_WE)

    # Extract the directory of the file
    get_filename_component(target_dir "${file_path}" DIRECTORY)

    # Compute relative path to src root for output placement
    file(RELATIVE_PATH target_dir "${src_root}" "${target_dir}")

    add_cuda_target("${target_name}" "${file_path}")
endforeach()