cmake_minimum_required(VERSION 3.25)

project(Learn-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(add_cuda_target target_name)

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

    # Add target
    add_executable("${target_name}" "${file_path}")

    # Set CUDA architectures
    set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)

    # Include the src directory for headers
    target_include_directories("${target_name}" PRIVATE src)

    # Set output directory to bin/<relative path of src>
    set_target_properties("${target_name}" PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${target_dir}"
    )
endforeach ()