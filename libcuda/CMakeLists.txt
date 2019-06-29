set(CMAKE_C_COMPILER "/usr/bin/gcc-7")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")
SET (CMAKE_C_COMPILER_WORKS 1)
SET (CMAKE_CXX_COMPILER_WORKS 1)
SET (CMAKE_CUDA_COMPILER_WORKS 1)

project(libcuda CXX CUDA)
enable_language(CUDA)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v; -lineinfo; -use_fast_math; -gencode arch=compute_50,code=sm_50)
list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)

cuda_add_library (
    libcuda STATIC
    device_multiplication.cu
    device_multiplication.h
    device_field_operators.h
    device_field.h
)

target_include_directories(libcuda PUBLIC ${CUDA_INCLUDE_DIRS})