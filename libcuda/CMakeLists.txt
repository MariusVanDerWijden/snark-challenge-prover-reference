set(CMAKE_C_COMPILER "/usr/bin/gcc-7")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-7")
SET (CMAKE_C_COMPILER_WORKS 1)
SET (CMAKE_CXX_COMPILER_WORKS 1)
SET (CMAKE_CUDA_COMPILER_WORKS 1)

project(libcuda CXX CUDA)
enable_language(CUDA)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xptxas; -v; -gencode arch=compute_75,code=sm_75; -O3; --ptxas-options=-v; -keep; -use_fast_math; -dc)
list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3) 

cuda_add_library (
    libcuda STATIC
    device_multiplication.cu
    device_multiplication.h
    device_field_operators.h
    device_field.h
)
target_include_directories(libcuda PUBLIC ${CUDA_INCLUDE_DIRS})
set_property(TARGET libcuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(libcuda ${CUDA_LIBRARIES})