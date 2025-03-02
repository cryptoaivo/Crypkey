cmake_minimum_required(VERSION 3.10)
project(CrypKey LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

set(SOURCES
    CrypKey.cpp
)

set(CUDA_SOURCES
    cuda_kernels.cu
    address_matcher.cu
    key_generator.cu
)

cuda_add_executable(CrypKey ${SOURCES} ${CUDA_SOURCES})
target_link_libraries(CrypKey ${CUDA_LIBRARIES})