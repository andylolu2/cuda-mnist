cmake_minimum_required(VERSION 3.18)

project(HelloWorld)
enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(cudnnExample src/cudnnExample.cu)

file(GLOB_RECURSE lib_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/*.cu)
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/src/test.cu ${lib_SRC})

set_target_properties(cudnnExample test PROPERTIES CUDA_ARCHITECTURES 75)

target_compile_options(
    test PRIVATE
    -Wall
    -Wno-unused-function
)

target_include_directories(
    cudnnExample PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)

target_include_directories(
    test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cudnn-frontend/include
)
target_link_libraries(
    cudnnExample cudnn
)
target_link_libraries(
    test cudnn
)

add_executable(
    cutlass_test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cutlass_test.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/operators/matmul_bias_relu.cu
)
set_target_properties(cutlass_test PROPERTIES CUDA_ARCHITECTURES 75)

target_compile_options(
    cutlass_test PRIVATE
    -Wall
    -Wno-unused-function
)

target_include_directories(
    cutlass_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)