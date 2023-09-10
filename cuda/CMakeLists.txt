cmake_minimum_required(VERSION 3.18)

project(HelloWorld)
enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# file(GLOB_RECURSE lib_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/*.cu)
add_executable(
    cute_test
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cute_test.cu
)
set_target_properties(cute_test PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    cute_test PRIVATE
    -Wall
    -Wno-unused-function
    --expt-relaxed-constexpr
    --extended-lambda
)
target_include_directories(
    cute_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)

# MNIST
add_executable(
    mnist
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mnist.cu
)
set_target_properties(mnist PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    mnist PRIVATE
    -Wall
    -Wno-unused-function
)
target_include_directories(
    mnist PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)

add_executable(
    playground
    ${CMAKE_CURRENT_SOURCE_DIR}/src/playground.cu
)
set_target_properties(playground PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    playground PRIVATE
    -Wall
    -Wno-unused-function
)
target_include_directories(
    playground PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)