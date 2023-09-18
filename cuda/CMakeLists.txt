cmake_minimum_required(VERSION 3.18)

project(HelloWorld)
enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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

add_executable(
    test_matmul
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_matmul.cu
)
set_target_properties(test_matmul PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    test_matmul PRIVATE
    -Wall
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_include_directories(
    test_matmul PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)

add_executable(
    test_linear
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_linear.cu
)
set_target_properties(test_linear PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    test_linear PRIVATE
    -Wall
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_include_directories(
    test_linear PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)

add_executable(
    test_mlp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_mlp.cu
)
set_target_properties(test_mlp PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    test_mlp PRIVATE
    -Wall
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_include_directories(
    test_mlp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)

add_executable(
    test_cross_entropy
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/test_cross_entropy.cu
)
set_target_properties(test_cross_entropy PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    test_cross_entropy PRIVATE
    -Wall
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_include_directories(
    test_cross_entropy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)