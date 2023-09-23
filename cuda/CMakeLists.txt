cmake_minimum_required(VERSION 3.18)

project(HelloWorld)
enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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
    -Wno-unused-function
)
target_include_directories(
    playground PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
)

FILE(GLOB tests ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cu)

FOREACH(test ${tests})
    GET_FILENAME_COMPONENT(filename ${test} NAME_WE)
    add_executable(${filename} ${test})
    set_target_properties(${filename} PROPERTIES CUDA_ARCHITECTURES 75)
    target_compile_options(
        ${filename} PRIVATE
        -Wno-unused-function
        --expt-relaxed-constexpr
    )
    target_include_directories(
        ${filename} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
    )
    add_test(${filename} ${CMAKE_CURRENT_BINARY_DIR}/${filename})
ENDFOREACH()
