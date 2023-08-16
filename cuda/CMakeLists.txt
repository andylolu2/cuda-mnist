cmake_minimum_required(VERSION 3.18)

project(HelloWorld)
enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(cudnnExample src/cudnnExample.cu)

file(GLOB_RECURSE lib_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/*.cu)
message(STATUS "lib_SRC: ${lib_SRC}")
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