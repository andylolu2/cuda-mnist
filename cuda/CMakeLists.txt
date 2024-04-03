cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(HelloWorld)
enable_language(CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

FILE(
    GLOB executables
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

FOREACH(exe ${executables})
    GET_FILENAME_COMPONENT(filename ${exe} NAME_WE)
    add_executable(${filename} ${exe})
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
