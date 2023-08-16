#pragma once

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <stdio.h>

#include <stdexcept>

#include "lib/types.h"

#define check_cuda_status(...) \
    { cuda_assert(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); }
inline void cuda_assert(cudaError status, const char *expr, const char *file, int line) {
    if (status != cudaSuccess) {
        printf("CUDA error at %s:%d, status=%d (%s) in '%s'\n", file, line, (int)status,
               cudaGetErrorString(status), expr);
        exit(status);
    }
}

#define check_cudnn_status(...) \
    { check_cudnn_error(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); }
inline void check_cudnn_error(cudnnStatus_t status, const char *expr, const char *file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("CUDNN error at %s:%d, status=%d (%s) in '%s'\n", file, line, (int)status,
               cudnnGetErrorString(status), expr);
        exit(status);
    }
}

int64_t hashString(std::string s);

cudnnStatus_t execute_cached_plan(cudnnHandle_t handle,
                                  cudnn_frontend::ExecutionPlanCache &plan_cache,
                                  cudnn_frontend::OperationGraph &opGraph,
                                  std::set<std::pair<uint64_t, void *>> &data_ptrs);
