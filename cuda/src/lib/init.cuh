#pragma once

#include "lib/types.h"

template <typename T>
__global__ void fill_kernel(T* devPtr, const T val, const size_t nwords) {
    int stride = blockDim.x * gridDim.x;

    for (size_t idx = threadIdx.x + blockDim.x * blockIdx.x; idx < nwords; idx += stride) {
        devPtr[idx] = __float2half(val);
    }
}
