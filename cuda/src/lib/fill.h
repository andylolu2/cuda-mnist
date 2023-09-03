#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cute/tensor.hpp>

namespace lib {
    namespace init {
        template <typename Tensor, typename T>
        __global__ void constant(Tensor tensor, T value) {
            static_assert(std::is_same_v<T, typename Tensor::value_type>, "value type mismatch");
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            for (; idx < tensor.size(); idx += stride) {
                tensor(idx) = value;
            }
        }

        template <typename Tensor>
        __global__ void identity(Tensor tensor) {
            using T = typename Tensor::value_type;
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int dim = min(cute::size<0>(tensor), cute::size<1>(tensor));
            for (; idx < dim; idx += stride) {
                tensor(idx, idx) = T(1);
            }
        }

        template <typename Tensor, typename T>
        __global__ void normal(Tensor tensor, T mean, T std = T(1), unsigned long long seed = 0) {
            static_assert(std::is_same_v<T, typename Tensor::value_type>, "value type mismatch");

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            curandState state;
            curand_init(seed, idx, 0, &state);

            for (; idx < tensor.size(); idx += stride) {
                tensor(idx) = curand_normal(&state) * std + mean;
            }
        }
    }  // namespace init
}  // namespace lib