#pragma once

#include <curand_kernel.h>

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Sets all elements of the tensor to a random normal value.
         */
        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        __global__ void normal_kernel(
            Tensor<Engine, Layout> tensor,
            T mean = T(0),
            T std = T(1),
            unsigned long long seed = 0) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            curandState state;
            curand_init(seed, idx, 0, &state);

            for (; idx < size(tensor); idx += stride) {
                tensor(idx) = curand_normal(&state) * std + mean;
            }
        }

        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        void normal(
            Tensor<Engine, Layout> tensor,
            T mean = T(0),
            T std = T(1),
            unsigned long long seed = 0) {
            auto [grid_size, block_size] =
                launch_config(normal_kernel<Engine, Layout, T>, size(tensor));
            normal_kernel<<<grid_size, block_size>>>(tensor, mean, std, seed);
        }
    }  // namespace op
}  // namespace lib