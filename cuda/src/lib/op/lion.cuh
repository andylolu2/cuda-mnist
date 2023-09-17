#pragma once

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Apply a LION step to the given tensor.
         */
        template <typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        __global__ void lion_kernel(
            Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> dx, float lr) {
            using TA = typename EngineA::value_type;
            using TB = typename EngineB::value_type;
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(x); idx += stride) {
                x(idx) += TA(dx(idx) > TB(0) ? -lr : lr);
            }
        }

        template <typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        void lion(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> dx, float lr) {
            assert(size(x) == size(dx));

            auto [grid_size, block_size] =
                launch_config(lion_kernel<EngineA, LayoutA, EngineB, LayoutB>, size(x));
            lion_kernel<<<grid_size, block_size>>>(x, dx, lr);
        }
    }  // namespace op
}  // namespace lib