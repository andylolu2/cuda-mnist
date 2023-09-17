#pragma once

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Apply a SGD step to the given tensor.
         */
        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename TA = typename EngineA::value_type,
            typename TB = typename EngineB::value_type>
        __global__ void sgd_kernel(
            Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> dx, float lr) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(x); idx += stride) {
                x(idx) -= TA(lr * dx(idx));
            }
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename TA = typename EngineA::value_type,
            typename TB = typename EngineB::value_type>
        void sgd(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> dx, float lr) {
            assert(size(x) == size(dx));

            auto [grid_size, block_size] =
                launch_config(sgd_kernel<EngineA, LayoutA, EngineB, LayoutB, TA, TB>, size(x));
            sgd_kernel<<<grid_size, block_size>>>(x, dx, lr);
        }
    }  // namespace op
}  // namespace lib