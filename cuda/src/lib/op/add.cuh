#pragma once

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Add two tensors element-wise.
         */
        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename TA = typename EngineA::value_type,
            typename TB = typename EngineB::value_type>
        __global__ void add_kernel(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(x); idx += stride) {
                x(idx) += TA(y(idx));
            }
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename TA = typename EngineA::value_type,
            typename TB = typename EngineB::value_type>
        void add(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y) {
            assert(size(x) == size(y));

            auto [grid_size, block_size] =
                launch_config(add_kernel<EngineA, LayoutA, EngineB, LayoutB, TA, TB>, size(x));

            add_kernel<<<grid_size, block_size>>>(x, y);
        }
    }  // namespace op
}  // namespace lib