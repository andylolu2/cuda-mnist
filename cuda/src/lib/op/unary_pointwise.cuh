#pragma once

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Applies a unary function to each element of a tensor.
         */
        template <
            typename UnaryFunc,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        __global__ void unary_pointwise_kernel(
            Tensor<EngineA, LayoutA> input, Tensor<EngineB, LayoutB> output, UnaryFunc func) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            for (; idx < size(input); idx += stride) {
                output(idx) = func(input(idx));
            }
        }

        template <
            typename UnaryFunc,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        void unary_pointwise(
            Tensor<EngineA, LayoutA> &input, Tensor<EngineB, LayoutB> &output, UnaryFunc &func) {
            assert(size(input) == size(output));

            auto [grid_size, block_size] = launch_config(
                unary_pointwise_kernel<UnaryFunc, EngineA, LayoutA, EngineB, LayoutB>, size(input));
            unary_pointwise_kernel<<<grid_size, block_size>>>(input, output, func);
        }
    }  // namespace op
}  // namespace lib