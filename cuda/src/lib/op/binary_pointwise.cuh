#pragma once

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Applies a binary function to each element of a tensor.
         */
        template <
            typename BinaryFunc,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        __global__ void binary_pointwise_kernel(
            Tensor<EngineA, LayoutA> tensor_a,
            Tensor<EngineB, LayoutB> tensor_b,
            Tensor<EngineC, LayoutC> output,
            BinaryFunc func) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            for (; idx < size(tensor_a); idx += stride) {
                output(idx) = func(tensor_a(idx), tensor_b(idx));
            }
        }

        template <
            typename BinaryFunc,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        void binary_pointwise(
            Tensor<EngineA, LayoutA> tensor_a,
            Tensor<EngineB, LayoutB> tensor_b,
            Tensor<EngineC, LayoutC> output,
            BinaryFunc func) {
            assert(size(tensor_a) == size(tensor_b));
            assert(size(tensor_a) == size(output));

            auto [grid_size, block_size] = launch_config(
                binary_pointwise_kernel<
                    BinaryFunc,
                    EngineA,
                    LayoutA,
                    EngineB,
                    LayoutB,
                    EngineC,
                    LayoutC>,
                size(tensor_a));
            binary_pointwise_kernel<<<grid_size, block_size>>>(tensor_a, tensor_b, output, func);
        }
    }  // namespace op
}  // namespace lib