#pragma once

#include <cute/algorithm/functional.hpp>
#include <cute/algorithm/tuple_algorithms.hpp>
#include <cute/config.hpp>
#include <cute/stride.hpp>
#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        namespace detail {
            template <int I, typename UnreduceOp, typename TensorOut, typename TensorIn>
            __global__ void unreduce_kernel(TensorOut output, TensorIn input, UnreduceOp op) {
                using T = typename TensorOut::value_type;

                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;

                int reduce_dim_size = size<I>(output);

                for (; idx < size(input); idx += stride) {
                    auto input_coord = idx2crd(idx, input.shape());
                    auto new_value = static_cast<T>(op(input(input_coord)));
                    for (int i = 0; i < reduce_dim_size; ++i) {
                        auto output_coord = insert<I>(input_coord, i);
                        output(output_coord) = new_value;
                    }
                }
            }

            template <int I, typename UnreduceOp, typename TensorOut, typename TensorIn>
            void unreduce(TensorOut const& output, TensorIn const& input, UnreduceOp op) {
                assert(remove<I>(output.shape()) == input.shape());

                auto [grid_size, block_size] = launch_config(size(input));

                unreduce_kernel<I><<<grid_size, block_size>>>(output, input, op);
            }

            template <typename TScale = uint8_t>
            struct RepeatWithScale {
                template <typename T>
                CUTE_HOST_DEVICE T operator()(T dx) const {
                    return dx * static_cast<T>(scale);
                }

                TScale scale = static_cast<TScale>(1);
            };
        }  // namespace detail

        template <int I, typename TensorOut, typename TensorIn>
        void repeat(TensorOut const& output, TensorIn const& input) {
            detail::RepeatWithScale op;
            detail::unreduce<I>(output, input, op);
        }
    }  // namespace op
}  // namespace lib