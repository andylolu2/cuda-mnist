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
            template <int I, typename ReduceOp, typename TensorX, typename TensorY, typename T>
            __global__ void reduce_kernel(TensorX x, TensorY y, ReduceOp op, T acc) {
                using TY = typename TensorY::value_type;

                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;

                int reduce_dim_size = size<I>(x.shape());

                for (; idx < size(y); idx += stride) {
                    auto y_coord = idx2crd(idx, y.shape());
                    for (int i = 0; i < reduce_dim_size; ++i) {
                        auto x_coord = insert<I>(y_coord, i);
                        acc = op(acc, x(x_coord));
                    }
                    y(y_coord) = static_cast<TY>(acc);
                }
            }

            template <int I, typename ReduceOp, typename TensorX, typename TensorY, typename T>
            void reduce(TensorX x, TensorY y, ReduceOp op, T acc) {
                assert(remove<I>(x.shape()) == y.shape());

                auto [grid_size, block_size] = launch_config(size(y));

                reduce_kernel<I><<<grid_size, block_size>>>(x, y, op, acc);
            }

            template <typename TAcc>
            struct Sum {
                template <typename T>
                CUTE_HOST_DEVICE TAcc operator()(TAcc a, T b) {
                    return a + static_cast<TAcc>(b);
                }
            };

            template <typename TAcc>
            struct Mean {
                template <typename T>
                CUTE_HOST_DEVICE TAcc operator()(TAcc a, T b) const {
                    return a + static_cast<TAcc>(b) / static_cast<TAcc>(size);
                }
                int size;
            };
        }  // namespace detail

        template <int I, typename TensorX, typename TensorY>
        void sum(TensorX x, TensorY y) {
            detail::Sum<float> func;
            detail::reduce<I>(x, y, func, 0.0f);
        }

        template <int I, typename TensorX, typename TensorY>
        void mean(TensorX x, TensorY y) {
            detail::Mean<float> func{size<I>(x)};
            detail::reduce<I>(x, y, func, 0.0f);
        }
    }  // namespace op
}  // namespace lib