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
        template <
            int I,
            typename UnaryOp,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        __global__ void unreduce_kernel(
            Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y, UnaryOp op) {
            using TA = typename EngineA::value_type;

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            int reduce_dim_size = size<I>(x.shape());

            for (; idx < size(y); idx += stride) {
                auto y_coord = idx2crd(idx, y.shape());
                auto new_value = op(y(y_coord));
                for (int i = 0; i < reduce_dim_size; ++i) {
                    auto x_coord = insert<I>(y_coord, i);
                    x(x_coord) = TA(new_value);
                }
            }
        }

        template <
            int I,
            typename UnaryOp,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        void unreduce(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y, UnaryOp op) {
            assert(remove<I>(x.shape()) == y.shape());

            auto [grid_size, block_size] = launch_config(
                unreduce_kernel<I, UnaryOp, EngineA, LayoutA, EngineB, LayoutB>, size(y));

            unreduce_kernel<I><<<grid_size, block_size>>>(x, y, op);
        }

        struct Repeat {
            template <typename T>
            CUTE_HOST_DEVICE T operator()(T a) const {
                return a;
            }
        };

        template <int I, typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        void repeat(
            Tensor<EngineA, LayoutA> x,  // output
            Tensor<EngineB, LayoutB> y   // input
        ) {
            Repeat repeat_func;
            unreduce<I>(x, y, repeat_func);
        }

        struct dMean {
            template <typename T>
            CUTE_HOST_DEVICE T operator()(T dx) const {
                return dx / T(size);
            }

            int size;
        };

        template <int I, typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        void mean_bwd(Tensor<EngineA, LayoutA> dx, Tensor<EngineB, LayoutB> dy) {
            dMean d_mean{size<I>(dx.shape())};
            unreduce<I>(dx, dy, d_mean);
        }
    }  // namespace op
}  // namespace lib