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
        /**
         * Reduce one dimension of a tensor.
         */
        template <
            int I,
            typename ReduceOp,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        __global__ void reduce_kernel(
            Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y, ReduceOp op) {
            using TB = typename EngineA::value_type;
            using TAcc = typename ReduceOp::acc_type;

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            int reduce_dim_size = size<I>(x.shape());

            for (; idx < size(y); idx += stride) {
                TAcc acc = TAcc(0);
                auto y_coord = idx2crd(idx, y.shape());
                for (int i = 0; i < reduce_dim_size; ++i) {
                    auto x_coord = insert<I>(y_coord, i);
                    acc = op(acc, x(x_coord));
                }
                y(y_coord) = TB(acc);
            }
        }

        template <
            int I,
            typename ReduceOp,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        void reduce(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y, ReduceOp op) {
            assert(remove<I>(x.shape()) == y.shape());

            auto [grid_size, block_size] = launch_config(
                reduce_kernel<I, ReduceOp, EngineA, LayoutA, EngineB, LayoutB>, size(y));

            reduce_kernel<I><<<grid_size, block_size>>>(x, y, op);
        }

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

        template <typename AccType>
        struct Add {
            using acc_type = AccType;

            template <typename T>
            CUTE_HOST_DEVICE AccType operator()(AccType a, T b) const {
                return a + AccType(b);
            }
        };

        template <int I, typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        void sum(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y) {
            Add<float> add;
            reduce<I>(x, y, add);
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

        template <int I, typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        void sum_bwd(Tensor<EngineA, LayoutA> dx, Tensor<EngineB, LayoutB> dy) {
            repat(dx, dy);
        }

        template <typename AccType>
        struct Mean {
            using acc_type = AccType;

            template <typename T>
            CUTE_HOST_DEVICE AccType operator()(AccType a, T b) const {
                return a + AccType(b) / AccType(size);
            }

            int size;
        };

        template <int I, typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        void mean(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y) {
            Mean<float> mean{size<I>(x.shape())};
            reduce<I>(x, y, mean);
        }

        struct dMean {
            template <typename T>
            CUTE_HOST_DEVICE T operator()(T a) const {
                return a / T(size);
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