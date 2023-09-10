#pragma once

#include <curand_kernel.h>

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

        template <
            typename UnaryRandomFunc,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        __global__ void unary_random_pointwise_kernel(
            Tensor<EngineA, LayoutA> input,
            Tensor<EngineB, LayoutB> output,
            UnaryRandomFunc func,
            unsigned long long seed) {
            using TB = typename EngineB::value_type;

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            curandState state;
            curand_init(seed, idx, 0, &state);

            for (; idx < size(input); idx += stride) {
                output(idx) = TB(func(input(idx), state));
            }
        }

        template <
            typename UnaryRandomFunc,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB>
        void unary_random_pointwise(
            Tensor<EngineA, LayoutA> &input,
            Tensor<EngineB, LayoutB> &output,
            UnaryRandomFunc &func,
            unsigned long long seed) {
            assert(size(input) == size(output));

            auto [grid_size, block_size] = launch_config(
                unary_random_pointwise_kernel<UnaryRandomFunc, EngineA, LayoutA, EngineB, LayoutB>,
                size(input));
            unary_random_pointwise_kernel<<<grid_size, block_size>>>(input, output, func, seed);
        }

        template <typename ComputeType>
        struct Normal {
            template <typename T>
            __device__ ComputeType operator()(T x, curandState &state) {
                return ComputeType(curand_normal(&state)) * stddev + mean;
            }

            ComputeType mean;
            ComputeType stddev;
        };
        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename TB = typename EngineB::value_type>
        void normal(
            Tensor<EngineA, LayoutA> &input,
            Tensor<EngineB, LayoutB> &output,
            TB mean = 0,
            TB stddev = 1,
            unsigned long long seed = 0) {
            Normal<TB> func{mean, stddev};
            unary_random_pointwise(input, output, func, seed);
        }

        template <typename ComputeType>
        struct Uniform {
            template <typename T>
            __device__ ComputeType operator()(T x, curandState &state) {
                return ComputeType(curand_uniform(&state)) * (max - min) + min;
            }

            ComputeType min;
            ComputeType max;
        };

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename TB = typename EngineB::value_type>
        void uniform(
            Tensor<EngineA, LayoutA> &input,
            Tensor<EngineB, LayoutB> &output,
            TB min = 0,
            TB max = 1,
            unsigned long long seed = 0) {
            Uniform<TB> func{min, max};
            unary_random_pointwise(input, output, func, seed);
        }
    }  // namespace op
}  // namespace lib