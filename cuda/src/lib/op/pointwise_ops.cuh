#pragma once

#include <curand_kernel.h>

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        namespace detail {
            template <typename NAryFunc, typename TensorOut, typename... TensorIn>
            __global__ void n_ary_pointwise_kernel(
                TensorOut output, NAryFunc func, TensorIn... inputs) {
                using T = typename TensorOut::value_type;
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;
                for (; idx < size(output); idx += stride) {
                    output(idx) = static_cast<T>(func(idx, inputs(idx)...));
                }
            }

            template <typename NAryFunc, typename TensorOut, typename... TensorIn>
            void pointwise(TensorOut &output, NAryFunc &func, TensorIn &...inputs) {
                auto [grid_size, block_size] = launch_config(
                    n_ary_pointwise_kernel<NAryFunc, TensorOut, TensorIn...>, size(output));
                n_ary_pointwise_kernel<<<grid_size, block_size>>>(output, func, inputs...);
            }

            template <typename NAryFunc, typename TensorOut, typename... TensorIn>
            __global__ void n_ary_random_pointwise_kernel(
                TensorOut output, NAryFunc func, int seed = 0, TensorIn... inputs) {
                using T = typename TensorOut::value_type;
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = blockDim.x * gridDim.x;

                curandState state;
                curand_init(seed, idx, 0, &state);

                for (; idx < size(output); idx += stride) {
                    output(idx) = static_cast<T>(func(idx, state, inputs(idx)...));
                }
            }

            template <typename NAryFunc, typename TensorOut, typename... TensorIn>
            void pointwise_random(
                TensorOut &output, NAryFunc &func, int seed, TensorIn &...inputs) {
                auto [grid_size, block_size] = launch_config(
                    n_ary_random_pointwise_kernel<NAryFunc, TensorOut, TensorIn...>, size(output));
                n_ary_random_pointwise_kernel<<<grid_size, block_size>>>(
                    output, func, seed, inputs...);
            }

            template <typename T>
            struct Constant {
                __device__ auto operator()(int idx) { return value; }

                T value;
            };

            template <typename T = int>
            struct Arange {
                __device__ auto operator()(int idx) { return start + step * idx; }

                T start;
                T step;
            };

            struct Normal {
                __device__ auto operator()(int idx, curandState &state) {
                    return curand_normal(&state) * stddev + mean;
                }

                float mean;
                float stddev;
            };

            struct Uniform {
                __device__ auto operator()(int idx, curandState &state) {
                    return curand_uniform(&state) * (max - min) + min;
                }

                float min;
                float max;
            };

            struct Convert {
                template <typename T>
                __device__ auto operator()(int idx, T x) {
                    return x;
                }
            };

            struct ReLU {
                template <typename T>
                __device__ T operator()(int idx, T x) {
                    return x > static_cast<T>(0) ? x : static_cast<T>(0);
                }
            };

            struct dReLU {
                template <typename TY, typename TX>
                __device__ auto operator()(int idx, TY dy, TX x) {
                    return x > static_cast<TX>(0) ? dy : static_cast<TY>(0);
                }
            };

            struct Add {
                template <typename T>
                __device__ auto operator()(int idx, T x, T y) {
                    return x + y;
                }
            };

            struct SGD {
                template <typename Tx, typename Tdx>
                __device__ auto operator()(int idx, Tx x, Tdx dx) {
                    return x - static_cast<Tx>(lr) * static_cast<Tx>(dx);
                }

                float lr;
            };

            struct LION {
                template <typename Tx, typename Tdx>
                __device__ auto operator()(int idx, Tx x, Tdx dx) {
                    Tx delta = static_cast<Tx>(lr);
                    if (dx > static_cast<Tdx>(0))
                        return x - delta;
                    else if (dx < static_cast<Tdx>(0))
                        return x + delta;
                    else
                        return x;
                }

                float lr;
            };
        }  // namespace detail

        template <typename Tensor, typename T = typename Tensor::value_type>
        void constant(Tensor &output, T value = static_cast<T>(0)) {
            detail::Constant<T> func{value};
            detail::pointwise(output, func);
        }

        template <typename Tensor, typename T = int>
        void arange(Tensor &output, T start = 0, T step = 1) {
            detail::Arange<T> func{start, step};
            detail::pointwise(output, func);
        }

        template <typename Tensor>
        void normal(Tensor &output, float mean = 0.0f, float stddev = 1.0f, int seed = 0) {
            detail::Normal func{mean, stddev};
            detail::pointwise_random(output, func, seed);
        }

        template <typename Tensor>
        void uniform(Tensor &output, float min = 0.0f, float max = 1.0f, int seed = 0) {
            detail::Uniform func{min, max};
            detail::pointwise_random(output, func, seed);
        }

        template <typename TensorOut, typename TensorIn>
        void convert(TensorOut &output, TensorIn &input) {
            detail::Convert func;
            detail::pointwise(output, func, input);
        }

        template <typename TensorOut, typename TensorIn>
        void relu(TensorOut &output, TensorIn &input) {
            detail::ReLU func;
            detail::pointwise(output, func, input);
        }

        template <typename TensorOut, typename TensorDy, typename TensorX>
        void drelu(TensorOut &output, TensorDy &dy, TensorX &x) {
            detail::dReLU func;
            detail::pointwise(output, func, dy, x);
        }

        template <typename TensorOut, typename TensorX, typename TensorY>
        void add(TensorOut &output, TensorX &x, TensorY &y) {
            detail::Add func;
            detail::pointwise(output, func, x, y);
        }

        template <typename TensorOut, typename TensorX, typename TensorDx>
        void sgd(TensorOut &output, TensorX &x, TensorDx &dx, float lr) {
            detail::SGD func{lr};
            detail::pointwise(output, func, x, dx);
        }

        template <typename TensorOut, typename TensorX, typename TensorDx>
        void lion(TensorOut &output, TensorX &x, TensorDx &dx, float lr) {
            detail::LION func{lr};
            detail::pointwise(output, func, x, dx);
        }
    }  // namespace op
}  // namespace lib