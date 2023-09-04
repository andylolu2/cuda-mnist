#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cute/tensor.hpp>

using namespace cute;

namespace lib {
    namespace init {

        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        __global__ void constant(Tensor<Engine, Layout> tensor, T value = T(0)) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            for (; idx < size(tensor); idx += stride) {
                tensor(idx) = value;
            }
        }

        template <typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        __global__ void constant(Tensor<EngineA, LayoutA> tensor, Tensor<EngineB, LayoutB> value) {
            assert(size(value) == 1);

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(tensor); idx += stride) {
                tensor(idx) = value(0);
            }
        }

        template <typename Engine, typename Layout>
        __global__ void identity(Tensor<Engine, Layout> tensor) {
            using T = typename Engine::value_type;
            static_assert(Layout::rank == 2, "identity matrix must be 2d");

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            int dim = min(tensor.shape());
            for (; idx < dim; idx += stride) {
                tensor(idx, idx) = T(1);
            }
        }

        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        __global__ void normal(
            Tensor<Engine, Layout> tensor,
            T mean = T(0),
            T std = T(1),
            unsigned long long seed = 0) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            curandState state;
            curand_init(seed, idx, 0, &state);

            for (; idx < size(tensor); idx += stride) {
                tensor(idx) = curand_normal(&state) * std + mean;
            }
        }

        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        __global__ void arange(Tensor<Engine, Layout> tensor, T start = T(0), T step = T(1)) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(tensor); idx += stride) {
                tensor(idx) = T(idx * step + start);
            }
        }

        template <typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        __global__ void cum_copy(Tensor<EngineA, LayoutA> src, Tensor<EngineB, LayoutB> dst) {
            using T = typename EngineB::value_type;

            assert(size(src) == size(dst));

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(src); idx += stride) {
                dst(idx) += T(src(idx));
            }
        }
    }  // namespace init
}  // namespace lib