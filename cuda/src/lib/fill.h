#pragma once

#include <cuda_runtime.h>

#pragma once

#include <cute/tensor.hpp>

using namespace cute;

namespace lib {
    namespace init {

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