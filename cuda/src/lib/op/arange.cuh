#pragma once

#include <cuda_runtime.h>

#pragma once

#include <cute/tensor.hpp>

using namespace cute;

namespace lib {
    namespace op {
        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        __global__ void arange_kernel(Tensor<Engine, Layout> tensor, T start, T step) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(tensor); idx += stride) {
                tensor(idx) = idx * step + start;
            }
        }

        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        void arange(Tensor<Engine, Layout> tensor, T start = T(0), T step = T(1)) {
            auto [grid_size, block_size] =
                launch_config(arange_kernel<Engine, Layout, T>, size(tensor));
            arange_kernel<<<grid_size, block_size>>>(tensor, start, step);
        }
    }  // namespace op
}  // namespace lib