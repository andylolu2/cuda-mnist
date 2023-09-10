#pragma once

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Initialize a tensor with the identity matrix.
         */
        template <typename Engine, typename Layout>
        __global__ void identity_kernel(Tensor<Engine, Layout> tensor) {
            using T = typename Engine::value_type;
            static_assert(Layout::rank == 2, "identity matrix must be 2d");

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            int dim = min(tensor.shape());
            for (; idx < dim; idx += stride) {
                tensor(idx, idx) = T(1);
            }
        }

        template <typename Engine, typename Layout>
        void identity(Tensor<Engine, Layout> tensor) {
            auto [grid_size, block_size] =
                launch_config(identity_kernel<Engine, Layout>, min(tensor.shape()));
            identity_kernel<<<grid_size, block_size>>>(tensor);
        }
    }  // namespace op
}  // namespace lib