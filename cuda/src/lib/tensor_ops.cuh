#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>

using namespace cute;

namespace lib {
    namespace op {
        template <typename Engine, typename Layout>
        auto transpose(Tensor<Engine, Layout> const& tensor) {
            auto new_shape = make_shape(size<1>(tensor.shape()), size<0>(tensor.shape()));
            auto new_stride = make_stride(size<1>(tensor.stride()), size<0>(tensor.stride()));
            return make_tensor(tensor.engine().begin(), new_shape, new_stride);
        }

        template <typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        __global__ void sum(Tensor<EngineA, LayoutA> x, Tensor<EngineB, LayoutB> y) {
            using T = typename EngineB::value_type;

            assert(size(x) == size(y));

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(x); idx += stride) {
                // dst(idx) += T(src(idx));
                T* y_ptr = (y.engine().begin() + y.layout()(idx)).get();
                atomicAdd(y_ptr, T(x(idx)));
            }
        }

        template <typename EngineA, typename LayoutA, typename EngineB, typename LayoutB>
        __global__ void sum_bwd(Tensor<EngineA, LayoutA> dx, Tensor<EngineB, LayoutB> dy) {
            using T = typename EngineA::value_type;

            assert(size(dx) == size(dy));

            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; idx < size(dx); idx += stride) {
                dx(idx) = T(dy(idx));
            }
        }
    }  // namespace op
}  // namespace lib
