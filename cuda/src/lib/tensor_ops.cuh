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

        template <
            int axis,
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        __global__ void cross_entropy_with_logits_bwd(
            Tensor<EngineA, LayoutA> y_pred,
            Tensor<EngineB, LayoutB> y_true,
            Tensor<EngineC, LayoutC> dy_pred) {
            using TC = typename EngineC::value_type;

            static_assert(LayoutA::rank == 2, "y_pred should be a 2D matrix");
            static_assert(LayoutB::rank == 1, "y_true should be a 1D matrix");
            static_assert(LayoutC::rank == 2, "dy_pred should be a 2D matrix");

            constexpr int b_axis = axis == 0 ? 1 : 0;

            // A batch is handled by one thread
            int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; batch_idx < size<b_axis>(y_pred); batch_idx += stride) {
                float max_v = -FLT_MAX;
                for (int i = 0; i < size<axis>(y_pred); ++i) {
                    auto coord = axis == 0 ? make_coord(batch_idx, i) : make_coord(i, batch_idx);
                    max_v = max(max_v, float(y_pred(coord)));
                }

                float sum = 0;
                for (int i = 0; i < size<axis>(y_pred); ++i) {
                    auto coord = axis == 0 ? make_coord(batch_idx, i) : make_coord(i, batch_idx);
                    sum += exp(y_pred(batch_idx, i) - max_v);
                }

                int label = y_true(batch_idx);
                for (int i = 0; i < size<axis>(y_pred); ++i) {
                    auto coord = axis == 0 ? make_coord(batch_idx, i) : make_coord(i, batch_idx);
                    float p = exp(y_pred(coord) - max_v) / sum;
                    dy_pred(coord) = i == label ? TC(p - 1) : TC(p);
                }
            }
        }
    }  // namespace op
}  // namespace lib
