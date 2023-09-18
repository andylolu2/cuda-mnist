#pragma once

#include <cutlass/fast_math.h>

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        __global__ void cross_entropy_with_logits_fwd_kernel(
            Tensor<EngineA, LayoutA> y_pred,
            Tensor<EngineB, LayoutB> y_true,
            Tensor<EngineC, LayoutC> loss) {
            using TC = typename EngineC::value_type;

            // A batch is handled by one thread
            int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; batch_idx < size<0>(y_pred); batch_idx += stride) {
                float max_v = -FLT_MAX;
                for (int i = 0; i < size<1>(y_pred); ++i) {
                    max_v = cutlass::fast_max(max_v, float(y_pred(batch_idx, i)));
                }

                float sum = 0;
                for (int i = 0; i < size<1>(y_pred); ++i) {
                    sum += cutlass::fast_exp(y_pred(batch_idx, i) - max_v);
                }

                int label = y_true(batch_idx);
                assert(label >= 0 && label < size<1>(y_pred));

                TC log_p =
                    static_cast<TC>(y_pred(batch_idx, label) - max_v - cutlass::fast_log(sum));
                loss(batch_idx) = -log_p;
            }
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        void cross_entropy_with_logits_fwd(
            Tensor<EngineA, LayoutA> y_pred,
            Tensor<EngineB, LayoutB> y_true,
            Tensor<EngineC, LayoutC> loss) {
            static_assert(LayoutA::rank == 2, "y_pred should be a 2D matrix");
            static_assert(LayoutB::rank == 1, "y_true should be a 1D matrix");
            static_assert(LayoutC::rank == 1, "loss should be a 1D matrix");

            assert(size<0>(y_pred) == size<0>(y_true));
            assert(size<0>(y_pred) == size<0>(loss));

            auto [grid_size, block_size] = launch_config(
                cross_entropy_with_logits_fwd_kernel<
                    EngineA,
                    LayoutA,
                    EngineB,
                    LayoutB,
                    EngineC,
                    LayoutC>,
                size<0>(y_pred));
            cross_entropy_with_logits_fwd_kernel<<<grid_size, block_size>>>(y_pred, y_true, loss);
        }

        /**
         * Backward pass for cross entropy with logits.
         *
         * If y_true is a one-hot vector, then the formula is:
         * dy_pred = softmax(y_pred) - y_true
         *
         * @param y_pred: 2D matrix of shape (batch_size, num_classes)
         * @param y_true: 1D matrix of shape (batch_size)
         * @param dy_pred: 2D matrix of shape (batch_size, num_classes)
         */
        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        __global__ void cross_entropy_with_logits_bwd_kernel(
            Tensor<EngineA, LayoutA> y_pred,
            Tensor<EngineB, LayoutB> y_true,
            Tensor<EngineC, LayoutC> dy_pred) {
            /**
             * Backward formula is:
             * dy_pred = (softmax(y_pred) - y_true) / batch_size
             * The / batch_size comes from taking the mean of the loss.
             */
            using TC = typename EngineC::value_type;

            // A batch is handled by one thread
            int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int batch_size = size<0>(y_pred);

            for (; batch_idx < batch_size; batch_idx += stride) {
                float max_v = -FLT_MAX;
                for (int i = 0; i < size<1>(y_pred); ++i) {
                    max_v = cutlass::fast_max(max_v, float(y_pred(batch_idx, i)));
                }

                float sum = 0;
                for (int i = 0; i < size<1>(y_pred); ++i) {
                    sum += cutlass::fast_exp(float(y_pred(batch_idx, i)) - max_v);
                }

                int label = y_true(batch_idx);
                for (int i = 0; i < size<1>(y_pred); ++i) {
                    float p = cutlass::fast_exp(float(y_pred(batch_idx, i)) - max_v) / sum;
                    float grad = i == label ? p - 1.0f : p;
                    dy_pred(batch_idx, i) = TC(grad / float(batch_size));
                }
            }
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        void cross_entropy_with_logits_bwd(
            Tensor<EngineA, LayoutA> y_pred,
            Tensor<EngineB, LayoutB> y_true,
            Tensor<EngineC, LayoutC> dy_pred) {
            static_assert(LayoutA::rank == 2, "y_pred should be a 2D matrix");
            static_assert(LayoutB::rank == 1, "y_true should be a 1D matrix");
            static_assert(LayoutC::rank == 2, "dy_pred should be a 2D matrix");

            assert(y_pred.shape() == dy_pred.shape());
            assert(size<0>(y_pred) == size<0>(y_true));

            auto [grid_size, block_size] = launch_config(
                cross_entropy_with_logits_bwd_kernel<
                    EngineA,
                    LayoutA,
                    EngineB,
                    LayoutB,
                    EngineC,
                    LayoutC>,
                size(y_pred));
            cross_entropy_with_logits_bwd_kernel<<<grid_size, block_size>>>(
                y_pred, y_true, dy_pred);
        }
    }  // namespace op
}  // namespace lib