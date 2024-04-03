#pragma once

#include <cutlass/fast_math.h>

#include <cute/tensor.hpp>

#include "lib/op/launch_config.cuh"

using namespace cute;

namespace lib {
    namespace op {
        template <typename TensorA, typename TensorB, typename TensorC>
        __global__ void cross_entropy_with_logits_fwd_kernel(
            TensorA y_pred, TensorB y_true, TensorC loss) {
            using TC = typename TensorC::value_type;

            // Each batch is handled by one thread
            int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;

            for (; batch_idx < size<0>(y_pred); batch_idx += stride) {
                // max_v = max(y_pred[batch_idx, :])
                float max_v = std::numeric_limits<float>::lowest();
                for (int i = 0; i < size<1>(y_pred); ++i) {
                    max_v = cutlass::fast_max(max_v, float(y_pred(batch_idx, i)));
                }

                // sum = sum(exp(y_pred[batch_idx, :] - max_v))
                float sum = 0;
                for (int i = 0; i < size<1>(y_pred); ++i) {
                    sum += cutlass::fast_exp(y_pred(batch_idx, i) - max_v);
                }

                // p = softmax(y_pred[batch_idx, :]) = exp(y_pred[batch_idx, :] - max_v) / sum
                int label = static_cast<int>(y_true(batch_idx));
                TC log_p =
                    static_cast<TC>(y_pred(batch_idx, label) - max_v - cutlass::fast_log(sum));
                loss(batch_idx) = -log_p;
            }
        }

        template <typename TensorA, typename TensorB, typename TensorC>
        void cross_entropy_with_logits_fwd(TensorA y_pred, TensorB y_true, TensorC loss) {
            static_assert(TensorA::rank == 2, "y_pred should be a 2D matrix");
            static_assert(TensorB::rank == 1, "y_true should be a 1D matrix");
            static_assert(TensorC::rank == 1, "loss should be a 1D matrix");

            assert(size<0>(y_pred) == size<0>(y_true));
            assert(size<0>(y_pred) == size<0>(loss));

            auto [grid_size, block_size] = launch_config(size<0>(y_pred));
            cross_entropy_with_logits_fwd_kernel<<<grid_size, block_size>>>(y_pred, y_true, loss);
        }

        template <typename TensorA, typename TensorB, typename TensorC>
        __global__ void cross_entropy_with_logits_bwd_kernel(
            TensorA y_pred, TensorB y_true, TensorC dy_pred) {
            /**
             * Backward formula is:
             * dy_pred = (softmax(y_pred) - y_true) / batch_size
             * The "/ batch_size" comes from taking the mean of the loss.
             */
            using TC = typename TensorC::value_type;

            // Each batch is handled by one thread
            int batch_idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            int batch_size = size<0>(y_pred);

            for (; batch_idx < batch_size; batch_idx += stride) {
                float max_v = std::numeric_limits<float>::lowest();
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
                    dy_pred(batch_idx, i) = static_cast<TC>(grad / float(batch_size));
                }
            }
        }

        template <typename TensorA, typename TensorB, typename TensorC>
        void cross_entropy_with_logits_bwd(TensorA y_pred, TensorB y_true, TensorC dy_pred) {
            static_assert(TensorA::rank == 2, "y_pred should be a 2D matrix");
            static_assert(TensorB::rank == 1, "y_true should be a 1D matrix");
            static_assert(TensorC::rank == 2, "dy_pred should be a 2D matrix");

            assert(y_pred.shape() == dy_pred.shape());
            assert(size<0>(y_pred) == size<0>(y_true));

            auto [grid_size, block_size] = launch_config(size(y_pred));
            cross_entropy_with_logits_bwd_kernel<<<grid_size, block_size>>>(
                y_pred, y_true, dy_pred);
        }
    }  // namespace op
}  // namespace lib