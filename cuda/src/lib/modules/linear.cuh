#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/op/gemm.cuh"
#include "lib/op/pointwise_ops.cuh"
#include "lib/op/reduce_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/op/unreduce_ops.cuh"
#include "lib/utils/device_tensor.cuh"
#include "lib/utils/macros.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace module {
        class Linear {
            using ParamType = half_t;
            using BaseParamType = float;
            using GradType = half_t;

            using ShapeW = Shape<int, int>;
            using ShapeB = Shape<int>;

           private:
            // Requires batch size and in_features to be multiples of 128/16 = 8
            static const int AccessGranularityBits = 128;
            int batch_size;
            int in_features;
            int out_features;

            // We will create two copies of each parameter, one in fp32 ("master" weights) and
            // one in fp16 ("clone" weights). The fp32 copy is used for updating the weights,
            // while the fp16 copy is used for the forward and backward passes.
            DeviceTensor<BaseParamType, Layout<ShapeW>> w_full;
            DeviceTensor<ParamType, Layout<ShapeW>> w_half;
            DeviceTensor<BaseParamType, Layout<ShapeB>> b_full;
            DeviceTensor<ParamType, Layout<ShapeW>> b_broadcasted_half;
            DeviceTensor<GradType, Layout<ShapeW>> dw;
            DeviceTensor<GradType, Layout<ShapeB>> db;
            DeviceAllocation<uint8_t> workspace;

           public:
            Linear(int batch_size_, int in_features_, int out_features_)
                : batch_size(batch_size_),
                  in_features(in_features_),
                  out_features(out_features_),
                  w_full(make_device_tensor<BaseParamType>(make_shape(in_features, out_features))),
                  w_half(make_device_tensor<ParamType>(make_shape(in_features, out_features))),
                  b_full(make_device_tensor<BaseParamType>(make_shape(out_features))),
                  b_broadcasted_half(
                      make_device_tensor<ParamType>(make_shape(batch_size, out_features))),
                  dw(make_device_tensor<GradType>(make_shape(in_features, out_features))),
                  db(make_device_tensor<GradType>(make_shape(out_features))) {}

            // Move constructor
            Linear(Linear&& other)
                : batch_size(other.batch_size),
                  in_features(other.in_features),
                  out_features(other.out_features),
                  w_full(std::move(other.w_full)),
                  w_half(std::move(other.w_half)),
                  b_full(std::move(other.b_full)),
                  b_broadcasted_half(std::move(other.b_broadcasted_half)),
                  dw(std::move(other.dw)),
                  db(std::move(other.db)) {}

            ~Linear() = default;

            auto weight() { return w_full.view(); }

            auto bias() { return b_full.view(); }

            auto weight_grad() { return dw.view(); }

            auto bias_grad() { return db.view(); }

            void init(int seed = 0, std::string mode = "kaiming") {
                if (mode == "kaiming") {  // Kaiming uniform
                    float upper = 1.0f / std::sqrt(in_features);
                    float lower = -upper;
                    lib::op::uniform(w_full.view(), lower, upper, seed);
                    lib::op::uniform(b_full.view(), lower, upper, seed);
                } else if (mode == "arange") {  // Useful for unit testing
                    lib::op::arange(w_full.view(), 0.0f, 1.0f / (in_features * out_features));
                    lib::op::arange(b_full.view(), 0.0f, 1.0f / out_features);
                } else {
                    throw std::runtime_error("Unknown init mode");
                }
            }

            template <typename EngineX, typename LayoutX, typename EngineY, typename LayoutY>
            void forward(Tensor<EngineX, LayoutX> const& x, Tensor<EngineY, LayoutY> const& y) {
                // Make fp16 copy of w and b (and broadcast b)
                lib::op::convert(w_half.view(), w_full.view());
                lib::op::repeat<0>(b_broadcasted_half.view(), b_full.view());

                // y = x @ w + b
                auto gemm_op = lib::op::gemm<AccessGranularityBits>(
                    x, w_half.view(), b_broadcasted_half.view(), y, workspace);
                CUTLASS_CHECK(gemm_op());
            }

            template <typename TensorX, typename TensorDy, typename TensorDx>
            void backward(TensorX const& x, TensorDy const& dy, TensorDx const& dx) {
                // Compute dw and db
                backward(x, dy);

                // dx = dy @ w.T
                auto w_T = lib::op::transpose<0, 1>(w_half.view());
                auto gemm_op = lib::op::gemm<AccessGranularityBits>(dy, w_T, dx, workspace);
                CUTLASS_CHECK(gemm_op());
            }

            template <typename TensorX, typename TensorDy>
            void backward(TensorX const& x, TensorDy const& dy) {
                // dw = x.T @ dy
                auto x_T = lib::op::transpose<0, 1>(x);
                auto gemm_op = lib::op::gemm<AccessGranularityBits>(x_T, dy, dw.view(), workspace);
                CUTLASS_CHECK(gemm_op());

                // db = sum(dy, axis=0)
                lib::op::sum<0>(dy, db.view());
            }

            void update(float lr) {
                lib::op::sgd(w_full.view(), w_full.view(), dw.view(), lr);
                lib::op::sgd(b_full.view(), b_full.view(), db.view(), lr);
            }

            /**
             * Compute the number of TFLOPs for each training step
             */
            auto tflops() {
                size_t b = batch_size;
                size_t m = in_features;
                size_t n = out_features;
                size_t fwd_flops = 2 * b * m * n;
                size_t bwd_flops = 2 * fwd_flops + b * n;
                size_t update_flops = (n + 1) * m;
                return (fwd_flops + bwd_flops + update_flops) / 1e12;
            }
        };
    }  // namespace module
}  // namespace lib