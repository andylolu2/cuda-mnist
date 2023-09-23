#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/op/gemm.cuh"
#include "lib/op/pointwise_ops.cuh"
#include "lib/op/reduce_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/op/unreduce_ops.cuh"
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
            using ParamEngine = ViewEngine<gmem_ptr<ParamType>>;
            using BaseParamEngine = ViewEngine<gmem_ptr<BaseParamType>>;
            using GradEngine = ViewEngine<gmem_ptr<GradType>>;
            using WTensor = Tensor<ParamEngine, Layout<ShapeW>>;
            using BaseWTensor = Tensor<BaseParamEngine, Layout<ShapeW>>;
            using BaseBTensor = Tensor<BaseParamEngine, Layout<ShapeB>>;
            using DWTensor = Tensor<GradEngine, Layout<ShapeW>>;
            using DBTensor = Tensor<GradEngine, Layout<ShapeB>>;

           private:
            static const int AccessGranularityBits = 128;
            int batch_size;
            int in_features;
            int out_features;
            DeviceAllocation<ParamType> w_data_half;
            DeviceAllocation<BaseParamType> w_data_full;
            DeviceAllocation<BaseParamType> b_data_full;
            DeviceAllocation<ParamType> b_data_broadcasted_half;
            DeviceAllocation<GradType> dw_data;
            DeviceAllocation<GradType> db_data;
            DeviceAllocation<uint8_t> workspace;
            WTensor w_half;
            BaseWTensor w_full;
            BaseBTensor b_full;
            WTensor b_broadcasted_half;
            DWTensor dw;
            DBTensor db;

           public:
            Linear(int batch_size_, int in_features_, int out_features_)
                : batch_size(batch_size_),
                  in_features(in_features_),
                  out_features(out_features_),
                  w_data_half(in_features * out_features),
                  w_data_full(in_features * out_features),
                  b_data_full(out_features),
                  b_data_broadcasted_half(batch_size * out_features),
                  dw_data(in_features * out_features),
                  db_data(out_features),
                  w_half(make_tensor(
                      make_gmem_ptr(w_data_half.get()), make_shape(in_features, out_features))),
                  w_full(make_tensor(
                      make_gmem_ptr(w_data_full.get()), make_shape(in_features, out_features))),
                  b_full(make_tensor(make_gmem_ptr(b_data_full.get()), make_shape(out_features))),
                  b_broadcasted_half(make_tensor(
                      make_gmem_ptr(b_data_broadcasted_half.get()),
                      make_shape(batch_size, out_features))),
                  dw(make_tensor(
                      make_gmem_ptr(dw_data.get()), make_shape(in_features, out_features))),
                  db(make_tensor(make_gmem_ptr(db_data.get()), make_shape(out_features))) {}

            // Move constructor
            Linear(Linear&& other)
                : batch_size(other.batch_size),
                  in_features(other.in_features),
                  out_features(other.out_features),
                  w_data_half(std::move(other.w_data_half)),
                  w_data_full(std::move(other.w_data_full)),
                  b_data_full(std::move(other.b_data_full)),
                  b_data_broadcasted_half(std::move(other.b_data_broadcasted_half)),
                  dw_data(std::move(other.dw_data)),
                  db_data(std::move(other.db_data)),
                  w_half(make_tensor(
                      make_gmem_ptr(w_data_half.get()), make_shape(in_features, out_features))),
                  w_full(make_tensor(
                      make_gmem_ptr(w_data_full.get()), make_shape(in_features, out_features))),
                  b_full(make_tensor(make_gmem_ptr(b_data_full.get()), make_shape(out_features))),
                  b_broadcasted_half(make_tensor(
                      make_gmem_ptr(b_data_broadcasted_half.get()),
                      make_shape(batch_size, out_features))),
                  dw(make_tensor(
                      make_gmem_ptr(dw_data.get()), make_shape(in_features, out_features))),
                  db(make_tensor(make_gmem_ptr(db_data.get()), make_shape(out_features))) {}

            ~Linear() = default;

            auto weight() { return w_full; }

            auto bias() { return b_full; }

            auto weight_grad() { return dw; }

            auto bias_grad() { return db; }

            void init(std::string mode = "kaiming") {
                if (mode == "kaiming") {
                    // Kaiming uniform
                    float upper = 1.0f / std::sqrt(in_features);
                    float lower = -upper;
                    lib::op::uniform(w_full, lower, upper);
                    lib::op::uniform(b_full, lower, upper);
                } else if (mode == "arange") {
                    lib::op::arange(w_full, 0.0f, 1.0f / (in_features * out_features));
                    lib::op::arange(b_full, 0.0f, 1.0f / out_features);
                } else {
                    throw std::runtime_error("Unknown init mode");
                }
            }

            template <typename EngineX, typename LayoutX, typename EngineY, typename LayoutY>
            void forward(Tensor<EngineX, LayoutX>& x, Tensor<EngineY, LayoutY>& y) {
                // Make fp16 copy of w and b (and broadcast b)
                lib::op::convert(w_half, w_full);
                lib::op::repeat<0>(b_broadcasted_half, b_full);

                // y = x @ w + b
                auto gemm_op = lib::op::gemm<128>(x, w_half, b_broadcasted_half, y, workspace);
                CUTLASS_CHECK(gemm_op());
            }

            template <typename TensorX, typename TensorDy, typename TensorDx>
            void backward(TensorX& x, TensorDy& dy, TensorDx& dx) {
                // Compute dw and db
                backward(x, dy);

                // dx = dy @ w.T
                Tensor w_T = lib::op::transpose<0, 1>(w_half);
                auto gemm_op = lib::op::gemm<AccessGranularityBits>(dy, w_T, dx, workspace);
                CUTLASS_CHECK(gemm_op());
            }

            template <typename TensorX, typename TensorDy>
            void backward(TensorX& x, TensorDy& dy) {
                // dw = x.T @ dy
                Tensor x_T = lib::op::transpose<0, 1>(x);
                auto gemm_op = lib::op::gemm<AccessGranularityBits>(x_T, dy, dw, workspace);
                CUTLASS_CHECK(gemm_op());

                // db = sum(dy, axis=0)
                lib::op::sum<0>(dy, db);
            }

            void update(float lr) {
                lib::op::sgd(w_full, w_full, dw, lr);
                lib::op::sgd(b_full, b_full, db, lr);
            }

            void clear_grad() {
                lib::op::constant(dw);
                lib::op::constant(db);
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