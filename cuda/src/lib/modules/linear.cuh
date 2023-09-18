#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/gemm_device.cuh"
#include "lib/matmul_bias_bwd.cuh"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/op/add.cuh"
#include "lib/op/arange.cuh"
#include "lib/op/constant.cuh"
#include "lib/op/lion.cuh"
#include "lib/op/sgd.cuh"
#include "lib/op/unary_pointwise.cuh"
#include "lib/tensor_ops.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace module {
        class Linear {
            using ParamType = half_t;
            using BaseParamType = float;
            using GradType = half_t;

            using ShapeW = Shape<int, int>;
            using StrideW = Stride<_1, int>;
            using ShapeB = Shape<int>;
            using StrideB = Stride<_1>;
            using ParamEngine = ViewEngine<gmem_ptr<ParamType>>;
            using BaseParamEngine = ViewEngine<gmem_ptr<BaseParamType>>;
            using GradEngine = ViewEngine<gmem_ptr<GradType>>;
            using WTensor = Tensor<ParamEngine, Layout<ShapeW, StrideW>>;
            using BaseWTensor = Tensor<BaseParamEngine, Layout<ShapeW, StrideW>>;
            // using BTensor = Tensor<ParamEngine, Layout<ShapeB, StrideB>>;
            using BaseBTensor = Tensor<BaseParamEngine, Layout<ShapeB, StrideB>>;
            using DWTensor = Tensor<GradEngine, Layout<ShapeW, StrideW>>;
            using DBTensor = Tensor<GradEngine, Layout<ShapeB, StrideB>>;

           private:
            int batch_size;
            int in_features;
            int out_features;
            DeviceAllocation<ParamType> w_data_half;
            DeviceAllocation<BaseParamType> w_data_full;
            DeviceAllocation<BaseParamType> b_data_full;
            DeviceAllocation<ParamType> b_data_broadcasted_half;
            DeviceAllocation<GradType> dw_data;
            DeviceAllocation<GradType> db_data;
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
                    lib::op::uniform(w_full, w_full, lower, upper);
                    lib::op::uniform(b_full, b_full, lower, upper);
                } else if (mode == "arange") {
                    lib::op::arange(w_full, 0.0f, 1.0f / (in_features * out_features));
                    lib::op::arange(b_full, 0.0f, 1.0f / out_features);
                } else {
                    throw std::runtime_error("Unknown init mode");
                }
            }

            template <typename EngineX, typename LayoutX, typename EngineY, typename LayoutY>
            void forward(Tensor<EngineX, LayoutX>& x, Tensor<EngineY, LayoutY>& y) {
                // const int access_size = std::max({
                //     sizeof_bits_v<typename EngineX::value_type>,
                //     sizeof_bits_v<typename EngineY::value_type>,
                //     sizeof_bits_v<ParamType>,
                // });
                lib::op::identity(w_full, w_half);
                lib::op::repeat<0>(b_broadcasted_half, b_full);
                auto gemm_op = lib::gemm<16>(x, w_half, b_broadcasted_half, y);
                CUTLASS_CHECK(gemm_op());
            }

            template <
                typename EngineX,
                typename LayoutX,
                typename EngineY,
                typename LayoutY,
                typename EngineDx,
                typename LayoutDx>
            void backward(
                Tensor<EngineX, LayoutX>& x,
                Tensor<EngineY, LayoutY>& dy,
                Tensor<EngineDx, LayoutDx>& dx) {
                {  // dw = x.T @ dy
                    // const int access_size = std::max(
                    //     {sizeof_bits_v<typename EngineX::value_type>,
                    //      sizeof_bits_v<typename EngineY::value_type>,
                    //      sizeof_bits_v<GradType>});
                    Tensor x_T = lib::op::transpose<0, 1>(x);
                    auto gemm_op_1 = lib::gemm<16>(x_T, dy, dw);
                    CUTLASS_CHECK(gemm_op_1());
                }

                // db = sum(dy, axis=0)
                lib::op::sum<0>(dy, db);

                {  // dx = dy @ w.T
                    // const int access_size = std::max(
                    //     {sizeof_bits_v<typename EngineY::value_type>,
                    //      sizeof_bits_v<ParamType>,
                    //      sizeof_bits_v<GradType>});
                    Tensor w_T = lib::op::transpose<0, 1>(w_half);
                    auto gemm_op_2 = lib::gemm<16>(dy, w_T, dx);
                    CUTLASS_CHECK(gemm_op_2());
                }
            }

            void update(float lr) {
                lib::op::sgd(w_full, dw, lr);
                lib::op::sgd(b_full, db, lr);
            }

            void clear_grad() {
                lib::op::constant(dw);
                lib::op::constant(db);
            }

            friend std::ostream& operator<<(std::ostream& os, const Linear& linear);
        };  // namespace module

        std::ostream& operator<<(std::ostream& os, const Linear& linear) {
            os << "Linear(" << linear.in_features << ", " << linear.out_features << ")";
            return os;
        }
    }  // namespace module
}  // namespace lib