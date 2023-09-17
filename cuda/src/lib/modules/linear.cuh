#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/gemm_device.cuh"
#include "lib/matmul_bias_bwd.cuh"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/op/add.cuh"
#include "lib/op/constant.cuh"
#include "lib/op/sgd.cuh"
#include "lib/op/unary_pointwise.cuh"
#include "lib/tensor_ops.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace module {
        template <
            typename ParamType,
            typename GradType,
            int AccessGranularityBits,
            typename EngineX,
            typename ShapeX,
            typename StrideX,
            typename EngineY,
            typename ShapeY,
            typename StrideY>
        class Linear {
            using ShapeW = Shape<int, int>;
            using StrideW = Stride<_1, int>;
            using ShapeB = Shape<int>;
            using StrideB = Stride<_1>;
            using ParamEngine = ViewEngine<gmem_ptr<ParamType>>;
            using GradEngine = ViewEngine<gmem_ptr<GradType>>;
            using WTensor = Tensor<ParamEngine, Layout<ShapeW, StrideW>>;
            using BTensor = Tensor<ParamEngine, Layout<ShapeB, StrideB>>;
            using DwTensor = Tensor<GradEngine, Layout<ShapeW, StrideW>>;
            using DbTensor = Tensor<GradEngine, Layout<ShapeB, StrideB>>;

           private:
            int in_features;
            int out_features;
            DeviceAllocation<ParamType> w_data;
            DeviceAllocation<ParamType> b_data;
            DeviceAllocation<GradType> dw_data;
            DeviceAllocation<GradType> db_data;
            WTensor w;
            BTensor b;
            DwTensor dw;
            DbTensor db;
            lib::GemmOperation<
                AccessGranularityBits,
                EngineX,
                ShapeX,
                StrideX,
                ParamEngine,
                ShapeW,
                StrideW,
                ParamEngine,
                ShapeW,
                StrideW,
                EngineY,
                ShapeY,
                StrideY>
                gemm_op;

           public:
            Linear(
                Tensor<EngineX, Layout<ShapeX, StrideX>>& x,
                Tensor<EngineY, Layout<ShapeY, StrideY>>& y) {
                in_features = size<1>(x);
                out_features = size<1>(y);
                w_data.reset(in_features * out_features);
                b_data.reset(out_features);
                dw_data.reset(in_features * out_features);
                db_data.reset(out_features);
                w = make_tensor(make_gmem_ptr(w_data.get()), make_shape(out_features, in_features));
                b = make_tensor(make_gmem_ptr(b_data.get()), make_shape(out_features));
                dw = make_tensor(
                    make_gmem_ptr(dw_data.get()), make_shape(out_features, in_features));
                db = make_tensor(make_gmem_ptr(db_data.get()), make_shape(out_features));
                gemm_op = make_gemm_op(x, w, b, y);
            }

            // Move constructor
            Linear(Linear&& other)
                : in_features(other.in_features),
                  out_features(other.out_features),
                  w_data(std::move(other.w_data)),
                  b_data(std::move(other.b_data)),
                  dw_data(std::move(other.dw_data)),
                  db_data(std::move(other.db_data)),
                  gemm_op(std::move(other.gemm_op)) {
                w = make_tensor(make_gmem_ptr(w_data.get()), make_shape(out_features, in_features));
                b = make_tensor(make_gmem_ptr(b_data.get()), make_shape(out_features));
                dw = make_tensor(
                    make_gmem_ptr(dw_data.get()), make_shape(out_features, in_features));
                db = make_tensor(make_gmem_ptr(db_data.get()), make_shape(out_features));
            }

            ~Linear() = default;

            auto weight() { return w; }

            auto bias() { return b; }

            auto weight_grad() { return dw; }

            auto bias_grad() { return db; }

            void init() {
                float fan_in = static_cast<float>(in_features);
                {
                    // Kaiming uniform
                    float gain = std::sqrt(2.0f);
                    float std = gain / std::sqrt(fan_in);
                    float upper = std::sqrt(3.0f) * std;
                    float lower = -upper;
                    lib::op::uniform(w, w, lower, upper);
                }
                {
                    float upper = std::sqrt(1.0f / fan_in);
                    float lower = -upper;
                    lib::op::uniform(b, b, lower, upper);
                }
            }

            template <typename EngineX, typename LayoutX, typename EngineY, typename LayoutY>
            void forward(Tensor<EngineX, LayoutX>& x, Tensor<EngineY, LayoutY>& y) {
                if (relu) {
                    lib::op::relu_matmul_bias(x, w, b, y);
                } else {
                    lib::op::matmul_bias(x, w, b, y);
                }
            }

            template <
                typename EngineX,
                typename LayoutX,
                typename EngineY,
                typename LayoutY,
                typename EngineDx,
                typename LayoutDx>
            void backward(
                const Tensor<EngineX, LayoutX>& x,
                const Tensor<EngineY, LayoutY>& dy,
                Tensor<EngineDx, LayoutDx>& dx) {
                if (relu) {
                    lib::op::relu_matmul_bias_bwd(x, w, b, dy, dx, dw, db);
                } else {
                    lib::op::matmul_bias_bwd(x, w, b, dy, dx, dw, db);
                }
            }

            void update(GradType lr) {
                lib::op::sgd(w, dw, lr);
                lib::op::sgd(b, db, lr);
            }

            void clear_grad() {
                lib::op::constant(dw);
                lib::op::constant(db);
            }

            template <typename ParamType_, typename GradType_>
            friend std::ostream& operator<<(
                std::ostream& os, const Linear<ParamType_, GradType_>& linear);
        };  // namespace module

        template <typename ParamType_, typename GradType_>
        std::ostream& operator<<(std::ostream& os, const Linear<ParamType_, GradType_>& linear) {
            os << "Linear(" << linear.in_features << ", " << linear.out_features << ")";
            return os;
        }
    }  // namespace module
}  // namespace lib