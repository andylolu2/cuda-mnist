#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/matmul_bias_bwd.cuh"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/op/add.cuh"
#include "lib/op/constant.cuh"
#include "lib/op/normal.cuh"
#include "lib/op/sgd.cuh"
#include "lib/tensor_ops.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace module {
        template <typename ParamType, typename GradType>
        class Linear {
            using WShape = Shape<int, int>;
            using BShape = Shape<int>;
            using WTensor = Tensor<ViewEngine<gmem_ptr<ParamType>>, Layout<WShape>>;
            using BTensor = Tensor<ViewEngine<gmem_ptr<ParamType>>, Layout<BShape>>;
            using DwTensor = Tensor<ViewEngine<gmem_ptr<GradType>>, Layout<WShape>>;
            using DbTensor = Tensor<ViewEngine<gmem_ptr<GradType>>, Layout<BShape>>;

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
            bool relu;

           public:
            Linear(int in_features, int out_features, bool relu = false)
                : in_features(in_features),
                  out_features(out_features),
                  w_data(in_features * out_features),
                  b_data(out_features),
                  dw_data(in_features * out_features),
                  db_data(out_features),
                  relu(relu),
                  w(make_tensor(
                      make_gmem_ptr(w_data.get()), make_shape(out_features, in_features))),
                  b(make_tensor(make_gmem_ptr(b_data.get()), make_shape(out_features))),
                  dw(make_tensor(
                      make_gmem_ptr(dw_data.get()), make_shape(out_features, in_features))),
                  db(make_tensor(make_gmem_ptr(db_data.get()), make_shape(out_features))) {}
            ~Linear() {}

            auto weight() { return w; }

            auto bias() { return b; }

            auto weight_grad() { return dw; }

            auto bias_grad() { return db; }

            void init() {
                lib::op::normal(w, ParamType(0.0), ParamType(0.01));
                lib::op::normal(b, ParamType(0.0), ParamType(0.01));
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
                Tensor<EngineX, LayoutX>& x,
                Tensor<EngineY, LayoutY>& dy,
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
        };

        template <typename ParamType_, typename GradType_>
        std::ostream& operator<<(std::ostream& os, const Linear<ParamType_, GradType_>& linear) {
            os << "Linear(" << linear.in_features << ", " << linear.out_features << ")";
            return os;
        }
    }  // namespace module
}  // namespace lib