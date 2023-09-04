#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/matmul_bias_bwd.cuh"
#include "lib/matmul_bias_pointwise.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace module {
        template <typename WType, typename BType, typename GradType>
        class Linear {
           private:
            int in_features;
            int out_features;
            DeviceAllocation<WType> w_data;
            DeviceAllocation<BType> b_data;
            DeviceAllocation<GradType> dw_data;
            DeviceAllocation<GradType> db_data;
            Tensor<ViewEngine<gmem_ptr<WType>>, Layout<Shape<int, int>>> w;
            Tensor<ViewEngine<gmem_ptr<BType>>, Layout<Shape<int>>> b;
            Tensor<ViewEngine<gmem_ptr<GradType>>, Layout<Shape<int, int>>> dw;
            Tensor<ViewEngine<gmem_ptr<GradType>>, Layout<Shape<int>>> db;

            auto b_expanded(int batch_size) {
                return make_tensor(
                    make_gmem_ptr(b_data.get()),
                    make_shape(batch_size, out_features),
                    make_stride(0, 1));
            }

            auto db_expanded(int batch_size) {
                return make_tensor(
                    make_gmem_ptr(db_data.get()),
                    make_shape(batch_size, out_features),
                    make_stride(0, 1));
            }

           public:
            Linear(int in_features, int out_features)
                : in_features(in_features),
                  out_features(out_features),
                  w_data(in_features * out_features),
                  b_data(out_features),
                  dw_data(in_features * out_features),
                  db_data(out_features),
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

            auto biad_grad() { return db; }

            void init() {
                lib::init::arange<<<1, 64>>>(w, 0.0f, 0.05f);
                lib::init::arange<<<1, 64>>>(b);
            }

            template <typename EngineX, typename LayoutX, typename EngineY, typename LayoutY>
            void forward(Tensor<EngineX, LayoutX>& x, Tensor<EngineY, LayoutY>& y) {
                int batch_size = size<0>(x);
                auto b_expand = b_expanded(batch_size);
                lib::op::matmul_bias(x, w, b_expand, y);
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
                int batch_size = size<0>(x);
                auto b_expand = b_expanded(batch_size);
                auto db_expand = db_expanded(batch_size);
                lib::op::matmul_bias_bwd(x, w, b_expand, dy, dx, dw, db_expand);
            }

            template <typename WType_, typename BType_, typename GradType_>
            friend std::ostream& operator<<(
                std::ostream& os, const Linear<WType_, BType_, GradType_>& linear);
        };

        template <typename WType_, typename BType_, typename GradType_>
        std::ostream& operator<<(
            std::ostream& os, const Linear<WType_, BType_, GradType_>& linear) {
            os << "Linear(" << linear.in_features << ", " << linear.out_features << ")";
            return os;
        }
    }  // namespace module
}  // namespace lib