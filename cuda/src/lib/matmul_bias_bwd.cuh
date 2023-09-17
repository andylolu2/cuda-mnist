#pragma once
#include <cutlass/numeric_types.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/functions.cuh"
#include "lib/gemm_device.cuh"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/op/binary_pointwise.cuh"
#include "lib/op/sum.cuh"
#include "lib/print.h"
#include "lib/tensor_ops.cuh"

using namespace cute;

namespace lib {
    namespace op {
        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC,
            typename EngineD,
            typename LayoutD,
            typename EngineE,
            typename LayoutE,
            typename EngineF,
            typename LayoutF,
            typename EngineG,
            typename LayoutG>
        void matmul_bias_bwd(
            const Tensor<EngineA, LayoutA> &x,  // (M K)
            const Tensor<EngineB, LayoutB> &w,  // (K N)
            // const Tensor<EngineC, LayoutC> &b,   // (N)
            const Tensor<EngineD, LayoutD> &dy,  // (M N)
            Tensor<EngineE, LayoutE> &dx,        // (M K)
            Tensor<EngineF, LayoutF> &dw,        // (K N)
            Tensor<EngineG, LayoutG> &db         // (N)
        ) {
            // dw = x.T @ dy
            Tensor x_T = transpose<0, 1>(x);
            lib::gemm(x_T, dy, dw)();
            // matmul(dy_T, x_T, dw);

            // db = sum(dy, axis=0)
            sum<0>(dy, db);

            // dx = dy @ w.T
            Tensor w_T = transpose<0, 1>(w);
            lib::gemm(dy, w_T, dx)();
            // matmul(dy, w_T, dx);
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC,
            typename EngineD,
            typename LayoutD,
            typename EngineE,
            typename LayoutE,
            typename EngineF,
            typename LayoutF,
            typename EngineG,
            typename LayoutG>
        void relu_matmul_bias_bwd(
            const Tensor<EngineA, LayoutA> &x,
            const Tensor<EngineB, LayoutB> &w,
            const Tensor<EngineC, LayoutC> &b,
            const Tensor<EngineD, LayoutD> &dy,
            Tensor<EngineE, LayoutE> &dx,
            Tensor<EngineF, LayoutF> &dw,
            Tensor<EngineG, LayoutG> &db) {
            matmul_bias_bwd(x, w, b, dy, dx, dw, db);

            // dx (M K) = dx' (M K) * (x (M K) > 0)
            lib::func::dReLU drelu;
            binary_pointwise(dx, x, dx, drelu);
        }
    }  // namespace op
}  // namespace lib
