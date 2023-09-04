#pragma once
#include <cutlass/numeric_types.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/functions.cuh"
#include "lib/gemm_device.cuh"
#include "lib/matmul_bias_pointwise.cuh"
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
            Tensor<EngineA, LayoutA> &x,
            Tensor<EngineB, LayoutB> &w,
            Tensor<EngineC, LayoutC> &b,
            Tensor<EngineD, LayoutD> &dy,
            Tensor<EngineE, LayoutE> &dx,
            Tensor<EngineF, LayoutF> &dw,
            Tensor<EngineG, LayoutG> &db) {
            // dw (N K) = dy.T (N M) @ x.T (K M)
            Tensor dy_T = transpose(dy);
            Tensor x_T = transpose(x);
            matmul_bias(dy_T, x_T, dw, dw);

            // db = dy
            // TODO: cum_copy is not thread-safe
            lib::init::cum_copy<<<1, 1>>>(dy, db);

            // dx (M K) = dy (M N) @ w.T (N K)
            Tensor w_T = transpose(w);
            matmul_bias(dy, w_T, dx, dx);
        }
    }  // namespace op
}  // namespace lib
