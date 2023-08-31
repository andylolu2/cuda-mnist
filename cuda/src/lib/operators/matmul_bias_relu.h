#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include "lib/3d_layout.h"

namespace lib {
    namespace ops {
        using Layout = cutlass::layout::RowMajor;
        using Element = cutlass::half_t;

        /* Computes y = ReLU(x @ w + b) */

        void batched_matmul_bias_relu(
            cutlass::HostTensor<Element, Layout>& x,                               // (B D1)
            cutlass::HostTensor<Element, Layout>& w,                               // (D2 D1)
            cutlass::HostTensor<Element, cutlass::layout::PackedVectorLayout>& b,  // (D2)
            cutlass::HostTensor<Element, Layout>& y                                // (B D2)
        );
    }  // namespace ops
}  // namespace lib