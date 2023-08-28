#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include "lib/3d_layout.h"

namespace lib {
    namespace ops {
        using BatchedLayout = cutlass::layout::BatchedRowMajor;
        using Layout = BatchedLayout::SubLayout;
        using Element = cutlass::half_t;

        /* Computes y = ReLU(x @ w + b) */
        void batched_matmul_bias_relu(
            cutlass::HostTensor<Element, BatchedLayout>& x,
            cutlass::HostTensor<Element, BatchedLayout>& w,
            cutlass::HostTensor<Element, BatchedLayout>& b,
            cutlass::HostTensor<Element, BatchedLayout>& y
        );
    }  // namespace ops
}  // namespace lib