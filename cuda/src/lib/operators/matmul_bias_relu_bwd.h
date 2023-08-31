#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_drelu.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include "lib/3d_layout.h"
#include "lib/operators/matmul_bias_relu.h"

namespace lib {
    namespace ops {
        using Layout = cutlass::layout::RowMajor;

        void batched_matmul_bias_relu_bwd(
            // Input tensors
            cutlass::HostTensor<Element, Layout>& x,                             // (B D1)
            cutlass::HostTensor<Element, Layout>& w,                             // (D2 D1)
            cutlass::HostTensor<float, cutlass::layout::PackedVectorLayout>& b,  // (D2)
            cutlass::HostTensor<Element, Layout>& dy,                            // (B D2)
            // Output tensors
            cutlass::HostTensor<Element, cutlass::layout::PackedVectorLayout>& db,  // (D2)
            cutlass::HostTensor<Element, Layout>& dw,                               // (D2 D1)
            cutlass::HostTensor<Element, Layout>& dx,                               // (B D1)
            // Intermediate tensors
            cutlass::HostTensor<float, Layout>& d_after_bias  // (B D2)
        );
    }  // namespace ops
};     // namespace lib