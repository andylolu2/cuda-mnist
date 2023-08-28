#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/tensor_fill.h>
#include <cutlass/util/reference/device/tensor_foreach.h>
#include <cutlass/util/tensor_view_io.h>

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/tensor_view.h"
#include "cutlass/util/reference/device/tensor_foreach.h"
#include "lib/3d_layout.h"
#include "lib/operators/matmul_bias_relu.h"

using Tensor3D = cutlass::HostTensor<cutlass::half_t, cutlass::layout::BatchedRowMajor>;

namespace detail {
    template <
        typename Element,  ///< Element type
        typename Layout>   ///< Layout function
    struct TensorMulFunc {
        /// View type
        using TensorView = cutlass::TensorView<Element, Layout>;

        /// Coordinate in tensor's index space
        using TensorCoord = typename TensorView::TensorCoord;

        /// Parameters structure
        struct Params {
            //
            // Data members
            //

            TensorView view;
            Element c;

            //
            // Methods
            //

            Params(TensorView view_ = TensorView(), Element c_ = Element(0)) : view(view_), c(c_) {}
        };

        //
        // Data members
        //
        Params params;

        //
        // Methods
        //
        CUTLASS_DEVICE
        TensorMulFunc(Params const &params) : params(params) {}

        CUTLASS_DEVICE
        void operator()(TensorCoord const &coord) {
            Element const &value = params.view.at(coord);
            params.view.at(coord) = params.c * value;
        }
    };
};  // namespace detail

template <typename Element, typename Layout>
void multiply(cutlass::TensorView<Element, Layout> view, Element c) {
    using Func = detail::TensorMulFunc<Element, Layout>;
    using Params = typename Func::Params;

    cutlass::reference::device::TensorForEach<Func, Layout::kRank, Params>(
        view.extent(), Params(view, c)
    );
};

cudaError_t cutlass_strided_batched_sgemm(
    Tensor3D &A, Tensor3D &B, Tensor3D &C, Tensor3D &D, cutlass::half_t alpha, cutlass::half_t beta
) {
    using Gemm = cutlass::gemm::device::GemmBatched<
        cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t, cutlass::layout::RowMajor>;
    Gemm gemm_op;

    auto batch_stride_A = A.layout().stride_batch();
    auto batch_stride_B = B.layout().stride_batch();
    auto batch_stride_C = C.layout().stride_batch();
    auto batch_stride_D = D.layout().stride_batch();
    auto n_batches = A.extent().at(0);
    auto M = A.extent().at(1);
    auto N = A.extent().at(2);
    auto K = B.extent().at(2);

    cutlass::Status status = gemm_op(
        {{M, N, K},
         //   A.device_ref(),
         {A.device_data(), A.layout().stride_row()},
         batch_stride_A,
         //   B.device_ref(),
         {B.device_data(), B.layout().stride_row()},
         batch_stride_B,
         //   C.device_ref(),
         {C.device_data(), C.layout().stride_row()},
         batch_stride_C,
         //   D.device_ref(),
         {D.device_data(), D.layout().stride_row()},
         batch_stride_D,
         {alpha, beta},
         n_batches}
    );

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

int main() {
    int batch = 8;
    int M = 1;
    int N = 32;
    int K = 8;

    Tensor3D x({batch, M, K});
    Tensor3D y({batch, N, K});
    Tensor3D z({batch, M, N});
    Tensor3D f({batch, M, N});

    cutlass::reference::device::TensorFillRandomUniform(x.device_view(), 42, 1.0_hf, -1.0_hf);
    cutlass::reference::device::TensorFillRandomUniform(y.device_view(), 41, 1.0_hf, -1.0_hf);
    // cutlass::reference::device::TensorFill(y.device_view(), 1.0_hf);
    // cutlass::reference::device::TensorFill(z.device_view(), 1.5_hf);

    // multiply<cutlass::half_t, Tensor3D::Layout>(x.device_view(), -1.0_hf);

    x.sync_host();
    y.sync_host();
    z.sync_host();

    std::cout << "x:\n" << x.host_view() << "\n" << std::endl;
    std::cout << "y:\n" << y.host_view() << "\n" << std::endl;
    std::cout << "z:\n" << z.host_view() << "\n" << std::endl;

    lib::ops::batched_matmul_bias_relu(x, y, z, z);

    z.sync_host();
    std::cout << "z:\n" << z.host_view() << "\n" << std::endl;

    return 0;
}