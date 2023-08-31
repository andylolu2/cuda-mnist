#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>

#include "lib/3d_layout.h"
#include "lib/operators/matmul_bias_relu.h"

#define check_cutlass(status)                                                   \
    {                                                                           \
        cutlass::Status error = status;                                         \
        if (error != cutlass::Status::kSuccess) {                               \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) \
                      << " at: " << __LINE__ << std::endl;                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

namespace lib {
    namespace ops {
        using Layout = cutlass::layout::RowMajor;

        void batched_matmul_bias_relu(
            cutlass::HostTensor<Element, Layout>& x,                               // (B D1)
            cutlass::HostTensor<Element, Layout>& w,                               // (D2 D1)
            cutlass::HostTensor<Element, cutlass::layout::PackedVectorLayout>& b,  // (D2)
            cutlass::HostTensor<Element, Layout>& y                                // (B D2)
        ) {
            using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
                Element,                                     // data type of output matrix
                128 / cutlass::sizeof_bits<Element>::value,  // num elements per memory access
                Element,                                     // data type of accumulator
                Element,                                     // data type of linear combination
                cutlass::epilogue::thread::ScaleType::NoBetaScaling  // no beta
                >;
            using Gemm = cutlass::gemm::device::Gemm<
                Element,                                 // Data type of A matrix
                Layout,                                  // Layout of batch items of A matrix
                Element,                                 // Data type of B matrix
                Layout,                                  // Layout of batch items of B matrix
                Element,                                 // Data type of C matrix
                Layout,                                  // Layout of batch items of C matrix
                Element,                                 // Data type of internal accumulation
                cutlass::arch::OpClassTensorOp,          // Use Tensor Cores
                cutlass::arch::Sm75,                     // Turing arch
                cutlass::gemm::GemmShape<128, 256, 32>,  // threadblock tile (M N K)
                cutlass::gemm::GemmShape<64, 64, 32>,    // warp tile (M N K)
                cutlass::gemm::GemmShape<16, 8, 8>,      // MMA Op tile (M N K)
                EpilogueOp                               // Add bias and apply ReLU
                >;
            Gemm gemm_op;

            check_cutlass(gemm_op({
                {
                    x.extent().row(),     // B
                    w.extent().row(),     // D2
                    x.extent().column(),  // D1
                },
                x.device_ref(),
                w.device_ref(),
                {b.device_data(), 0},  // use stride of 0 to broadcast bias
                y.device_ref(),
                {
                    Element(1.0),  // alpha
                    Element(1.0),  // beta
                },
            }));
        };
    }  // namespace ops
}  // namespace lib