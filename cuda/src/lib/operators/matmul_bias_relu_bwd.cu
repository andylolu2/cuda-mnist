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
        void batched_matmul_bias_relu(
            cutlass::HostTensor<Element, BatchedLayout>& tensor_a,
            cutlass::HostTensor<Element, BatchedLayout>& tensor_b,
            cutlass::HostTensor<Element, BatchedLayout>& tensor_c,
            cutlass::HostTensor<Element, BatchedLayout>& tensor_d
        ) {
            using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
                Element,                                     // data type of output matrix
                128 / cutlass::sizeof_bits<Element>::value,  // num elements per memory access
                Element, Element>;
            using Gemm = cutlass::gemm::device::GemmBatched<
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
                EpilogueOp,                              // Add bias and apply ReLU
                cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
                2  // Number of stages in the pipelined mainloop
                >;
            Gemm gemm_op;

            check_cutlass(gemm_op({
                {
                    tensor_a.extent().row(),     // M
                    tensor_b.extent().row(),     // N
                    tensor_a.extent().column(),  // K
                },
                {tensor_a.device_data(), tensor_a.layout().stride_row()},
                tensor_a.layout().stride_batch(),
                {tensor_b.device_data(), tensor_b.layout().stride_row()},
                tensor_b.layout().stride_batch(),
                {tensor_c.device_data(), tensor_c.layout().stride_row()},
                tensor_c.layout().stride_batch(),
                {tensor_d.device_data(), tensor_d.layout().stride_row()},
                tensor_d.layout().stride_batch(),
                {
                    Element(1.0),  // alpha
                    Element(1.0),  // beta
                    Element(0.0)   // ReLU threshold
                },
                tensor_a.extent().batch()  // num batches
            }));
        };
    }  // namespace ops
}  // namespace lib