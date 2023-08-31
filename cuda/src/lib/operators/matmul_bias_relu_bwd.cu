#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination_drelu.h>
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

        /*
         * Formula: (http://cs231n.stanford.edu/handouts/linear-backprop.pdf)
         * 1. d_after_bias (B D2) = reluBwd(x (B D1) @ W (D1 D2) + b (D2), dy (B D2)))
         * 2. db (D2) = sum(d_after_bias (B D2), dims=0)
         * 3. dw (D1 D2) = X^T (D1 B) @ d_after_bias (B D2)
         * 4. dx (B D1) = d_after_bias (B D2) @ W^T (D2 D1)
         */
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
        ) {
            // Step 1: d_after_bias (B D2) = reluBwd(x (B D1) @ W (D1 D2) + b (D2), dy (B D2)))
            {
                using EpilogueOp = cutlass::epilogue::thread::LinearCombinationDRelu<
                    float,                                    // data type of output
                    float,                                    // data type of accumulator
                    float,                                    // data type of source tensor
                    float,                                    // data type of addition tensor
                    128 / cutlass::sizeof_bits<float>::value  // num elements per memory access
                    >;

                using Gemm = cutlass::gemm::device::Gemm<
                    Element,                                 // Data type of A matrix
                    Layout,                                  // Layout of batch items of A matrix
                    Element,                                 // Data type of B matrix
                    Layout,                                  // Layout of batch items of B matrix
                    float,                                   // Data type of C matrix
                    Layout,                                  // Layout of batch items of C matrix
                    float,                                   // Data type of internal accumulation
                    cutlass::arch::OpClassTensorOp,          // Use Tensor Cores
                    cutlass::arch::Sm75,                     // Turing arch
                    cutlass::gemm::GemmShape<128, 128, 32>,  // threadblock tile (M N K)
                    cutlass::gemm::GemmShape<64, 64, 32>,    // warp tile (M N K)
                    cutlass::gemm::GemmShape<16, 8, 8>,      // MMA Op tile (M N K)
                    EpilogueOp,                              // Add bias and apply ReLU
                    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8> >;
                Gemm gemm_op;

                gemm_op({
                    {
                        x.extent().row(),     // B
                        w.extent().row(),     // D2
                        x.extent().column(),  // D1
                    },
                    x.device_ref(),
                    w.device_ref(),
                    {b.device_data(), 0},  // use stride of 0 to broadcast bias
                    d_after_bias.device_ref(),
                    // {
                    //     1.0,  // alpha
                    //     1.0   // beta
                    // },
                });
            }
            // Step 2: db (D2) = sum(d_after_bias (B D2), dims=0)
            // TODO!

            // Step 3: dw (D1 D2) = X^T (D1 B) @ d_after_bias (B D2)
            // {
            //     using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            //         Element,                                     // data type of output matrix
            //         128 / cutlass::sizeof_bits<Element>::value,  // num elements per memory
            //         access Element,                                     // data type of
            //         accumulator Element,                                     // data type of
            //         linear combination cutlass::epilogue::thread::ScaleType::Nothing>;
            //     using Gemm = cutlass::gemm::device::Gemm<
            //         Element,                                         // Data type of A matrix
            //         cutlass::layout::LayoutTranspose<Layout>::type,  // Layout of batch items of
            //         A
            //                                                          // matrix
            //         Element,                                         // Data type of B matrix
            //         Layout,                                  // Layout of batch items of B matrix
            //         Element,                                 // Data type of C matrix
            //         Layout,                                  // Layout of batch items of C matrix
            //         Element,                                 // Data type of internal
            //         accumulation cutlass::arch::OpClassTensorOp,          // Use Tensor Cores
            //         cutlass::arch::Sm75,                     // Turing arch
            //         cutlass::gemm::GemmShape<128, 256, 32>,  // threadblock tile (M N K)
            //         cutlass::gemm::GemmShape<64, 64, 32>,    // warp tile (M N K)
            //         cutlass::gemm::GemmShape<16, 8, 8>,      // MMA Op tile (M N K)
            //         EpilogueOp                               // Add bias and apply ReLU
            //         >;
            //     Gemm gemm_op;

            //     gemm_op({
            //         {
            //             x.extent().column(),             // D1
            //             d_after_bias.extent().column(),  // D2
            //             x.extent().row(),                // B
            //         },
            //         {x.device_data(), x.stride(0)},  // transpose x
            //         d_after_bias.device_ref(),
            //         dw.device_ref(),  // Dummy input for addition term, ignored since beta = 0
            //         dw.device_ref(),
            //         {
            //             Element(1.0),  // alpha
            //             Element(0.0),  // beta
            //         },
            //     });
            // }
        }
    }  // namespace ops
};     // namespace lib