#pragma once
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include "lib/functions.cuh"
#include "lib/gemm_device.cuh"
#include "lib/op/constant.cuh"
#include "lib/tensor_ops.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace op {
        namespace detail {
            template <
                typename PointwisePrologueOp,
                typename PointwiseEpilogueOp,
                typename ABlockLayout,
                typename EngineA,
                typename LayoutA,
                typename AThreadLayout,
                typename BBlockLayout,
                typename BThreadLayout,
                typename EngineB,
                typename LayoutB,
                typename CBlockLayout,
                typename CThreadLayout,
                typename EngineC,
                typename LayoutC,
                typename EngineD,
                typename LayoutD>
            __global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void matmul_bias_pointwise_device(
                Tensor<EngineA, LayoutA> mA,
                Tensor<EngineB, LayoutB> mB,
                Tensor<EngineC, LayoutC> mC,
                Tensor<EngineD, LayoutD> mD,
                ABlockLayout blockA,
                AThreadLayout tA,
                BBlockLayout blockB,
                BThreadLayout tB,
                CBlockLayout blockC,
                CThreadLayout tC) {
                using X = Underscore;
                using TC = typename EngineC::value_type;

                auto M = size<0>(mA);
                auto N = size<0>(mB);
                auto K = size<1>(mA);
                auto BLK_M = shape<0>(blockC);
                auto BLK_N = shape<1>(blockC);
                auto BLK_K = shape<1>(blockA);

                auto blk_shape = make_shape(BLK_M, BLK_N, BLK_K);
                auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);  // (m n k)

                // Allocate accumulator
                auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1, _1, X>{});  // (BLK_M BLK_N)
                auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
                auto tCrC = make_fragment_like(tCgC);  // (BLK_M/THR_M BLK_N/THR_N)

                // Compute GEMM for the local tile to tCrC
                lib::gemm_device<PointwisePrologueOp, PointwiseEpilogueOp>(
                    mA, mB, mC, tCrC, blockA, tA, blockB, tB, blockC, tC);

                // Local tile for D (for the write-out)
                auto gD = local_tile(mD, blk_shape, blk_coord, Step<_1, _1, X>{});  // (BLK_M BLK_N)
                auto tCgD = local_partition(gD, tC, threadIdx.x, Step<_1, _1>{});

                //
                // Epilogue
                //

                // Compute the "residues"
                auto m_residue = min(BLK_M, M - blockIdx.x * BLK_M);
                auto n_residue = min(BLK_N, N - blockIdx.y * BLK_N);

                // Mask for the write-out
                Tensor cC = make_identity_tensor(gC.shape());
                Tensor tCcC = local_partition(cC, tC, threadIdx.x);

                CUTE_UNROLL
                for (int i = 0; i < size(tCrC); ++i) {
                    if (elem_less(tCcC(i), make_coord(m_residue, n_residue))) {
                        tCgD(i) = tCrC(i);
                    }
                }
            }

            template <
                typename PointwisePrologueOp,
                typename PointwiseEpilogueOp,
                typename EngineA,
                typename LayoutA,
                typename EngineB,
                typename LayoutB,
                typename EngineC,
                typename LayoutC,
                typename EngineD,
                typename LayoutD>
            void matmul_bias_pointwise(
                const Tensor<EngineA, LayoutA> &x,
                const Tensor<EngineB, LayoutB> &w,
                const Tensor<EngineC, LayoutC> &b,
                Tensor<EngineD, LayoutD> &y) {
                static_assert(LayoutA::rank == 2, "x must be a matrix");
                static_assert(LayoutB::rank == 2, "w must be a matrix");
                static_assert(LayoutC::rank == 1, "b must be a vector");
                static_assert(LayoutD::rank == 2, "y must be a matrix");

                auto M = size<0>(x);
                auto N = size<0>(w);
                auto K = size<1>(x);

                assert(x.shape() == make_shape(M, K));
                assert(w.shape() == make_shape(N, K));
                assert(b.shape() == make_shape(N));
                assert(y.shape() == make_shape(M, N));

                Tensor b_expanded = lib::op::expand<0>(b, M);  // (N) -> (M N)

                // Define block sizes (static)
                auto bM = Int<128>{};
                auto bN = Int<128>{};
                auto bK = Int<8>{};

                // Define the block layouts (static)
                auto sA = make_layout(make_shape(bM, bK));  // split (M K) into (bM bK)
                auto sB = make_layout(make_shape(bN, bK));  // split (N K) into (bN bK)
                auto sC = make_layout(make_shape(bM, bN));  // split (M N) into (bM bN)

                // Define the thread layouts (static)
                auto tA =
                    make_layout(make_shape(Int<32>{}, Int<8>{}));  // partitioning (bM bK) for copy
                auto tB =
                    make_layout(make_shape(Int<32>{}, Int<8>{}));  // partitioning (bN bK) for copy
                auto tC = make_layout(
                    make_shape(Int<16>{}, Int<16>{}));  // partitioning (bM bN) for compute

                dim3 dimBlock(size(tC));
                dim3 dimGrid(ceil_div(M, bM), ceil_div(N, bN));

                matmul_bias_pointwise_device<PointwisePrologueOp, PointwiseEpilogueOp>
                    <<<dimGrid, dimBlock>>>(x, w, b_expanded, y, sA, tA, sB, tB, sC, tC);
            }

            template <
                typename PointwisePrologueOp,
                typename PointwiseEpilogueOp,
                typename EngineA,
                typename LayoutA,
                typename EngineB,
                typename LayoutB,
                typename EngineC,
                typename LayoutC>
            void matmul_pointwise(
                const Tensor<EngineA, LayoutA> &x,
                const Tensor<EngineB, LayoutB> &w,
                Tensor<EngineC, LayoutC> &y) {
                using TC = typename EngineC::value_type;

                DeviceAllocation<TC> b_data(1);
                Tensor b = make_tensor(
                    make_gmem_ptr(b_data.get()), make_shape(size<1>(y)), make_stride(0));
                lib::op::constant(b);

                matmul_bias_pointwise<PointwisePrologueOp, PointwiseEpilogueOp>(x, w, b, y);
            }
        }  // namespace detail

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC>
        void matmul(
            const Tensor<EngineA, LayoutA> &x,
            const Tensor<EngineB, LayoutB> &w,
            Tensor<EngineC, LayoutC> &y) {
            detail::matmul_pointwise<lib::func::Identity, lib::func::Identity>(x, w, y);
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC,
            typename EngineD,
            typename LayoutD>
        void matmul_bias_relu(
            const Tensor<EngineA, LayoutA> &x,
            const Tensor<EngineB, LayoutB> &w,
            const Tensor<EngineC, LayoutC> &b,
            Tensor<EngineD, LayoutD> &y) {
            detail::matmul_bias_pointwise<lib::func::Identity, lib::func::ReLU>(x, w, b, y);
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC,
            typename EngineD,
            typename LayoutD>
        void matmul_bias(
            const Tensor<EngineA, LayoutA> &x,
            const Tensor<EngineB, LayoutB> &w,
            const Tensor<EngineC, LayoutC> &b,
            Tensor<EngineD, LayoutD> &y) {
            detail::matmul_bias_pointwise<lib::func::Identity, lib::func::Identity>(x, w, b, y);
        }

        template <
            typename EngineA,
            typename LayoutA,
            typename EngineB,
            typename LayoutB,
            typename EngineC,
            typename LayoutC,
            typename EngineD,
            typename LayoutD>
        void relu_matmul_bias(
            const Tensor<EngineA, LayoutA> &x,
            const Tensor<EngineB, LayoutB> &w,
            const Tensor<EngineC, LayoutC> &b,
            Tensor<EngineD, LayoutD> &y) {
            detail::matmul_bias_pointwise<lib::func::ReLU, lib::func::Identity>(x, w, b, y);
        }
    }  // namespace op
}  // namespace lib
