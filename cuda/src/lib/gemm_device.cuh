#pragma once
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_universal.h>
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cuda/std/type_traits>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_epilogue.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <optional>

#include "lib/functions.cuh"
#include "lib/helper.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    template <
        typename PrologueOp,
        typename EpilogueOp,
        typename EngineA,
        typename LayoutA,
        typename ABlockLayout,
        typename AThreadLayout,
        typename EngineB,
        typename LayoutB,
        typename BBlockLayout,
        typename BThreadLayout,
        typename EngineC,
        typename LayoutC,
        typename CBlockLayout,
        typename CThreadLayout,
        typename EngineD,
        typename LayoutD>
    __device__ void gemm_device(
        Tensor<EngineA, LayoutA> &mA,
        Tensor<EngineB, LayoutB> &mB,
        Tensor<EngineC, LayoutC> &mC,
        Tensor<EngineD, LayoutD> &tCrC,  // Output
        ABlockLayout &blockA,
        AThreadLayout &tA,
        BBlockLayout &blockB,
        BThreadLayout &tB,
        CBlockLayout &blockC,
        CThreadLayout &tC) {
        using X = Underscore;
        using TA = typename EngineA::value_type;
        using TB = typename EngineB::value_type;

        // Preconditions
        CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
        CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
        CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

        CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
        CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
        CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

        // Each thread should handle exactly two copy operations and one compute operation
        CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
        CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

        CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));  // BLK_M
        CUTE_STATIC_ASSERT_V(shape<0>(blockB) == shape<1>(blockC));  // BLK_N
        CUTE_STATIC_ASSERT_V(shape<1>(blockA) == shape<1>(blockB));  // BLK_K

        // --- Global ---
        auto M = size<0>(mA.shape());
        auto N = size<0>(mB.shape());
        auto K = size<1>(mA.shape());

        // --- Thread block level ---
        //  1. copy gA (global memory) -> sA (shared memory)
        //  2. copy gB (global memory) -> sB (shared memory)
        //  3. compute gC += sA @ sB
        // where gA & sA are fragments of mA, gB & sB are fragments of mB, gC is a fragment of mC.

        // Block size of the compute tile
        auto BLK_M = shape<0>(blockC);
        auto BLK_N = shape<1>(blockC);
        auto BLK_K = shape<1>(blockA);

        // Compute the "residues"
        auto m_residue = min(BLK_M, M - blockIdx.x * BLK_M);
        auto n_residue = min(BLK_N, N - blockIdx.y * BLK_N);

        __shared__ TA smemA[cosize_v<ABlockLayout>];
        __shared__ TB smemB[cosize_v<BBlockLayout>];
        auto sA = make_tensor(make_smem_ptr(smemA), blockA);  // (BLK_M BLK_K)
        auto sB = make_tensor(make_smem_ptr(smemB), blockB);  // (BLK_N BLK_K)

        auto blk_shape = make_shape(BLK_M, BLK_N, BLK_K);
        auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);  // (m n k)

        // k := K / BLK_K, the number of blocks in the K-dimension
        auto gA = local_tile(mA, blk_shape, blk_coord, Step<_1, X, _1>{});  // (BLK_M BLK_K k)
        auto gB = local_tile(mB, blk_shape, blk_coord, Step<X, _1, _1>{});  // (BLK_N BLK_K k)
        auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1, _1, X>{});  // (BLK_M BLK_N)

        auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});

        // --- Thread level ---
        // for k_blk_idx in range(k):
        //   1. copy tAgA[:, :, k_blk_idx] (global memory) -> tAsA (shared memory)
        //   2. copy tBgB[:, :, k_blk_idx] (global memory) -> tBsB (shared memory)
        //   3. compute tCrC += tCsA @ tCsB
        //  where
        //   - tAgA is a fragment of gA
        //   - tBgB is a fragment of gB
        //   - tAsA is a fragment of sA (view used for copy)
        //   - tCsA is a fragment of sA (view used for compute)
        //   - tBsB is a fragment of sB (view used for copy)
        //   - tCsB is a fragment of sB (view used for compute)
        //   - tCrC is a fragment of gC (stored in register memory)

        // gA (BLK_M BLK_K k) / tA -> tAgA (BLK_M/tA.M BLK_K/tA.K k)
        auto tAgA = local_partition(gA, tA, threadIdx.x);
        // sA (BLK_M BLK_K) / tA -> tAsA (BLK_M/tA.M BLK_K/tA.K)
        auto tAsA = local_partition(sA, tA, threadIdx.x);
        // predicate mask for the tAgA -> tAsA copy (BLK_M/tA.M BLK_K/tA.K)
        auto tApA = make_tensor<bool>(make_shape(size<0>(tAgA), size<1>(tAgA)));

        // gB (BLK_N BLK_K k) / tB -> tBgB (BLK_N/tB.N BLK_K/tB.K k)
        auto tBgB = local_partition(gB, tB, threadIdx.x);
        // sB (BLK_N BLK_K) / tB -> tBsB (BLK_N/tB.N BLK_K/tB.K)
        auto tBsB = local_partition(sB, tB, threadIdx.x);
        // predicate mask for the tBgB -> tBsB copy (BLK_N/tB.N BLK_K/tB.K)
        auto tBpB = make_tensor<bool>(make_shape(size<0>(tBgB), size<1>(tBgB)));

        // sA (BLK_M BLK_K) by the rows of tC (tC.M tC.N) -> (BLK_M/tC.M BLK_K)
        auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
        // sB (BLK_N BLK_K) by the cols of tC (tC.M tC.N) -> (BLK_N/tC.N BLK_K)
        auto tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});

        // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
        Tensor tAcA = local_partition(cA, tA, threadIdx.x);
        // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));
        Tensor tBcB = local_partition(cB, tB, threadIdx.x);

        // Clear the accumulators
        clear(tCrC);

        auto k = size<2>(tAgA);
        for (int k_blk_idx = 0; k_blk_idx < k; ++k_blk_idx) {
            // Step 1 & 2
            auto k_residue = K - k_blk_idx * BLK_K;
            clear(tAsA);
            clear(tBsB);

            // Populate the predicate masks
            CUTE_UNROLL
            for (int m = 0; m < size<0>(tApA); ++m) {
                CUTE_UNROLL
                for (int k_idx = 0; k_idx < BLK_K; ++k_idx) {
                    tApA(m, k_idx) = elem_less(tAcA(m, k_idx), make_coord(m_residue, k_residue));
                }
            }
            CUTE_UNROLL
            for (int n = 0; n < size<0>(tBpB); ++n) {
                CUTE_UNROLL
                for (int k_idx = 0; k_idx < BLK_K; ++k_idx) {
                    tBpB(n, k_idx) = elem_less(tBcB(n, k_idx), make_coord(n_residue, k_residue));
                }
            }
            copy_if(tApA, tAgA(_, _, k_blk_idx), tAsA);
            copy_if(tBpB, tBgB(_, _, k_blk_idx), tBsB);

            CUTE_UNROLL
            for (int i = 0; i < size(tAsA); ++i) {
                tAsA(i) = PrologueOp{}(tAsA(i));
            }

            cp_async_fence();
            cp_async_wait<0>();

            __syncthreads();

            // Step 3
            cute::gemm(tCsA, tCsB, tCrC);

            __syncthreads();
        }

        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); ++i) {
            auto x = tCrC(i) + tCgC(i);  // A @ B + C
            tCrC(i) = EpilogueOp{}(x);   // Epilogue(A @ B + C)
        }
    }

    using ScaleType = cutlass::epilogue::thread::ScaleType;

    template <
        int AccessGranularityBits,  // Problem size (in bits) needs to be a multiple
                                    // of this number. 128 gives the best performance.
        ScaleType::Kind Scale,      /// Control Alpha and Beta scaling
        typename EngineA,
        typename ShapeA,
        typename StrideA,
        typename EngineB,
        typename ShapeB,
        typename StrideB,
        typename EngineC,
        typename ShapeC,
        typename StrideC,
        typename EngineD,
        typename ShapeD,
        typename StrideD>
    class GemmOperation {
        static_assert(rank_v<ShapeA> == 2, "A must be a matrix");
        static_assert(rank_v<ShapeB> == 2, "B must be a matrix");
        static_assert(rank_v<ShapeC> == 2, "C must be a matrix");
        static_assert(rank_v<ShapeD> == 2, "D must be a matrix");

        using TA = typename EngineA::value_type;
        using TB = typename EngineB::value_type;
        using TC = typename EngineC::value_type;
        using TD = typename EngineD::value_type;
        using LayoutV2A = gemm::detail::StrideToLayoutTagA_t<StrideA>;
        using LayoutV2B = gemm::detail::StrideToLayoutTagA_t<StrideB>;
        using LayoutV2C = gemm::detail::StrideToLayoutTagA_t<StrideC>;
        using LayoutV2D = gemm::detail::StrideToLayoutTagA_t<StrideD>;

        using ElementAccumulator = half_t;                  // data type of accumulator
        using ElementComputeEpilogue = ElementAccumulator;  // data type of epilogue operations
        using MMAOp = arch::OpClassTensorOp;
        using SmArch = arch::Sm75;

        using DefaultConfig =
            gemm::device::DefaultGemmConfiguration<MMAOp, SmArch, TA, TB, TC, ElementAccumulator>;

        using ShapeMMAThreadBlock =
            typename DefaultConfig::ThreadblockShape;                 // threadblock tile MNK
        using ShapeMMAWarp = typename DefaultConfig::WarpShape;       // warp tile MNK
        using ShapeMMAOp = typename DefaultConfig::InstructionShape;  // MMA tile MNK
        using EpilogueOp = epilogue::thread::LinearCombination<
            TD,  // data type of output matrix
            AccessGranularityBits /
                cutlass::sizeof_bits<TC>::value,  // elements per vectorized memory access
            ElementAccumulator,                   // data type of accumulator
            ElementComputeEpilogue,               // the data type of epilogue operation
            Scale                                 // operation to update the destination
            >;
        using ThreadblockSwizzle =
            typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
        static const int Stages = DefaultConfig::kStages;
        static const int AlignmentA = AccessGranularityBits / cutlass::sizeof_bits<TA>::value;
        static const int AlignmentB = AccessGranularityBits / cutlass::sizeof_bits<TB>::value;

        using Gemm = cutlass::gemm::device::GemmUniversal<
            TA,
            LayoutV2A,
            TB,
            LayoutV2B,
            TC,
            LayoutV2C,
            ElementAccumulator,
            MMAOp,
            SmArch,
            ShapeMMAThreadBlock,
            ShapeMMAWarp,
            ShapeMMAOp,
            EpilogueOp,
            ThreadblockSwizzle,
            Stages,
            AlignmentA,
            AlignmentB>;
        using Arguments = typename Gemm::Arguments;

       private:
        DeviceAllocation<uint8_t> workspace;
        Gemm gemm_op;
        Arguments args;

       public:
        GemmOperation(
            Tensor<EngineA, Layout<ShapeA, StrideA>> &mA,
            Tensor<EngineB, Layout<ShapeB, StrideB>> &mB,
            Tensor<EngineC, Layout<ShapeC, StrideC>> &mC,
            Tensor<EngineD, Layout<ShapeD, StrideD>> &mD) {
            int M = size<0>(mA.shape());
            int N = size<1>(mB.shape());
            int K = size<1>(mA.shape());

            auto leading_A =
                LayoutV2A::packed({get<1>(mA.stride()), get<0>(mA.stride())}).stride(0);
            auto leading_B =
                LayoutV2B::packed({get<1>(mB.stride()), get<0>(mB.stride())}).stride(0);
            auto leading_C =
                LayoutV2C::packed({get<1>(mC.stride()), get<0>(mC.stride())}).stride(0);
            auto leading_D =
                LayoutV2D::packed({get<1>(mD.stride()), get<0>(mD.stride())}).stride(0);
            int split_k_slices = (K + 127) / 128;  // Some random heuristic I invented

            args = {
                cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
                {M, N, K},  // problem size (M N K)
                split_k_slices,
                {ElementComputeEpilogue(1),   // alpha
                 ElementComputeEpilogue(1)},  // beta
                mA.data().get(),              // ptr to A (input)
                mB.data().get(),              // ptr to B (input)
                mC.data().get(),              // ptr to C (input)
                mD.data().get(),              // ptr to D (output)
                size(mA),                     // numel(A)
                size(mB),                     // numel(B)
                size(mC),                     // numel(C)
                size(mD),                     // numel(D)
                leading_A,                    // leading dimension of A
                leading_B,                    // leading dimension of B
                leading_C,                    // leading dimension of C
                leading_D                     // leading dimension of D
            };

            CUTLASS_CHECK(gemm_op.can_implement(args));

            size_t workspace_size = Gemm::get_workspace_size(args);
            workspace.reset(workspace_size);

            CUTLASS_CHECK(gemm_op.initialize(args, workspace.get()));
        }

        // Move constructor
        // GemmOperation(GemmOperation &&other)
        //     : workspace(std::move(other.workspace)), args(std::move(other.args)) {
        //     CUTLASS_CHECK(gemm_op.initialize(args, workspace.get()));
        // }

        void operator()() { CUTLASS_CHECK(gemm_op()); }
    };

    template <
        int AccessGranularityBits = 128,
        ScaleType::Kind Scale = ScaleType::Kind::NoBetaScaling,
        typename EngineA,
        typename ShapeA,
        typename StrideA,
        typename EngineB,
        typename ShapeB,
        typename StrideB,
        typename EngineC,
        typename ShapeC,
        typename StrideC,
        typename EngineD,
        typename ShapeD,
        typename StrideD>
    auto make_gemm_op(
        Tensor<EngineA, Layout<ShapeA, StrideA>> &mA,
        Tensor<EngineB, Layout<ShapeB, StrideB>> &mB,
        Tensor<EngineC, Layout<ShapeC, StrideC>> &mC,
        Tensor<EngineD, Layout<ShapeD, StrideD>> &mD) {
        return GemmOperation<
            AccessGranularityBits,
            Scale,
            EngineA,
            ShapeA,
            StrideA,
            EngineB,
            ShapeB,
            StrideB,
            EngineC,
            ShapeC,
            StrideC,
            EngineD,
            ShapeD,
            StrideD>(mA, mB, mC, mD);
    };

    template <
        int AccessGranularityBits = 128,
        typename EngineA,
        typename ShapeA,
        typename StrideA,
        typename EngineB,
        typename ShapeB,
        typename StrideB,
        typename EngineD,
        typename ShapeD,
        typename StrideD>
    auto make_gemm_op(
        Tensor<EngineA, Layout<ShapeA, StrideA>> &mA,
        Tensor<EngineB, Layout<ShapeB, StrideB>> &mB,
        Tensor<EngineD, Layout<ShapeD, StrideD>> &mD) {
        return GemmOperation<
            AccessGranularityBits,
            ScaleType::Kind::OnlyAlphaScaling,
            EngineA,
            ShapeA,
            StrideA,
            EngineB,
            ShapeB,
            StrideB,
            EngineD,
            ShapeD,
            StrideD,
            EngineD,
            ShapeD,
            StrideD>(mA, mB, mD, mD);
    };

    template <
        int AccessGranularityBits = 16,  // Problem size (in bits) needs to be a multiple
                                         // of this number. 128 gives the best performance.
        ScaleType::Kind Scale = ScaleType::Kind::NoBetaScaling,  /// Control Alpha and Beta scaling
        typename EngineA,
        typename ShapeA,
        typename StrideA,
        typename EngineB,
        typename ShapeB,
        typename StrideB,
        typename EngineC,
        typename ShapeC,
        typename StrideC,
        typename EngineD,
        typename ShapeD,
        typename StrideD>
    auto gemm(
        Tensor<EngineA, Layout<ShapeA, StrideA>> &mA,
        Tensor<EngineB, Layout<ShapeB, StrideB>> &mB,
        Tensor<EngineC, Layout<ShapeC, StrideC>> &mC,
        Tensor<EngineD, Layout<ShapeD, StrideD>> &mD,
        std::optional<device_memory::allocation<uint8_t>> workspace_or_null = std::nullopt) {
        // device_memory::allocation<uint8_t> &workspace) {
        static_assert(rank_v<ShapeA> == 2, "A must be a matrix");
        static_assert(rank_v<ShapeB> == 2, "B must be a matrix");
        static_assert(rank_v<ShapeC> == 2, "C must be a matrix");
        static_assert(rank_v<ShapeD> == 2, "D must be a matrix");

        using TA = typename EngineA::value_type;
        using TB = typename EngineB::value_type;
        using TC = typename EngineC::value_type;
        using TD = typename EngineD::value_type;
        static_assert(std::is_same_v<TA, half_t>, "A dtype must be half");
        static_assert(std::is_same_v<TB, half_t>, "B dtype must be half");
        static_assert(std::is_same_v<TC, half_t>, "C dtype must be half");
        static_assert(std::is_same_v<TD, half_t>, "D dtype must be half");

        using LayoutV2A = gemm::detail::StrideToLayoutTagA_t<StrideA>;
        using LayoutV2B = gemm::detail::StrideToLayoutTagA_t<StrideB>;
        using LayoutV2C = gemm::detail::StrideToLayoutTagA_t<StrideC>;
        using LayoutV2D = gemm::detail::StrideToLayoutTagA_t<StrideD>;
        static_assert(std::is_same_v<LayoutV2C, LayoutV2D>, "C and D must have the same layout");

        using ElementAccumulator = float;                   // data type of accumulator
        using ElementComputeEpilogue = ElementAccumulator;  // data type of epilogue operations
        using MMAOp = arch::OpClassTensorOp;
        using SmArch = arch::Sm75;

        using DefaultConfig =
            gemm::device::DefaultGemmConfiguration<MMAOp, SmArch, TA, TB, TC, ElementAccumulator>;

        const int Stages = DefaultConfig::kStages;
        const int AlignmentA = AccessGranularityBits / cutlass::sizeof_bits<TA>::value;
        const int AlignmentB = AccessGranularityBits / cutlass::sizeof_bits<TB>::value;
        const int EpilogueAccessSize = AccessGranularityBits / cutlass::sizeof_bits<TC>::value;

        using ShapeMMAThreadBlock =
            typename DefaultConfig::ThreadblockShape;                 // threadblock tile MNK
        using ShapeMMAWarp = typename DefaultConfig::WarpShape;       // warp tile MNK
        using ShapeMMAOp = typename DefaultConfig::InstructionShape;  // MMA tile MNK
        using EpilogueOp = epilogue::thread::LinearCombination<
            TD,                      // data type of output matrix
            EpilogueAccessSize,      // elements per vectorized memory access
            ElementAccumulator,      // data type of accumulator
            ElementComputeEpilogue,  // the data type of epilogue operation
            Scale                    // operation to update the destination
            >;
        using ThreadblockSwizzle =
            typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

        using Gemm = cutlass::gemm::device::GemmUniversal<
            TA,
            LayoutV2A,
            TB,
            LayoutV2B,
            TC,
            LayoutV2C,
            ElementAccumulator,
            MMAOp,
            SmArch,
            ShapeMMAThreadBlock,
            ShapeMMAWarp,
            ShapeMMAOp,
            EpilogueOp,
            ThreadblockSwizzle,
            Stages,
            AlignmentA,
            AlignmentB>;

        int M = size<0>(mA.shape());
        int N = size<1>(mB.shape());
        int K = size<1>(mA.shape());

        auto leading_A = LayoutV2A::packed({get<1>(mA.stride()), get<0>(mA.stride())}).stride(0);
        auto leading_B = LayoutV2B::packed({get<1>(mB.stride()), get<0>(mB.stride())}).stride(0);
        auto leading_C = LayoutV2C::packed({get<1>(mC.stride()), get<0>(mC.stride())}).stride(0);
        auto leading_D = LayoutV2D::packed({get<1>(mD.stride()), get<0>(mD.stride())}).stride(0);
        int split_k_slices = (K + 127) / 128;  // Some random heuristic I invented

        typename Gemm::Arguments args{
            cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
            {M, N, K},                    // problem size (M N K)
            split_k_slices,               // batch size if mode=kBatched, k-tiles if mode=kGemm
                                          // kGemmSplitKParallel, 1 otherwise.
            {ElementComputeEpilogue(1),   // alpha
             ElementComputeEpilogue(1)},  // beta
            mA.data().get(),              // ptr to A (input)
            mB.data().get(),              // ptr to B (input)
            mC.data().get(),              // ptr to C (input)
            mD.data().get(),              // ptr to D (output)
            size(mA),                     // numel(A)
            size(mB),                     // numel(B)
            size(mC),                     // numel(C)
            size(mD),                     // numel(D)
            leading_A,                    // leading dimension of A
            leading_B,                    // leading dimension of B
            leading_C,                    // leading dimension of C
            leading_D                     // leading dimension of D
        };

        Gemm gemm_op;
        CUTLASS_CHECK(gemm_op.can_implement(args));

        size_t workspace_size = Gemm::get_workspace_size(args);
        device_memory::allocation<uint8_t> workspace;
        if (workspace_or_null.has_value()) {
            workspace = workspace_or_null.value();
        } else {
            workspace.reset(workspace_size);
        }
        // workspace_or_null.value_or(device_memory::allocate<uint8_t>(workspace_size));

        if (workspace.bytes() < workspace_size) {
            workspace.reset(workspace_size);
        }

        CUTLASS_CHECK(gemm_op.initialize(args, workspace.get()));

        return gemm_op;
    }

    template <
        int AccessGranularityBits = 16,
        typename EngineA,
        typename ShapeA,
        typename StrideA,
        typename EngineB,
        typename ShapeB,
        typename StrideB,
        typename EngineD,
        typename ShapeD,
        typename StrideD>
    auto gemm(
        Tensor<EngineA, Layout<ShapeA, StrideA>> &mA,
        Tensor<EngineB, Layout<ShapeB, StrideB>> &mB,
        Tensor<EngineD, Layout<ShapeD, StrideD>> &mD,
        std::optional<device_memory::allocation<uint8_t>> workspace_or_null = std::nullopt) {
        return gemm<AccessGranularityBits, ScaleType::Kind::OnlyAlphaScaling>(
            mA, mB, mD, mD, workspace_or_null);
    }
}  // namespace lib
