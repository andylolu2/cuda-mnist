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

#include <cute/tensor.hpp>
#include <optional>

#include "lib/utils/macros.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace op {
        using ScaleType = cutlass::epilogue::thread::ScaleType;

        template <
            int AccessGranularityBits = 16,  // Problem size (in bits) needs to be a multiple
                                             // of this number. 128 gives the best performance.
            ScaleType::Kind Scale = ScaleType::Kind::NoBetaScaling,  /// Control Alpha and Beta
                                                                     /// scaling
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
            Tensor<EngineA, Layout<ShapeA, StrideA>> const &mA,
            Tensor<EngineB, Layout<ShapeB, StrideB>> const &mB,
            Tensor<EngineC, Layout<ShapeC, StrideC>> const &mC,
            Tensor<EngineD, Layout<ShapeD, StrideD>> const &mD,
            DeviceAllocation<uint8_t> &workspace) {
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
            static_assert(
                std::is_same_v<LayoutV2C, LayoutV2D>, "C and D must have the same layout");

            using ElementAccumulator = half_t;                  // Data type of accumulator
            using ElementComputeEpilogue = ElementAccumulator;  // Data type of epilogue operations
            using MMAOp = arch::OpClassTensorOp;                // Use Tensor Cores
            using SmArch = arch::Sm75;                          // Turing architecture

            using DefaultConfig = gemm::device::
                DefaultGemmConfiguration<MMAOp, SmArch, TA, TB, TC, ElementAccumulator>;

            // Number of pipelining stages, defaults to 2
            const int Stages = DefaultConfig::kStages;
            const int AlignmentA = AccessGranularityBits / cutlass::sizeof_bits<TA>::value;
            const int AlignmentB = AccessGranularityBits / cutlass::sizeof_bits<TB>::value;
            const int EpilogueAccessSize = AccessGranularityBits / cutlass::sizeof_bits<TC>::value;

            // Threadblock partition size, defaults to (128 256 32)
            using ShapeMMAThreadBlock = typename DefaultConfig::ThreadblockShape;
            // Warp partition size, defaults to (64 64 32)
            using ShapeMMAWarp = typename DefaultConfig::WarpShape;
            // Instruction size, defaults to (16 8 8)
            using ShapeMMAOp = typename DefaultConfig::InstructionShape;
            using EpilogueOp = epilogue::thread::LinearCombination<
                TD,                      // data type of C and D
                EpilogueAccessSize,      // elements per vectorized memory access for C and D
                ElementAccumulator,      // data type of accumulator
                ElementComputeEpilogue,  // the data type of epilogue operation
                Scale                    // operation to update the destination
                >;

            // Swizzling ensures that nearby threadblocks are executed together to improve cache
            // efficiency
            using ThreadblockSwizzle =
                typename cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

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

            auto leading_A =
                LayoutV2A::packed({get<1>(mA.stride()), get<0>(mA.stride())}).stride(0);
            auto leading_B =
                LayoutV2B::packed({get<1>(mB.stride()), get<0>(mB.stride())}).stride(0);
            auto leading_C =
                LayoutV2C::packed({get<1>(mC.stride()), get<0>(mC.stride())}).stride(0);
            auto leading_D =
                LayoutV2D::packed({get<1>(mD.stride()), get<0>(mD.stride())}).stride(0);
            int split_k_slices = 1;

            typename Gemm::Arguments args{
                cutlass::gemm::GemmUniversalMode::kGemm,
                {M, N, K},       // problem size (M N K)
                split_k_slices,  // batch count or splitk slices
                {
                    static_cast<ElementComputeEpilogue>(1),  // alpha
                    static_cast<ElementComputeEpilogue>(1)   // beta
                },
                mA.data().get(),  // ptr to A (input)
                mB.data().get(),  // ptr to B (input)
                mC.data().get(),  // ptr to C (input)
                mD.data().get(),  // ptr to D (output)
                size(mA),         // batch stride A
                size(mB),         // batch stride B
                size(mC),         // batch stride C
                size(mD),         // batch stride D
                leading_A,        // leading dimension A
                leading_B,        // leading dimension B
                leading_C,        // leading dimension C
                leading_D         // leading dimension D
            };

            Gemm gemm_op;
            CUTLASS_CHECK(gemm_op.can_implement(args));

            size_t workspace_size = Gemm::get_workspace_size(args);
            if (workspace.capacity < workspace_size) {
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
            Tensor<EngineA, Layout<ShapeA, StrideA>> const &mA,
            Tensor<EngineB, Layout<ShapeB, StrideB>> const &mB,
            Tensor<EngineD, Layout<ShapeD, StrideD>> const &mD,
            DeviceAllocation<uint8_t> &workspace) {
            return gemm<AccessGranularityBits, ScaleType::Kind::OnlyAlphaScaling>(
                mA, mB, mD, mD, workspace);
        }
    }  // namespace op
}  // namespace lib
