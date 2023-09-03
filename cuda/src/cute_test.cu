#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/print.h"

using namespace cute;

template <
    // typename MShape,
    // typename NShape,
    // typename KShape,
    // typename TA,
    // typename AStride,
    typename ABlockLayout,
    typename EngineA,
    typename LayoutA,
    typename AThreadLayout,
    // typename TB,
    // typename BStride,
    typename BBlockLayout,
    typename BThreadLayout,
    typename EngineB,
    typename LayoutB,
    // typename TC,
    // typename CStride,
    typename CBlockLayout,
    typename CThreadLayout,
    typename EngineC,
    typename LayoutC,
    // typename DBlockLayout,
    typename EngineD,
    typename LayoutD>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void matmul_bias_relu_device(
    // MShape M,
    // NShape N,
    // KShape K,
    Tensor<EngineA, LayoutA> mA,
    Tensor<EngineB, LayoutB> mB,
    Tensor<EngineC, LayoutC> mC,
    Tensor<EngineD, LayoutD> mD,
    ABlockLayout blockA,
    AThreadLayout tA,
    BBlockLayout blockB,
    BThreadLayout tB,
    // TC* C,
    // CStride dC,
    CBlockLayout blockC,
    CThreadLayout tC) {
    using X = Underscore;
    using TA = typename EngineA::value_type;
    using TB = typename EngineB::value_type;
    using TC = typename EngineC::value_type;

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
    auto gD = local_tile(mD, blk_shape, blk_coord, Step<_1, _1, X>{});  // (BLK_M BLK_N)

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
    // gC (BLK_M BLK_N) by the tile of tC (tC.M tC.N) -> (BLK_M/tC.M BLK_N/tC.N)
    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    // gD (BLK_M BLK_N) by the tile of tC (tC.M tC.N) -> (BLK_M/tC.M BLK_N/tC.N)
    auto tCgD = local_partition(gD, tC, threadIdx.x, Step<_1, _1>{});
    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC);  // (BLK_M/THR_M BLK_N/THR_N)

    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    Tensor tAcA = local_partition(cA, tA, threadIdx.x);
    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));
    Tensor tBcB = local_partition(cB, tB, threadIdx.x);
    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC)));
    Tensor tCcC = local_partition(cC, tC, threadIdx.x);

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

        cp_async_fence();
        cp_async_wait<0>();

        __syncthreads();

        // Step 3
        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }

    //
    // Epilogue
    //
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        if (elem_less(tCcC(i), make_coord(m_residue, n_residue))) {
            auto x = tCrC(i) + tCgC(i);
            tCgD(i) = x > TC(0) ? x : TC(0);
        }
    }
}

template <
    typename EngineA,
    typename LayoutA,
    // typename TA,
    typename EngineB,
    typename LayoutB,
    // typename TB,
    typename EngineC,
    typename LayoutC,
    // typename TC,
    typename EngineD,
    typename LayoutD
    // typename TD
    >
void matmul_bias_relu(
    Tensor<EngineA, LayoutA> x,
    Tensor<EngineB, LayoutB> w,
    Tensor<EngineC, LayoutC> b,
    Tensor<EngineD, LayoutD> y,
    // TA* x_ptr,
    // TB* w_ptr,
    // TC* b_ptr,
    // TD* y_ptr,
    cudaStream_t stream = 0) {
    // static_assert(std::is_same_v<typename EngineA::value_type, TA>);
    // static_assert(std::is_same_v<typename EngineB::value_type, TB>);
    // static_assert(std::is_same_v<typename EngineC::value_type, TC>);
    // static_assert(std::is_same_v<typename EngineD::value_type, TD>);

    assert(size<0>(y.shape()) == size<0>(x.shape()));  // match M
    assert(size<1>(y.shape()) == size<0>(w.shape()));  // match N
    assert(size<1>(w.shape()) == size<1>(x.shape()));  // match K
    assert(b.shape() == y.shape());

    auto M = size<0>(x.shape());
    auto N = size<0>(w.shape());
    auto K = size<1>(x.shape());

    // Define strides (mixed)
    // auto dA = x.stride();
    // auto dB = w.stride();
    // auto dC = y.stride();

    // Define block sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};

    // Define the block layouts (static)
    auto sA = make_layout(make_shape(bM, bK));  // split (M K) into (bM bK)
    auto sB = make_layout(make_shape(bN, bK));  // split (N K) into (bN bK)
    auto sC = make_layout(make_shape(bM, bN));  // split (M N) into (bM bN)

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));   // partitioning (bM bK) for copy
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}));   // partitioning (bN bK) for copy
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));  // partitioning (bM bN) for compute

    dim3 dimBlock(size(tC));
    dim3 dimGrid(ceil_div(M, bM), ceil_div(N, bN));

    matmul_bias_relu_device<<<dimGrid, dimBlock, 0, stream>>>(x, w, b, y, sA, tA, sB, tB, sC, tC);
    // M, N, K, x, w, y, sA, tA, sB, tB, sC, tC);
    // x_ptr, dA, sA, tA, w_ptr, dB, sB, tB, y_ptr, dC, sC, tC);
}

template <typename Tensor>
auto transpose(Tensor& tensor) {
    auto new_shape = make_shape(size<1>(tensor.shape()), size<0>(tensor.shape()));
    auto new_stride = make_stride(size<1>(tensor.stride()), size<0>(tensor.stride()));
    return make_tensor(tensor.engine().begin(), new_shape, new_stride);
}

int main(int argc, char const* argv[]) {
    if (argc != 4) {
        printf("Usage: %s M N K\n", argv[0]);
        return 0;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    // int M = 6;
    // int N = 9;
    // int K = 9;
    // int M_sub = 6;
    // int N_sub = 9;
    // int K_sub = 3;

    cutlass::DeviceAllocation<cutlass::half_t> A(M * K);
    cutlass::DeviceAllocation<cutlass::half_t> B(N * K);
    cutlass::DeviceAllocation<cutlass::half_t> C(N);
    cutlass::DeviceAllocation<cutlass::half_t> D(M * N);

    Tensor tensor_x = make_tensor(make_gmem_ptr(A.get()), make_shape(M, K));
    Tensor tensor_w = make_tensor(make_gmem_ptr(B.get()), make_shape(N, K));
    Tensor tensor_b = make_tensor(make_gmem_ptr(C.get()), make_shape(N));
    Tensor tensor_b_expanded =
        make_tensor(make_gmem_ptr(C.get()), make_shape(M, N), make_stride(0, 1));
    Tensor tensor_y = make_tensor(make_gmem_ptr(D.get()), make_shape(M, N));

    lib::init::normal<<<1, 64>>>(tensor_x, 0.0_hf, 1.0_hf);
    lib::init::identity<<<1, 64>>>(tensor_w);
    lib::init::constant<<<1, 64>>>(tensor_b, 1.0_hf);
    // lib::init::constant<<<1, 64>>>(tensor_w, 1.0_hf);
    // lib::init::identity<<<1, 64>>>(tensor_w);

    lib::print_device_tensor("Tensor x", tensor_x);
    lib::print_device_tensor("Tensor w", tensor_w);
    lib::print_device_tensor("Tensor b", tensor_b);
    lib::print_device_tensor("Tensor y", tensor_y);

    matmul_bias_relu(tensor_x, tensor_w, tensor_b_expanded, tensor_y);

    lib::print_device_tensor("Tensor y after", tensor_y);

    return 0;
}