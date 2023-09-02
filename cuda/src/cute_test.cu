#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>

#include "lib/fill.h"
#include "lib/print.h"

using namespace cute;

template <
    class MShape,
    class NShape,
    class KShape,
    class TA,
    class AStride,
    class ABlockLayout,
    class AThreadLayout,
    class TB,
    class BStride,
    class BBlockLayout,
    class BThreadLayout,
    class TC,
    class CStride,
    class CBlockLayout,
    class CThreadLayout>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void matmul_bias_relu_device(
    MShape M,
    NShape N,
    KShape K,
    TA const* A,
    AStride dA,
    ABlockLayout blockA,
    AThreadLayout tA,
    TB const* B,
    BStride dB,
    BBlockLayout blockB,
    BThreadLayout tB,
    TC* C,
    CStride dC,
    CBlockLayout blockC,
    CThreadLayout tC) {
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<ABlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BBlockLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CBlockLayout>::value);

    CUTE_STATIC_ASSERT(is_static<AThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<BThreadLayout>::value);
    CUTE_STATIC_ASSERT(is_static<CThreadLayout>::value);

    CUTE_STATIC_ASSERT_V(size(tA) == size(tC));
    CUTE_STATIC_ASSERT_V(size(tB) == size(tC));

    CUTE_STATIC_ASSERT_V(shape<0>(blockA) == shape<0>(blockC));  // BLK_M
    CUTE_STATIC_ASSERT_V(shape<0>(blockB) == shape<1>(blockC));  // BLK_N
    CUTE_STATIC_ASSERT_V(shape<1>(blockA) == shape<1>(blockB));  // BLK_K

    // Block size of the compute tile
    auto BLK_M = shape<0>(blockC);
    auto BLK_N = shape<1>(blockC);
    auto BLK_K = shape<1>(blockA);

    // Compute the "residues"
    // TODO: Handle K-out-of-bounds
    // auto k_residue = K - BLK_K * (ceil_div(K, BLK_K));             // (-BLK_K,0]
    auto m_residue = min(BLK_M, M - blockIdx.x * BLK_M);
    auto n_residue = min(BLK_N, N - blockIdx.y * BLK_N);

    // --- Global ---
    auto mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);  // (M K):(1 M)
    auto mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);  // (N K):(1 N)
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);  // (M N):(1 M)

    // --- Thread block level ---
    //  1. copy gA (global memory) -> sA (shared memory)
    //  2. copy gB (global memory) -> sB (shared memory)
    //  3. compute gC += sA @ sB
    // where gA & sA are fragments of mA, gB & sB are fragments of mB, gC is a fragment of mC.
    __shared__ TA smemA[cosize_v<ABlockLayout>];
    __shared__ TB smemB[cosize_v<BBlockLayout>];
    auto sA = make_tensor(make_smem_ptr(smemA), blockA);  // (BLK_M BLK_K)
    auto sB = make_tensor(make_smem_ptr(smemB), blockB);  // (BLK_N BLK_K)

    auto blk_shape = make_shape(size<0>(sA), size<0>(sB), size<1>(sB));  // (BLK_M BLK_N BLK_K)
    auto blk_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m n k)

    // k := K / BLK_K, the number of blocks in the K-dimension
    auto gA =
        local_tile(mA, blk_shape, blk_coord, Step<_1, X, _1>{});  // (BLK_M BLK_K k):(1 M M*BLK_K)
    auto gB =
        local_tile(mB, blk_shape, blk_coord, Step<X, _1, _1>{});  // (BLK_N,BLK_K,k):(1 N N*BLK_K)
    auto gC = local_tile(mC, blk_shape, blk_coord, Step<_1, _1, X>{});  // (BLK_M BLK_N):(1 M)

    // --- Thread level ---
    // for k_idx in 0..K/BLK_K:
    //   1. copy tAgA[:, :, k_idx] (global memory) -> tAsA (shared memory)
    //   2. copy tBgB[:, :, k_idx] (global memory) -> tBsB (shared memory)
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
    auto tApA = make_tensor<bool>(
        make_shape(size<0>(tAgA), size<1>(tAgA)), make_stride(Int<1>{}, Int<0>{}));

    // gB (BLK_N BLK_K k) / tB -> tBgB (BLK_N/tB.N BLK_K/tB.K k)
    auto tBgB = local_partition(gB, tB, threadIdx.x);
    // sB (BLK_N BLK_K) / tB -> tBsB (BLK_N/tB.N BLK_K/tB.K)
    auto tBsB = local_partition(sB, tB, threadIdx.x);
    // predicate mask for the tBgB -> tBsB copy (BLK_N/tB.N BLK_K/tB.K)
    auto tBpB = make_tensor<bool>(
        make_shape(size<0>(tBgB), size<1>(tBgB)), make_stride(Int<1>{}, Int<0>{}));

    // sA (BLK_M BLK_K) by the rows of tC (tC.M tC.N) -> (BLK_M/tC.M BLK_K)
    auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    // sB (BLK_N BLK_K) by the cols of tC (tC.M tC.N) -> (BLK_N/tC.N BLK_K)
    auto tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    // gC (BLK_M BLK_N) by the tile of tC (tC.M tC.N) -> (BLK_M/tC.M BLK_N/tC.N)
    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC);  // (BLK_M/THR_M BLK_N/THR_N)

    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));
    Tensor tAcA = local_partition(cA, tA, threadIdx.x);
    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));
    Tensor tBcB = local_partition(cB, tB, threadIdx.x);
    Tensor cC = make_identity_tensor(make_shape(size<0>(gC), size<1>(gC)));
    Tensor tCcC = local_partition(cC, tC, threadIdx.x);

    // Populate the predicate masks
    CUTE_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
        tApA(m, 0) = get<0>(tAcA(m, 0)) < m_residue;
    }
    CUTE_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
        tBpB(n, 0) = get<0>(tBcB(n, 0)) < n_residue;
    }

    // Clear the accumulators
    clear(tCrC);

    // if (threadIdx.x == 0) {
    //     print("tAsA: ");
    //     print(tAsA);
    //     print("\ntBsB: ");
    //     print(tBsB);
    //     print("\ntApA: ");
    //     print(tApA);
    //     print("\ntBpB: ");
    //     print(tBpB);
    //     print("\ntCsA: ");
    //     print(tCsA);
    //     print("\ntCsB: ");
    //     print(tCsB);
    //     print("\ntCrC: ");
    //     print(tCrC);
    //     print("\ntAcA: ");
    //     print(tAcA);
    //     print("\ntBcB: ");
    //     print(tBcB);
    //     print("\ntApA: ");
    //     print(tApA);
    //     print("\ntBpB: ");
    //     print(tBpB);
    //     print("\nBLK_M: ");
    //     print(BLK_M);
    //     print("\nBLK_N: ");
    //     print(BLK_N);
    //     print("\nBLK_K: ");
    //     print(BLK_K);
    //     print("\nm_residue: ");
    //     print(m_residue);
    //     print("\nn_residue: ");
    //     print(n_residue);
    //     print("\nk_residue: ");
    //     print(k_residue);
    //     print("\n");
    // }

    auto k = size<2>(tAgA);
    for (int k_idx = 0; k_idx < k; ++k_idx) {
        // Step 1 & 2
        copy_if(tApA, tAgA(_, _, k_idx), tAsA);
        copy_if(tBpB, tBgB(_, _, k_idx), tBsB);

        // if (threadIdx.x == 0) {
        //     for (int i = 0; i < size(sA); ++i) {
        //         print(sA(i));
        //     }
        //     print("\n");
        // }
        // copy(tAgA(_, _, k_idx), tAsA);
        // copy(tBgB(_, _, k_idx), tBsB);

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
            tCgC(i) = x > TC(0) ? x : TC(0);
        }
    }
}

template <typename TA, typename TB, typename TC>
void matmul_bias_relu(
    int m,
    int n,
    int k,
    TA const* A,
    int ldA,
    TB const* B,
    int ldB,
    TC* C,
    int ldC,
    cudaStream_t stream = 0) {
    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);

    // Define strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    // Define block sizes (static)
    auto bM = Int<16>{};
    auto bN = Int<16>{};
    auto bK = Int<8>{};

    // Define the block layouts (static)
    auto sA = make_layout(make_shape(bM, bK));
    auto sB = make_layout(make_shape(bN, bK));
    auto sC = make_layout(make_shape(bM, bN));

    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<2>{}, Int<2>{}));  // partitioning of A for copy
    auto tB = make_layout(make_shape(Int<2>{}, Int<2>{}));  // partitioning of B for copy
    auto tC = make_layout(make_shape(Int<2>{}, Int<2>{}));  // partitioning of C for compute

    dim3 dimBlock(size(tC));
    dim3 dimGrid(ceil_div(size(M), size(bM)), ceil_div(size(N), size(bN)));
    matmul_bias_relu_device<<<dimGrid, dimBlock, -1, stream>>>(
        M, N, K, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC);
}

int main() {
    int M = 7;
    int N = 4;
    int K = 4;
    int i = 0;

    cutlass::DeviceAllocation<cutlass::half_t> A(M * K);
    cutlass::DeviceAllocation<cutlass::half_t> B(N * K);
    cutlass::DeviceAllocation<cutlass::half_t> C(M * N);

    Tensor tensor_a = make_tensor(make_gmem_ptr(A.get()), make_shape(M, K));
    Tensor tensor_b = make_tensor(make_gmem_ptr(B.get()), make_shape(N, K));
    Tensor tensor_c = make_tensor(make_gmem_ptr(C.get()), make_shape(M, N));

    std::cout << cute::max(tensor_a.shape()) << cute::min(tensor_a.shape()) << std::endl;

    lib::init::normal<<<1, 64>>>(tensor_a, 0.0_hf);
    lib::init::identity<<<1, 64>>>(tensor_b);

    lib::print_device_tensor("Tensor A", tensor_a);
    lib::print_device_tensor("Tensor B", tensor_b);
    lib::print_device_tensor("Tensor C", tensor_c);

    matmul_bias_relu(M, N, K, A.get(), M, B.get(), N, C.get(), M);

    lib::print_device_tensor("Tensor C after", tensor_c);

    return 0;
}