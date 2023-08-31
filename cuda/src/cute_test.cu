#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>

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
    class CThreadLayout,
    class Alpha,
    class Beta>
__global__ static __launch_bounds__(decltype(size(CThreadLayout{}))::value) void gemm_device(
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
    CThreadLayout tC,
    Alpha alpha,
    Beta beta) {
    using namespace cute;
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
    // gB (BLK_N BLK_K k) / tB -> tBgB (BLK_N/tB.N BLK_K/tB.K k)
    auto tBgB = local_partition(gB, tB, threadIdx.x);
    // sA (BLK_M BLK_K) / tA -> tAsA (BLK_M/tA.M BLK_K/tA.K)
    auto tAsA = local_partition(sA, tA, threadIdx.x);
    // sA (BLK_M BLK_K) by the rows of tC (tC.M tC.N) -> (BLK_M/tC.M BLK_K)
    auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    // sB (BLK_N BLK_K) / tB -> tBsB (BLK_N/tB.N BLK_K/tB.K)
    auto tBsB = local_partition(sB, tB, threadIdx.x);
    // sB (BLK_N BLK_K) by the cols of tC (tC.M tC.N) -> (BLK_N/tC.N BLK_K)
    auto tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    // gC (BLK_M BLK_N) by the tile of tC (tC.M tC.N) -> (BLK_M/tC.M BLK_N/tC.N)
    auto tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    // Allocate the accumulators -- same size as the projected data
    auto tCrC = make_fragment_like(tCgC);  // (BLK_M/THR_M BLK_N/THR_N):(1 BLK_M)

    // Clear the accumulators
    clear(tCrC);

    // TUTORIAL: Example of a very simple compute loop
    //   Data is read from global to shared memory via the tA|tB partitioning
    //   gemm(.) operates on the shared memory directly via the tC partitioning

    auto k_max = size<2>(tAgA);

    for (int k = 0; k < k_max; ++k) {
        // Step 1 & 2
        copy(tAgA(_, _, k), tAsA);
        copy(tBgB(_, _, k), tBsB);

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

    axpby(alpha, tCrC, beta, tCgC);
}

template <typename TA, typename TB, typename TC, typename Alpha, typename Beta>
void my_gemm(
    int m,
    int n,
    int k,
    Alpha alpha,
    TA const* A,
    int ldA,
    TB const* B,
    int ldB,
    Beta beta,
    TC* C,
    int ldC,
    cudaStream_t stream = 0) {
    using namespace cute;

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
    auto tA = make_layout(make_shape(Int<8>{}, Int<8>{}));  // partitioning of A for copy
    auto tB = make_layout(make_shape(Int<8>{}, Int<8>{}));  // partitioning of B for copy
    auto tC = make_layout(make_shape(Int<8>{}, Int<8>{}));  // partitioning of C for compute

    dim3 dimBlock(size(tC));
    dim3 dimGrid(ceil_div(size(M), size(bM)), ceil_div(size(N), size(bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
        M, N, K, A, dA, sA, tA, B, dB, sB, tB, C, dC, sC, tC, alpha, beta);
}

template <typename Tensor>
__global__ void fill_kernel(Tensor tensor, cutlass::half_t value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < tensor.size(); idx += stride) {
        tensor(idx) = value;
    }
}

int main() {
    int M = 8;
    int N = 8;
    int K = 8;

    cutlass::DeviceAllocation<cutlass::half_t> A(M * K);
    cutlass::DeviceAllocation<cutlass::half_t> B(N * K);
    cutlass::DeviceAllocation<cutlass::half_t> C(M * N);

    std::vector<cutlass::half_t> A_host(A.size());
    std::vector<cutlass::half_t> B_host(B.size());
    std::vector<cutlass::half_t> C_host(C.size());

    Tensor tensor_a = make_tensor(make_gmem_ptr(A.get()), make_shape(M, K));
    Tensor tensor_b = make_tensor(make_gmem_ptr(B.get()), make_shape(N, K));
    Tensor tensor_c = make_tensor(make_gmem_ptr(C.get()), make_shape(M, N));

    fill_kernel<<<1, 64>>>(tensor_a, 1.0_hf);
    fill_kernel<<<1, 64>>>(tensor_b, 1.0_hf);

    Tensor tensor_a_host = make_tensor(A_host.data(), make_shape(M, K));
    Tensor tensor_b_host = make_tensor(B_host.data(), make_shape(N, K));
    Tensor tensor_c_host = make_tensor(C_host.data(), make_shape(M, N));

    A.copy_to_host(A_host.data());
    B.copy_to_host(B_host.data());
    C.copy_to_host(C_host.data());
    for (int i = 0; i < tensor_a_host.size(); i++) {
        std::cout << tensor_a_host(i) << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < tensor_b_host.size(); i++) {
        std::cout << tensor_b_host(i) << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < tensor_c_host.size(); i++) {
        std::cout << tensor_c_host(i) << " ";
    }
    std::cout << std::endl;

    my_gemm(M, N, K, 1.0_hf, A.get(), M, B.get(), N, 0.0_hf, C.get(), M);

    A.copy_to_host(A_host.data());
    B.copy_to_host(B_host.data());
    C.copy_to_host(C_host.data());
    for (int i = 0; i < tensor_a_host.size(); i++) {
        std::cout << tensor_a_host(i) << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < tensor_b_host.size(); i++) {
        std::cout << tensor_b_host(i) << " ";
    }
    std::cout << std::endl;
    for (int i = 0; i < tensor_c_host.size(); i++) {
        std::cout << tensor_c_host(i) << " ";
    }
    std::cout << std::endl;

    return 0;
}