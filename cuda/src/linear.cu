#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/memory.h"
#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"
#include "linear.h"

using namespace std::string_literals;
using namespace lib;

cudnn_frontend::Tensor linear_backward(cudnn_frontend::Tensor& dL_dy, cudnn_frontend::Tensor& x,
                                       std::string name,
                                       std::vector<cudnn_frontend::Operation>& ops,
                                       std::set<std::pair<uint64_t, void*>>& data_ptrs) {
    /*
     * Formula: (http://cs231n.stanford.edu/handouts/linear-backprop.pdf)
     * 1. dL_dAfterBias (B M N) = reluBwd(x=afterBias, dL_dy) (B M N)
     * 2. dL_dAfterMatmul (B M N) = dL_dAfterBias (B M N)
     * 3. dL_db (1 1 N) = sum(dL_dAfterBias (B M N), dims=(0, 1))
     * 4. dL_dW (1 K N) = sum(X^T (B K M) x dL_dAfterMatmul (B M N), dims=0)
     * 5. dL_dX (B M K) = dL_dAfterMatmul (B M N) x W^T (1 N K)
     */
    int64_t B = dL_dy.getDim()[0];
    int64_t M = dL_dy.getDim()[1];
    int64_t N = dL_dy.getDim()[2];
    int64_t K = x.getDim()[2];

    // Create the tensors
    auto dtype = CUDNN_DATA_HALF;
    auto layout = Layout::ROW_MAJOR;
    // Real tensors
    auto weight = tensor(shape{1, K, N}, dtype, name + "_w"s, layout, false, false);
    auto dL_db = tensor(shape{B, 1, N}, CUDNN_DATA_FLOAT, name + "_dL_db"s, layout, false, false);
    // auto dL_dW = createTensor(shape{1, K, N}, dtype, name + "_dL_dW"s, layout, false, false);
    // auto dL_dX = createTensor(shape{B, M, K}, dtype, name + "_dL_dX"s, layout, false, false);

    // Allocate memory for real tensors
    void *dL_dbPtr, *dL_dWPtr, *dL_dXPtr;
    check_cudnn_status(cudaMalloc(&dL_dbPtr, tensor_size(shape{B, 1, N}, CUDNN_DATA_FLOAT)));
    // checkCudaStatus(cudaMalloc(&dL_dWPtr, tensorSize(shape{1, K, N}, dtype)));
    // checkCudaStatus(cudaMalloc(&dL_dXPtr, tensorSize(shape{B, M, K}, dtype)));
    data_ptrs.insert(std::pair<uint64_t, void*>((uint64_t)dL_db.getId(), dL_dbPtr));
    // data_ptrs.insert(std::pair<uint64_t, void*>((uint64_t)dL_dW.getId(), dL_dWPtr));
    // data_ptrs.insert(std::pair<uint64_t, void*>((uint64_t)dL_dX.getId(), dL_dXPtr));

    // Step 1
    auto afterBias = tensor(shape{B, M, N}, dtype, name + "_after_bias"s, layout, true, false);
    auto dL_dAfterBias =
        tensor(shape{B, M, N}, dtype, name + "_dY_dAfterBias"s, layout, true, false);
    auto reluBwdDesc =
        cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_RELU_BWD).build();
    auto reluBwdOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                         .setdyDesc(dL_dy)
                         .setxDesc(afterBias)
                         .setdxDesc(dL_dAfterBias)
                         .setpwDesc(reluBwdDesc)
                         .build();
    ops.push_back(std::move(reluBwdOp));

    // // Step 2 (included just for completeness)
    // auto dL_dAfterMatmul =
    //     createTensor(shape{B, M, N}, dtype, name + "_dL_dAfterMatmul"s, layout, true, false);
    // auto noOpDesc =
    //     cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_IDENTITY).build();
    // auto noOp = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    //                 .setxDesc(dL_dAfterBias)
    //                 .setyDesc(dL_dAfterMatmul)
    //                 .setpwDesc(noOpDesc)
    //                 .build();
    // ops.push_back(std::move(noOp));

    // Step 3
    auto sumDesc =
        cudnn_frontend::ReductionDescBuilder().setReductionOp(CUDNN_REDUCE_TENSOR_ADD).build();
    auto sumOp1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                      .setxDesc(dL_dAfterBias)
                      .setyDesc(dL_db)
                      .setreductionDesc(sumDesc)
                      .build();
    ops.push_back(std::move(sumOp1));

    // // Step 4.1 (transpose)
    // auto x_T = createTensor(shape{B, M, K}, shape{M * K, 1, M}, dtype, name + "_x_T"s, true,
    // false); auto transposeOp1 =
    //     cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    //         .setxDesc(x)
    //         .setyDesc(x_T)
    //         .setpwDesc(noOpDesc)
    //         .build();
    // ops.push_back(std::move(transposeOp1));

    // // Step 4.2 (matmul)
    // auto beforeSum = createTensor(shape{B, K, N}, dtype, name + "_beforeSum"s, layout, true,
    // false); auto mm_desc = cudnn_frontend::MatMulDescBuilder().setComputeType(dtype).build();
    // auto matmulOp1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
    //                      .setaMatDesc(x_T)
    //                      .setbMatDesc(dL_dAfterMatmul)
    //                      .setcMatDesc(beforeSum)
    //                      .setmatmulDesc(mm_desc)
    //                      .build();
    // ops.push_back(std::move(matmulOp1));

    // // Step 4.3 (sum)
    // auto sumOp2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
    //                   .setxDesc(beforeSum)
    //                   .setyDesc(dL_dW)
    //                   .setreductionDesc(sumDesc)
    //                   .build();
    // ops.push_back(std::move(sumOp2));

    // // Step 5.1 (transpose)
    // auto w_T = createTensor(shape{1, K, N}, shape{K * N, 1, K}, dtype, name + "_w_T"s, true,
    // false); auto transposeOp2 =
    //     cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    //         .setxDesc(weight)
    //         .setyDesc(w_T)
    //         .setpwDesc(noOpDesc)
    //         .build();
    // ops.push_back(std::move(transposeOp2));

    // // Step 5.2 (matmul)
    // auto matmulOp2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
    //                      .setaMatDesc(dL_dAfterMatmul)
    //                      .setbMatDesc(w_T)
    //                      .setcMatDesc(dL_dX)
    //                      .setmatmulDesc(mm_desc)
    //                      .build();
    // ops.push_back(std::move(matmulOp2));

    return dL_db;
}