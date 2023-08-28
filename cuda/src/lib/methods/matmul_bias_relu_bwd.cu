#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/cache.h"
#include "lib/methods/matmul_bias_relu_bwd.h"
#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"

using namespace lib;

void matmul_bias_relu_bwd(tensor::Tensor &x, tensor::Tensor &w, tensor::Tensor &b,
                          tensor::Tensor &dL_dy, tensor::Tensor &dL_dx, tensor::Tensor &dL_dw,
                          tensor::Tensor &dL_db, cudnnHandle_t handle,
                          PlanCacheManager &plan_cache_manager) {
    /*
     * Formula: (http://cs231n.stanford.edu/handouts/linear-backprop.pdf)
     * 1. after_bias (B M N) = x (B M K) @ W (1 K N) + b (1 1 N)
     * 2. dL_dAfterBias (B M N) = reluBwd(x=after_bias, dL_dy) (B M N)
     * 3. dL_db (1 1 N) = sum(dL_dAfter_bias (B M N), dims=(0, 1))
     * 4. dL_dW (1 K N) = sum(X^T (B K M) @ dL_dAfterBias (B M N), dims=0)
     * 5. dL_dX (B M K) = dL_dAfterBias (B M N) @ W^T (1 N K)
     */
    int64_t B = x.get_dims()[0];
    int64_t M = x.get_dims()[1];
    int64_t K = x.get_dims()[2];
    int64_t N = dL_dy.get_dims()[2];

    auto dtype = CUDNN_DATA_HALF;
    auto mm_desc = cudnn_frontend::MatMulDescBuilder().setComputeType(dtype).build();
    auto add_desc = cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_ADD).build();
    auto sum_desc = cudnn_frontend::ReductionDescBuilder()
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                        .build();

    tensor::Tensor d_after_bias("d_after_bias", {B, M, N}, {M * N, N, 1}, CUDNN_DATA_FLOAT, false,
                                false);
    {  // Compute d_after_bias
        tensor::Tensor after_mm("after_mm", {B, M, N}, {M * N, N, 1}, dtype, true, false);
        auto mm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                         .setaMatDesc(x.get_cudnn())
                         .setbMatDesc(w.get_cudnn())
                         .setcMatDesc(after_mm.get_cudnn())
                         .setmatmulDesc(mm_desc)
                         .build();
        tensor::Tensor after_bias("after_bias", {B, M, N}, {M * N, N, 1}, dtype, true, false);
        auto bias_op =
            cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                .setxDesc(after_mm.get_cudnn())
                .setbDesc(b.get_cudnn())
                .setyDesc(after_bias.get_cudnn())
                .setpwDesc(add_desc)
                .build();
        auto act_desc =
            cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_RELU_BWD).build();
        auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setdyDesc(dL_dy.get_cudnn())
                          .setxDesc(after_bias.get_cudnn())
                          .setdxDesc(d_after_bias.get_cudnn())
                          .setpwDesc(act_desc)
                          .build();
        Ops ops{&mm_op, &bias_op, &act_op};
        DataPtrs data_ptrs{x.id_and_ptr(), w.id_and_ptr(), b.id_and_ptr(), dL_dy.id_and_ptr(),
                           d_after_bias.id_and_ptr()};
        auto &plan_cache = plan_cache_manager.get_plan_cache("matmul_bias_relu_bwd_1");
        execute_ops(handle, plan_cache, ops, data_ptrs);
    }
    {  // Compute dL_db
        auto sum_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                          .setxDesc(d_after_bias.get_cudnn())
                          .setyDesc(dL_db.get_cudnn())
                          .setreductionDesc(sum_desc)
                          .build();
        std::cout << d_after_bias << std::endl;
        std::cout << dL_db << std::endl;
        Ops ops{&sum_op};
        DataPtrs data_ptrs{d_after_bias.id_and_ptr(), dL_db.id_and_ptr()};
        auto &plan_cache = plan_cache_manager.get_plan_cache("matmul_bias_relu_bwd_2");
        execute_ops(handle, plan_cache, ops, data_ptrs);
        std::cout << dL_db << std::endl;
    }
    {  // Compute dL_dw
        tensor::Tensor dL_dw_before_sum("dL_dw_before_sum", {B, K, N}, {K * N, N, 1}, dtype, false,
                                        false);
        {
            auto x_T = x.transpose(1, 2);
            auto mm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                             .setaMatDesc(x_T.get_cudnn())
                             .setbMatDesc(d_after_bias.get_cudnn())
                             .setcMatDesc(dL_dw_before_sum.get_cudnn())
                             .setmatmulDesc(mm_desc)
                             .build();
            Ops ops{&mm_op};
            DataPtrs data_ptrs{x_T.id_and_ptr(), d_after_bias.id_and_ptr(),
                               dL_dw_before_sum.id_and_ptr()};
            auto &plan_cache = plan_cache_manager.get_plan_cache("matmul_bias_relu_bwd_3");
            execute_ops(handle, plan_cache, ops, data_ptrs);
        }
        {
            auto sum_op =
                cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                    .setxDesc(dL_dw_before_sum.get_cudnn())
                    .setyDesc(dL_dw.get_cudnn())
                    .setreductionDesc(sum_desc)
                    .build();
            std::cout << dL_dw_before_sum << std::endl;
            std::cout << dL_dw << std::endl;
            Ops ops{&sum_op};
            DataPtrs data_ptrs{dL_dw_before_sum.id_and_ptr(), dL_dw.id_and_ptr()};
            auto &plan_cache = plan_cache_manager.get_plan_cache("matmul_bias_relu_bwd_4");
            execute_ops(handle, plan_cache, ops, data_ptrs);
        }
    }
    {  // Compute dL_dx
        auto w_T = w.transpose(1, 2);
        auto mm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                         .setaMatDesc(d_after_bias.get_cudnn())
                         .setbMatDesc(w_T.get_cudnn())
                         .setcMatDesc(dL_dx.get_cudnn())
                         .setmatmulDesc(mm_desc)
                         .build();
        Ops ops{&mm_op};
        DataPtrs data_ptrs{d_after_bias.id_and_ptr(), w_T.id_and_ptr(), dL_dx.id_and_ptr()};
        auto &plan_cache = plan_cache_manager.get_plan_cache("matmul_bias_relu_bwd_5");
        execute_ops(handle, plan_cache, ops, data_ptrs);
    }
}