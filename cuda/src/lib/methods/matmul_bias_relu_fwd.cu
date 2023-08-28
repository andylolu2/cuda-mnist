#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/cache.h"
#include "lib/methods/matmul_bias_relu_fwd.h"
#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"

using namespace lib;

void matmul_bias_relu_fwd(tensor::Tensor &x, tensor::Tensor &w, tensor::Tensor &b,
                          tensor::Tensor &y, cudnnHandle_t handle,
                          PlanCacheManager &plan_cache_manager) {
    /* Formula:
     * 1. after_mm (B M N) = X (B M K) @ W (1 K N)
     * 2. after_bias (B M N) = after_mm (B M N) + b (1 1 N)
     * 3. Y (B M N) = relu(after_bias (B M N))
     */
    int64_t B = x.get_dims()[0];
    int64_t M = x.get_dims()[1];
    int64_t N = y.get_dims()[2];

    auto dtype = CUDNN_DATA_HALF;

    // Step 1
    tensor::Tensor after_mm("after_mm", {B, M, N}, {M * N, N, 1}, dtype, true, false);  // virtual
    auto mm_desc = cudnn_frontend::MatMulDescBuilder().setComputeType(dtype).build();
    auto mm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                     .setaMatDesc(x.get_cudnn())
                     .setbMatDesc(w.get_cudnn())
                     .setcMatDesc(after_mm.get_cudnn())
                     .setmatmulDesc(mm_desc)
                     .build();

    // Step 2
    tensor::Tensor after_bias("after_bias", {B, M, N}, {M * N, N, 1}, dtype, true,
                              false);  // virtual
    auto bias_desc = cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_ADD).build();
    auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                       .setxDesc(after_mm.get_cudnn())
                       .setbDesc(b.get_cudnn())
                       .setyDesc(after_bias.get_cudnn())
                       .setpwDesc(bias_desc)
                       .build();

    // Step 3
    auto act_desc =
        cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_RELU_FWD).build();
    auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(after_bias.get_cudnn())
                      .setyDesc(y.get_cudnn())
                      .setpwDesc(act_desc)
                      .build();

    // Create the graph
    Ops ops{&mm_op, &bias_op, &act_op};
    DataPtrs data_ptrs{x.id_and_ptr(), w.id_and_ptr(), b.id_and_ptr(), y.id_and_ptr()};
    auto &plan_cache = plan_cache_manager.get_plan_cache("matmul_bias_relu_fwd");
    execute_ops(handle, plan_cache, ops, data_ptrs);
}