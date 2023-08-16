#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/methods/matmul_bias_relu_fwd.h"
#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"

using namespace std::string_literals;
using namespace lib;

void matmul_bias_relu_fwd(cudnn_frontend::Tensor& x, void* x_ptr, cudnn_frontend::Tensor& w,
                          void* w_ptr, cudnn_frontend::Tensor& b, void* b_ptr,
                          cudnn_frontend::Tensor& y, void* y_ptr, cudnnHandle_t handle) {
    /* Formula:
     * 1. after_mm (B M N) = X (B M K) x W (1 K N)
     * 2. after_bias (B M N) = after_mm (B M N) + b (1 1 N)
     * 3. Y (B M N) = relu(after_bias (B M N))
     */
    Ops ops;
    DataPtrs data_ptrs;
    int64_t B = x.getDim()[0];
    int64_t M = x.getDim()[1];
    int64_t K = x.getDim()[2];
    int64_t N = y.getDim()[2];

    // Insert real tensors
    data_ptrs.insert(std::pair<uint64_t, void*>((uint64_t)x.getId(), x_ptr));
    data_ptrs.insert(std::pair<uint64_t, void*>((uint64_t)w.getId(), w_ptr));
    data_ptrs.insert(std::pair<uint64_t, void*>((uint64_t)b.getId(), b_ptr));
    data_ptrs.insert(std::pair<uint64_t, void*>((uint64_t)y.getId(), y_ptr));

    // Create virtual tensors
    auto dtype = CUDNN_DATA_HALF;
    auto layout = tensor::layout::Row;
    auto after_mm = tensor::create_cudnn(shape{B, M, N}, dtype, "after_mm"s, layout, true, false);
    auto after_bias =
        tensor::create_cudnn(shape{B, M, N}, dtype, "after_bias"s, layout, true, false);

    // Step 1
    auto mm_desc = cudnn_frontend::MatMulDescBuilder().setComputeType(dtype).build();
    auto mm_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                     .setaMatDesc(x)
                     .setbMatDesc(w)
                     .setcMatDesc(after_mm)
                     .setmatmulDesc(mm_desc)
                     .build();
    ops.push_back(std::move(mm_op));

    // Step 2
    auto bias_desc = cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_ADD).build();
    auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                       .setxDesc(after_mm)
                       .setbDesc(b)
                       .setyDesc(after_bias)
                       .setpwDesc(bias_desc)
                       .build();
    ops.push_back(std::move(bias_op));

    // Step 3
    auto act_desc =
        cudnn_frontend::PointWiseDescBuilder().setMode(CUDNN_POINTWISE_RELU_FWD).build();
    auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                      .setxDesc(after_bias)
                      .setyDesc(y)
                      .setpwDesc(act_desc)
                      .build();
    ops.push_back(std::move(act_op));

    // Create the graph
    std::vector<cudnn_frontend::Operation const*> all_ops;
    for (auto& op : ops) {
        all_ops.push_back(&op);
    }
    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                        .setHandle(handle)
                        .setOperationGraph(all_ops.size(), all_ops.data())
                        .build();

    cudnn_frontend::ExecutionPlanCache plan_cache("test_cache");
    execute_cached_plan(handle, plan_cache, op_graph, data_ptrs);
}