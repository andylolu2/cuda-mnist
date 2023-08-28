#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <stdlib.h>

#include "lib/utils.h"

using namespace lib;

int64_t hashString(std::string name) {
    auto x = std::hash<std::string>{}(name);
    return abs((int64_t)x);
}

size_t cudnn_dtype_size(cudnnDataType_t dtype) {
    switch (dtype) {
        case CUDNN_DATA_FLOAT:
            return sizeof(float);
        case CUDNN_DATA_HALF:
            return sizeof(half);
        case CUDNN_DATA_BFLOAT16:
            return sizeof(half);
        case CUDNN_DATA_INT8:
            return sizeof(int8_t);
        case CUDNN_DATA_INT32:
            return sizeof(int32_t);
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

std::string cudnn_value_to_str(void *ptr, cudnnDataType_t dtype) {
    std::stringstream ss;
    switch (dtype) {
        case CUDNN_DATA_FLOAT:
            ss << *(float *)ptr;
            break;
        case CUDNN_DATA_HALF:
            ss << __half2float(*(half *)ptr);
            break;
        case CUDNN_DATA_BFLOAT16:
            ss << __half2float(*(half *)ptr);
            break;
        case CUDNN_DATA_INT8:
            ss << *(int8_t *)ptr;
            break;
        case CUDNN_DATA_INT32:
            ss << *(int32_t *)ptr;
            break;
        default:
            throw std::runtime_error("Unsupported data type");
    }
    return ss.str();
}

static bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

cudnnStatus_t execute_ops(cudnnHandle_t handle, cudnn_frontend::ExecutionPlanCache &plan_cache,
                          Ops &ops, DataPtrs &data_ptrs) {
    cudnnBackendDescriptor_t raw_plan;
    int64_t workspace_size = 0;

    auto op_graph = cudnn_frontend::OperationGraphBuilder()
                        .setHandle(handle)
                        .setOperationGraph(ops.size(), ops.data())
                        .build();

    cudnn_frontend::ExecutionPlan const *cached_plan;
    if (plan_cache.get_plan_from_cache(op_graph, cached_plan)) {
        std::cout << "Cached execution plan found." << cached_plan->getTag() << std::endl;
        workspace_size = cached_plan->getWorkspaceSize();
        raw_plan = cached_plan->get_raw_desc();
    } else {
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<1>(
            {"heuristics_mode_a"}, op_graph, ::allowAllConfig, filtered_configs, true);

        if (filtered_configs.size() == 0) {
            cudnn_frontend::set_error_and_throw_exception(nullptr, CUDNN_STATUS_NOT_SUPPORTED,
                                                          "No config returned by the heuristics");
        }

        auto plan = cudnn_frontend::ExecutionPlanBuilder()
                        .setHandle(handle)
                        .setEngineConfig(filtered_configs[0], op_graph.getTag())
                        .build();

        std::cout << "New execution plan created." << plan.getTag() << std::endl;

        plan_cache.add_plan_to_cache(op_graph, plan);
        workspace_size = plan.getWorkspaceSize();
        raw_plan = plan.get_raw_desc();
    }

    std::cout << "Workspace size: " << workspace_size << std::endl;
    void *workspace_ptr = nullptr;
    check_cuda_status(cudaMalloc(&workspace_ptr, workspace_size));
    auto variantPack = cudnn_frontend::VariantPackBuilder()
                           .setWorkspacePointer(workspace_ptr)
                           .setDataPointers(data_ptrs)
                           .build();
    cudnnStatus_t status = cudnnBackendExecute(handle, raw_plan, variantPack.get_raw_desc());
    check_cuda_status(cudaFree(workspace_ptr));

    cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); },
                             "Plan execute error", status);
    return status;
}