#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <stdlib.h>

#include "lib/memory.h"
#include "lib/utils.h"

using namespace lib;

int64_t hashString(std::string name) {
    auto x = std::hash<std::string>{}(name);
    return abs((int64_t)x);
}

static bool allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

cudnnStatus_t execute_cached_plan(cudnnHandle_t handle,
                                  cudnn_frontend::ExecutionPlanCache &plan_cache,
                                  cudnn_frontend::OperationGraph &opGraph,
                                  std::set<std::pair<uint64_t, void *>> &data_ptrs) {
    cudnnBackendDescriptor_t raw_plan;
    int64_t workspace_size = 0;

    cudnn_frontend::ExecutionPlan const *cached_plan;
    if (plan_cache.get_plan_from_cache(opGraph, cached_plan)) {
        std::cout << "Cached execution plan found." << cached_plan->getTag() << std::endl;
        workspace_size = cached_plan->getWorkspaceSize();
        raw_plan = cached_plan->get_raw_desc();
    } else {
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = cudnn_frontend::get_heuristics_list<1>(
            {"heuristics_mode_a"}, opGraph, ::allowAllConfig, filtered_configs, true);

        if (filtered_configs.size() == 0) {
            cudnn_frontend::set_error_and_throw_exception(
                nullptr, CUDNN_STATUS_NOT_SUPPORTED,
                "run_mha_fprop: No config returned by the heuristics");
        }

        auto plan_ = cudnn_frontend::ExecutionPlanBuilder()
                         .setHandle(handle)
                         .setEngineConfig(filtered_configs[0], opGraph.getTag())
                         .build();

        plan_cache.add_plan_to_cache(opGraph, plan_);
        workspace_size = plan_.getWorkspaceSize();
        raw_plan = plan_.get_raw_desc();
    }

    {  // Scope of workspace memory
        std::cout << "Workspace size: " << workspace_size << std::endl;
        memory::DeviceMemory<char> workspace_mem(workspace_size);
        auto variantPack = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_mem.get_ptr())
                               .setDataPointers(data_ptrs)
                               .build();
        cudnnStatus_t status = cudnnBackendExecute(handle, raw_plan, variantPack.get_raw_desc());
        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); },
                                 "Plan execute error", status);
        return status;
    }
}