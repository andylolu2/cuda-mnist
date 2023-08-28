#include <cudnn_frontend.h>

#include <string>
#include <unordered_map>

#include "lib/cache.h"

namespace lib {
    PlanCacheManager::PlanCacheManager() = default;
    PlanCacheManager::~PlanCacheManager() = default;

    cudnn_frontend::ExecutionPlanCache& PlanCacheManager::get_plan_cache(const std::string& name) {
        if (plan_cache_map.find(name) == plan_cache_map.end()) {
            plan_cache_map[name] =
                std::make_shared<cudnn_frontend::ExecutionPlanCache>(name.c_str());
        }
        return *plan_cache_map[name];
    }
}  // namespace lib