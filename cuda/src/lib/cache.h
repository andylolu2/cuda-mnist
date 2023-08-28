#pragma once
#include <cudnn_frontend.h>

#include <string>
#include <unordered_map>

namespace lib {
    class PlanCacheManager {
       public:
        PlanCacheManager();
        ~PlanCacheManager();

        cudnn_frontend::ExecutionPlanCache& get_plan_cache(const std::string& name);

       private:
        std::unordered_map<std::string, std::shared_ptr<cudnn_frontend::ExecutionPlanCache>>
            plan_cache_map;
    };
}  // namespace lib