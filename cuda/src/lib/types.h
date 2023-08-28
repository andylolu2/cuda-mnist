#pragma once

#include <cudnn_frontend.h>

namespace lib {
    using shape = std::vector<int64_t>;
    using Ops = std::vector<cudnn_frontend::Operation const *>;
    using DataPtrs = std::set<std::pair<uint64_t, void *>>;
}  // namespace lib