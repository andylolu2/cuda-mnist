#pragma once

using shape = std::vector<int64_t>;
using Ops = std::vector<cudnn_frontend::Operation>;
using DataPtrs = std::set<std::pair<uint64_t, void *>>;