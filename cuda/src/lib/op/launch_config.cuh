#pragma once

#include <cute/tensor.hpp>

using namespace cute;

namespace lib {
    namespace op {
        const uint32_t N_THREADS = 128;  // 128 threads per block

        std::tuple<int, int> launch_config(int problem_size) {
            int n_blocks = ceil_div(problem_size, N_THREADS);
            return std::make_tuple(n_blocks, N_THREADS);
        }
    }  // namespace op
}  // namespace lib