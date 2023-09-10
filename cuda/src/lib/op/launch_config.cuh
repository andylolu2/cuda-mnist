#pragma once

#include <cute/tensor.hpp>

using namespace cute;

namespace lib {
    namespace op {
        template <typename T>
        std::tuple<int, int> launch_config(T kernel, int problem_size) {
            int block_size;
            int min_grid_size;

            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);

            int grid_size = (problem_size + block_size - 1) / block_size;

            return std::make_tuple(grid_size, block_size);
        }
    }  // namespace op
}  // namespace lib