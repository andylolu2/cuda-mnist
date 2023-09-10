#pragma once

#include <cute/algorithm/functional.hpp>
#include <cute/config.hpp>
#include <cute/tensor.hpp>

#include "lib/op/unary_pointwise.cuh"

using namespace cute;

namespace lib {
    namespace op {
        /**
         * Sets all elements of the tensor to the given value.
         */
        template <typename Engine, typename Layout, typename T = typename Engine::value_type>
        void constant(Tensor<Engine, Layout> tensor, T value = T(0)) {
            cute::constant_fn<T> op{value};
            unary_pointwise(tensor, tensor, op);
        }
    }  // namespace op
}  // namespace lib