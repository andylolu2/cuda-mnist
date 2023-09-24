#pragma once

#include <cute/config.hpp>
#include <cute/tensor.hpp>

using namespace cute;

namespace lib {
    namespace op {
        template <int I1, int I2, typename TensorX>
        auto transpose(TensorX const& x) {
            auto new_layout_ = replace<I2>(x.layout(), get<I1>(x.layout()));
            auto new_layout = replace<I1>(new_layout_, get<I2>(x.layout()));

            return make_tensor(x.data(), new_layout);
        }

        template <int I, typename TensorX>
        auto expand(TensorX const& x, int n) {
            auto new_shape = insert<I>(x.layout().shape(), n);
            auto new_stride = insert<I>(x.layout().stride(), _0{});
            auto new_layout = make_layout(new_shape, new_stride);

            return make_tensor(x.data(), new_layout);
        }

        template <int I, typename TensorX>
        auto squeeze(TensorX const& x) {
            auto new_shape = remove<I>(x.layout().shape());
            auto new_stride = remove<I>(x.layout().stride());
            auto new_layout = make_layout(new_shape, new_stride);

            return make_tensor(x.data(), new_layout);
        }

        template <typename TensorX, typename NewLayout>
        auto reshape(TensorX const& x, NewLayout const& new_layout) {
            return x.compose(new_layout);
        }
    }  // namespace op
}  // namespace lib
