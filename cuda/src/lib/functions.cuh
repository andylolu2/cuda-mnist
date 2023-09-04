#pragma once
#include <cute/config.hpp>

namespace lib {
    namespace func {
        struct Identity {
            template <typename T>
            CUTE_HOST_DEVICE T operator()(T &x) {
                return x;
            }
        };
        struct ReLU {
            template <typename T>
            CUTE_HOST_DEVICE T operator()(T &x) {
                return x > T(0) ? x : T(0);
            }
        };
        struct dReLU {
            template <typename T>
            CUTE_HOST_DEVICE T operator()(T &x) {
                return x > T(0) ? T(1) : T(0);
            }
        };
    }  // namespace func
}  // namespace lib