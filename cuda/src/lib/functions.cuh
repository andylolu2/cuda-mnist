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
            template <typename TA, typename TB>
            CUTE_HOST_DEVICE TA operator()(TA &dy, TB &x) {
                return TA(x > TB(0) ? dy : 0);
            }
        };
    }  // namespace func
}  // namespace lib