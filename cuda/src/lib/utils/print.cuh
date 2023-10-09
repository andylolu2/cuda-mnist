#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace utils {
        template <typename T>
        T get_device_value(T const *dev_ptr) {
            T host_value;
            device_memory::copy_to_host(&host_value, dev_ptr, 1);
            return host_value;
        }

        template <typename TensorA>
        void print_device_tensor(TensorA const &tensor) {
            using T = typename TensorA::value_type;

            std::vector<T> host_mem(cosize(tensor.layout()));
            Tensor host_tensor = make_tensor(host_mem.data(), tensor.layout());

            // Copy the data from device to host, one element at a time since the data might not be
            // contiguous.
            for (int i = 0; i < size(tensor); i++) {
                T *dev_ptr = (tensor.engine().begin() + tensor.layout()(i)).get();
                host_tensor(i) = get_device_value(dev_ptr);
            }
            std::cout << host_tensor << std::endl;
        }

        template <typename TensorA>
        void print_device_tensor(std::string name, TensorA const &tensor) {
            std::cout << name << ": ";
            print_device_tensor(tensor);
        }
    }  // namespace utils
}  // namespace lib