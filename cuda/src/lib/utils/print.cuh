#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

using namespace cute;
using namespace cutlass::device_memory;

namespace lib {
    namespace utils {
        template <typename Engine, typename Layout>
        void print_device_tensor(Tensor<Engine, Layout> const &tensor) {
            using T = typename Engine::value_type;

            std::vector<T> host_mem(size(tensor));
            Tensor host_tensor = make_tensor(host_mem.data(), tensor.shape());

            // Copy the data from device to host, one element at a time since the data might not be
            // contiguous.
            for (int i = 0; i < size(tensor); i++) {
                T *dev_ptr = (tensor.engine().begin() + tensor.layout()(i)).get();
                T *host_ptr = host_tensor.engine().begin() + host_tensor.layout()(i);
                copy_to_host(host_ptr, dev_ptr, 1);
            }

            std::cout << "Shape: " << tensor.shape() << " "
                      << "Stride: " << tensor.stride() << " ";
            std::cout << host_tensor << std::endl;
        }

        template <typename Engine, typename Layout>
        void print_device_tensor(std::string name, Tensor<Engine, Layout> const &tensor) {
            std::cout << name << ": ";
            print_device_tensor(tensor);
        }
    }  // namespace utils
}  // namespace lib