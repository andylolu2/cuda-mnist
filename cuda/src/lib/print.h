#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <cute/util/print.hpp>

using namespace cute;
using namespace cutlass::device_memory;

namespace lib {
    template <typename Tensor>
    void print_device_tensor(Tensor &tensor) {
        using T = typename Tensor::value_type;

        std::vector<T> host_mem(tensor.size());

        copy_to_host(host_mem.data(), tensor.data().get(), tensor.size());

        std::cout << make_tensor(host_mem.data(), tensor.shape()) << std::endl;
    }

    template <typename Tensor>
    void print_device_tensor(std::string name, Tensor &tensor) {
        std::cout << name << ": ";
        print_device_tensor(tensor);
    }
}  // namespace lib