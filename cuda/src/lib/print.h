#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

using namespace cute;
using namespace cutlass::device_memory;

namespace lib {
    template <typename Engine, typename Layout>
    void print_device_tensor(Tensor<Engine, Layout> const &tensor) {
        using T = typename Engine::value_type;

        std::vector<T> host_mem(tensor.size());

        copy_to_host(host_mem.data(), tensor.data().get(), tensor.size());

        std::cout << make_tensor(host_mem.data(), tensor.layout()) << std::endl;
    }

    template <typename Engine, typename Layout>
    void print_device_tensor(std::string name, Tensor<Engine, Layout> const &tensor) {
        std::cout << name << ": ";
        print_device_tensor(tensor);
    }
}  // namespace lib