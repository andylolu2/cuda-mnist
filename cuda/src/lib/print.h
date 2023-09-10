#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

using namespace cute;
using namespace cutlass::device_memory;

namespace lib {
    template <typename Engine, typename Layout>
    void print_device_tensor(Tensor<Engine, Layout> const &tensor) {
        using T = typename Engine::value_type;

        std::vector<T> host_mem(size(tensor));

        for (size_t i = 0; i < host_mem.size(); ++i) {
            T *dev_ptr = (tensor.engine().begin() + tensor.layout()(i)).get();
            copy_to_host(host_mem.data() + i, dev_ptr, 1);
        }

        std::cout << make_tensor(host_mem.data(), tensor.layout()) << std::endl;
    }

    template <typename Engine, typename Layout>
    void print_device_tensor(std::string name, Tensor<Engine, Layout> const &tensor) {
        std::cout << name << ": ";
        print_device_tensor(tensor);
    }
}  // namespace lib