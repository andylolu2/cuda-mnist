#pragma once
#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

using namespace cute;

template <
    typename T,
    typename LayoutArg,
    typename... LayoutArgs,
    __CUTE_REQUIRES(not is_layout<LayoutArg>::value)>
class DeviceTensor {
    using Tensor = Tensor < ViewEngine<gmem_ptr<T>> using T = Engine::value_type;

   private:
    Tensor tensor;
    cutlass::DeviceAllocation<T> device_memory;

   public:
    DeviceTensor(size_t size, 
    ~DeviceTensor();
};

DeviceTensor::DeviceTensor(/* args */) {}

DeviceTensor::~DeviceTensor() {}
