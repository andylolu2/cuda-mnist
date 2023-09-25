#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/utils/macros.cuh"

using namespace cute;
using namespace cutlass;

template <typename T, typename Layout>
class DeviceTensor {
    using TensorA = Tensor<ViewEngine<gmem_ptr<T>>, Layout>;

   private:
    DeviceAllocation<T> data;
    TensorA tensor;

   public:
    // Constructors
    DeviceTensor(Layout layout)
        : data(size(layout)), tensor(make_tensor(make_gmem_ptr(data.get()), layout)) {}

    // Move constructor
    DeviceTensor(DeviceTensor &&other)
        : data(std::move(other.data)),
          tensor(make_tensor(make_gmem_ptr(data.get()), other.tensor.layout())) {}

    // Copy constructor
    DeviceTensor(const DeviceTensor &other)
        : data(size(other.tensor.layout())),
          tensor(make_tensor(make_gmem_ptr(data.get()), other.tensor.layout())) {}

    ~DeviceTensor() = default;

    TensorA &view() const { return tensor; }
    TensorA &view() { return tensor; }

    T *data_ptr() { return data.get(); }
};

template <
    typename T,
    typename Layout,
    __CUTE_REQUIRES(has_dereference<T>::value &&is_layout<Layout>::value)>
DeviceTensor<T, Layout> make_device_tensor(Layout layout) {
    return DeviceTensor<T, Layout>(layout);
}

template <
    typename T,
    typename Shape,
    __CUTE_REQUIRES(is_tuple<Shape>::value || is_integral<Shape>::value)>
DeviceTensor<T, Layout<Shape>> make_device_tensor(Shape shape) {
    return DeviceTensor<T, Layout<Shape>>(make_layout(shape));
}

template <
    typename T,
    typename Shape,
    typename Stride,
    __CUTE_REQUIRES(
        (is_tuple<Shape>::value || is_integral<Shape>::value) &&
        (is_tuple<Stride>::value || is_integral<Stride>::value))>
DeviceTensor<T, Layout<Shape, Stride>> make_device_tensor(Shape shape, Stride stride) {
    return DeviceTensor<T, Layout<Shape, Stride>>(make_layout(shape, stride));
}
