#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/op/pointwise_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;

int main(int argc, char** argv) {
    DeviceAllocation<half_t> data(24);
    DeviceAllocation<half_t> data_sum(12);

    Tensor x = make_tensor(make_gmem_ptr(data.get()), make_shape(_2{}, _3{}, _4{}));
    Tensor x_sum = make_tensor(make_gmem_ptr(data_sum.get()), make_shape(_3{}, _4{}));

    lib::op::normal(x);

    // std::cout << x << std::endl;

    // auto original_layout = make_layout(make_shape(2, 3, 4));
    // auto layout = make_layout(make_shape(2, 3, make_shape(2, 2)));
    // std::cout << layout << std::endl;
    // std::cout << flatten(layout) << std::endl;
    // std::cout << unflatten(x.layout(), layout.shape()) << std::endl;

    // std::cout << x.compose(make_layout(make_shape(_6{}, _4{}))) << std::endl;
    lib::utils::print_device_tensor(x);

    // std::cout << x.compose(group<0, 3>(x.layout())) << std::endl;

    lib::utils::print_device_tensor(lib::op::transpose<0, 2>(x));

    // auto expanded = lib::op::expand<0>(x, 2);
    // lib::utils::print_device_tensor(expanded);

    // lib::utils::print_device_tensor(lib::op::squeeze<0>(expanded));

    lib::op::sum<0>(x, x_sum);
    lib::utils::print_device_tensor(x_sum);

    Tensor test = make_tensor(
        make_gmem_ptr(data.get()),
        make_shape(make_shape(_32{}, _32{})),
        make_stride(make_stride(_32{}, _1{})));
    std::cout << test.layout() << std::endl;
    std::cout << upcast<16>(test.layout()) << std::endl;

    // std::cout << coalesce(x, layout) << std::endl;

    // Tensor y = coalesce(x);

    // std::cout << y << std::endl;

    return 0;
}