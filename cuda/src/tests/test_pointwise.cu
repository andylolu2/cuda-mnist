#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/op/pointwise_ops.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    int M = 4;
    int N = 4;

    using T = half_t;

    DeviceAllocation<T> x_data(M * N);
    DeviceAllocation<T> y_data(M * N);

    Tensor x = make_tensor(make_gmem_ptr(x_data.get()), make_shape(M, N));
    Tensor y = make_tensor(make_gmem_ptr(y_data.get()), make_shape(M, N));

    lib::op::constant(x, 2);
    lib::utils::print_device_tensor("x=2", x);

    lib::op::normal(y);
    lib::utils::print_device_tensor("y~N(0,1)", y);

    lib::op::relu(y, y);
    lib::utils::print_device_tensor("relu(y)", y);

    lib::op::add(y, x, y);
    lib::utils::print_device_tensor("relu(y)+x", y);

    return 0;
}