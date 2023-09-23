#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/op/cross_entropy_with_logits.cuh"
#include "lib/op/pointwise_ops.cuh"
#include "lib/op/reduce_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " B C" << std::endl;
        return 1;
    }

    int B = atoi(argv[1]);
    int C = atoi(argv[2]);
    bool print_tensors = true;

    DeviceAllocation<half_t> y_data(B * C);
    DeviceAllocation<half_t> dy_data(B * C);
    DeviceAllocation<int> y_true_data(B);
    DeviceAllocation<half_t> loss_data(B);
    DeviceAllocation<half_t> loss_scalar_data(1);

    Tensor y = make_tensor(make_gmem_ptr(y_data.get()), make_shape(B, C));
    Tensor dy = make_tensor(make_gmem_ptr(dy_data.get()), make_shape(B, C));
    Tensor y_true = make_tensor(make_gmem_ptr(y_true_data.get()), make_shape(B));
    Tensor loss = make_tensor(make_gmem_ptr(loss_data.get()), make_shape(B));
    Tensor loss_scalar = make_tensor(make_gmem_ptr(loss_scalar_data.get()), make_shape(1));

    lib::op::arange(y, 0_hf, half_t(1.0f / float(size(y))));
    lib::op::arange(y_true);

    for (int i = 0; i < 2; i++) {
        std::cout << "Iteration " << i << std::endl;
        lib::utils::print_device_tensor("y", y);
        lib::utils::print_device_tensor("y_true", y_true);

        lib::op::cross_entropy_with_logits_bwd(y, y_true, dy);
        lib::utils::print_device_tensor("dy", dy);

        lib::op::cross_entropy_with_logits_fwd(y, y_true, loss);
        Tensor loss_expanded = lib::op::expand<1>(loss, 1);  // (B) -> (B, 1)
        lib::op::mean<0>(loss_expanded, loss_scalar);        // (B, 1) -> (1)
        lib::utils::print_device_tensor("loss", loss_scalar);
    }
    return 0;
}