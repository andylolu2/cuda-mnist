#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <iomanip>

#include "lib/dataset/data_loader.hpp"
#include "lib/fill.h"
#include "lib/matmul_bias_bwd.cuh"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/modules/linear.cuh"
#include "lib/modules/mlp.cuh"
#include "lib/op/cross_entropy_with_logits.cuh"
#include "lib/print.h"
#include "lib/tensor_ops.cuh"

#define W 28
#define H 28
#define B 32
#define CLASSES 10

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path-to-mnist>" << std::endl;
        return 1;
    }
    std::string mnist_path = argv[1];

    // Load MNIST dataset
    auto loader = lib::mnist::DataLoader<half_t>(mnist_path, lib::mnist::Split::TRAIN, B);

    // Data on device
    DeviceAllocation<half_t> x_data(B * W * H);
    Tensor x = make_tensor(make_gmem_ptr(x_data.get()), make_shape(B, W * H));
    DeviceAllocation<int> y_true_data(B);
    Tensor y_true = make_tensor(make_gmem_ptr(y_true_data.get()), make_shape(B));

    // Activations on device
    DeviceAllocation<half_t> y_data(B * CLASSES);
    Tensor y = make_tensor(make_gmem_ptr(y_data.get()), make_shape(B, CLASSES));
    DeviceAllocation<half_t> loss_data(B);
    Tensor loss = make_tensor(make_gmem_ptr(loss_data.get()), make_shape(B));
    DeviceAllocation<half_t> loss_scalar_data(1);
    Tensor loss_scalar = make_tensor(make_gmem_ptr(loss_scalar_data.get()), make_shape(1));
    DeviceAllocation<float> dy_data(B * CLASSES);
    Tensor dy = make_tensor(make_gmem_ptr(dy_data.get()), make_shape(B, CLASSES));
    DeviceAllocation<float> dx_data(B * W * H);
    Tensor dx = make_tensor(make_gmem_ptr(dx_data.get()), make_shape(B, W * H));

    lib::module::MLP<half_t, float, half_t> mlp(W * H, {512, 256, CLASSES}, B);
    mlp.init();

    // Forward pass
    for (int step = 0; step < 5000; step++) {
        // Load data in host
        auto [image_host, label_host] = loader.next();

        // Copy data to device
        x_data.copy_from_host(image_host.data());
        y_true_data.copy_from_host(label_host.data());

        mlp.forward(x, y);

        lib::op::cross_entropy_with_logits_fwd(y, y_true, loss);
        Tensor loss_expanded = lib::op::expand<1>(loss, 1);  // (B) -> (B, 1)
        lib::op::mean<0>(loss_expanded, loss_scalar);        // (B, 1) -> (1)
        lib::print_device_tensor("loss", loss_scalar);

        lib::op::cross_entropy_with_logits_bwd(y, y_true, dy);
        mlp.backward(x, dy, dx);

        mlp.update(0.003);
        mlp.clear_grad();
    }

    return 0;
}