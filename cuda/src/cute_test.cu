#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <chrono>
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
#define B 16
#define CLASSES 10

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <path-to-mnist> steps" << std::endl;
        return 1;
    }
    std::string mnist_path = argv[1];
    int steps = std::stoi(argv[2]);

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
    DeviceAllocation<half_t> dy_data(B * CLASSES);
    Tensor dy = make_tensor(make_gmem_ptr(dy_data.get()), make_shape(B, CLASSES));
    DeviceAllocation<half_t> dx_data(B * W * H);
    Tensor dx = make_tensor(make_gmem_ptr(dx_data.get()), make_shape(B, W * H));

    lib::module::MLP mlp(B, W * H, {128, CLASSES});
    mlp.init();

    const auto start_time = std::chrono::system_clock::now();

    loader.next();
    x_data.copy_from_host(loader.get_image_data());
    y_true_data.copy_from_host(loader.get_label_data());

    // Forward pass
    for (int step = 0; step < steps; step++) {
        // Load data in host
        // loader.next();

        // Copy data to device
        // x_data.copy_from_host(loader.get_image_data());
        // y_true_data.copy_from_host(loader.get_label_data());

        mlp.forward(x, y);
        lib::op::cross_entropy_with_logits_bwd(y, y_true, dy);
        mlp.backward(x, dy, dx);

        if (step % 100 == 0) {
            // lib::print_device_tensor("dy", dy);
            lib::op::cross_entropy_with_logits_fwd(y, y_true, loss);
            Tensor loss_expanded = lib::op::expand<1>(loss, 1);  // (B) -> (B, 1)
            lib::op::mean<0>(loss_expanded, loss_scalar);        // (B, 1) -> (1)
            lib::print_device_tensor("loss", loss_scalar);
            // lib::print_device_tensor("y", y);
            // lib::print_device_tensor("1.dw", mlp[1].weight_grad());
            // lib::print_device_tensor("1.db", mlp[1].bias_grad());
        }

        mlp.update(0.003f);
        mlp.clear_grad();
    }

    const auto end_time = std::chrono::system_clock::now();
    const auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    const auto duration_s = duration_ms / 1000.0;
    std::cout << "Duration: " << duration_s << "s" << std::endl;

    return 0;
}