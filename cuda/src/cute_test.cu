#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <iomanip>

#include "lib/dataset/mnist_reader.hpp"
#include "lib/fill.h"
#include "lib/matmul_bias_bwd.cuh"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/module.cuh"
#include "lib/op/cross_entropy_with_logits.cuh"
#include "lib/print.h"
#include "lib/tensor_ops.cuh"

#define W 28
#define H 28
#define B 16
#define CLASSES 10

using namespace cute;
using namespace cutlass;

int main(int argc, char const *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path-to-mnist>" << std::endl;
        return 1;
    }
    std::string mnist_path = argv[1];

    // Load MNIST dataset
    auto dataset = lib::mnist::read_dataset<std::vector, std::vector, int>(mnist_path);

    // Data on host
    std::vector<half_t> x_host_data(B * W * H);
    Tensor x_host = make_tensor(x_host_data.data(), make_shape(B, W * H));
    std::vector<int> y_true_host_data(B);
    Tensor y_true_host = make_tensor(y_true_host_data.data(), make_shape(B));

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

    lib::module::Linear<half_t, float> linear(W * H, CLASSES);
    linear.init();

    // Forward pass
    for (int step = 0; step < 2000; step++) {
        // Load data in host
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < W * H; j++) {
                auto batch_idx = (B * step + i) % dataset.training_images.size();
                x_host(i, j) = dataset.training_images[i][j] / 255.0_hf;
            }
        }
        for (int i = 0; i < B; i++) {
            y_true_host(i) = static_cast<int>(dataset.training_labels[i]);
        }
        // Copy data to device
        x_data.copy_from_host(x_host_data.data());
        y_true_data.copy_from_host(y_true_host_data.data());

        linear.forward(x, y);
        lib::op::cross_entropy_with_logits_fwd(y, y_true, loss);
        Tensor loss_expanded = lib::op::expand<1>(loss, 1);  // (B) -> (B, 1)
        lib::op::mean<0>(loss_expanded, loss_scalar);        // (B, 1) -> (1)
        lib::print_device_tensor("loss", loss_scalar);

        lib::op::cross_entropy_with_logits_bwd(y, y_true, dy);
        linear.backward(x, dy, dx);

        // lib::print_device_tensor("db", linear.bias_grad());
        // lib::print_device_tensor("dw", linear.weight_grad());

        linear.update(0.01);
        linear.clear_grad();
    }

    // linear.forward(x, y);

    // lib::print_device_tensor("y_true", y_true);

    // Set loss

    // lib::print_device_tensor("dy", dy);

    return 0;
}