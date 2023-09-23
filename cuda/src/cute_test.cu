#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <chrono>
#include <cute/tensor.hpp>
#include <iomanip>

#include "lib/dataset/data_loader.hpp"
#include "lib/modules/linear.cuh"
#include "lib/modules/mlp.cuh"
#include "lib/op/cross_entropy_with_logits.cuh"
#include "lib/op/reduce_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/utils/gpu_timer.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <path-to-mnist> steps" << std::endl;
        return 1;
    }
    std::string mnist_path = argv[1];
    int steps = std::stoi(argv[2]);
    int W = 28;
    int H = 28;
    int B = 32;
    int CLASSES = 10;
    std::vector<int> layer_sizes = {128, 128, 128, CLASSES};
    float learning_rate = 0.003f;

    // Load MNIST dataset
    auto loader = lib::mnist::DataLoader<half_t>(mnist_path, lib::mnist::Split::TRAIN, B);

    // Activations on device
    DeviceAllocation<half_t> y_data(B * CLASSES);
    Tensor y = make_tensor(make_gmem_ptr(y_data.get()), make_shape(B, CLASSES));
    DeviceAllocation<half_t> loss_data(B);
    Tensor loss = make_tensor(make_gmem_ptr(loss_data.get()), make_shape(B));
    DeviceAllocation<half_t> loss_scalar_data(1);
    Tensor loss_scalar = make_tensor(make_gmem_ptr(loss_scalar_data.get()), make_shape(_1{}));
    DeviceAllocation<half_t> dy_data(B * CLASSES);
    Tensor dy = make_tensor(make_gmem_ptr(dy_data.get()), make_shape(B, CLASSES));
    DeviceAllocation<half_t> dx_data(B * W * H);
    Tensor dx = make_tensor(make_gmem_ptr(dx_data.get()), make_shape(B, W * H));

    lib::module::MLP mlp(B, W * H, layer_sizes);
    mlp.init();

    lib::utils::GpuTimer timer;
    timer.start();

    Tensor x = loader.get_batch_array();
    Tensor y_true = loader.get_batch_labels();

    for (int step = 1; step <= steps; ++step) {
        loader.next();  // Update x and y_true

        mlp.forward(x, y);
        lib::op::cross_entropy_with_logits_bwd(y, y_true, dy);
        mlp.backward(x, dy, dx);

        if (step % 500 == 0) {
            std::cout << "Step: " << step << std::endl;
            lib::op::cross_entropy_with_logits_fwd(y, y_true, loss);
            Tensor loss_expanded = lib::op::expand<1>(loss, 1);  // (B) -> (B, 1)
            lib::op::mean<0>(loss_expanded, loss_scalar);        // (B, 1) -> (1)

            lib::utils::print_device_tensor("loss", loss_scalar);
        }

        mlp.update(learning_rate);
        mlp.clear_grad();
    }

    timer.stop();
    const auto duration_s = timer.elapsed() / 1000.0f;
    std::cout << "Duration: " << duration_s << "s" << std::endl;

    return 0;
}