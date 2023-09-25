#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <chrono>
#include <cute/tensor.hpp>
#include <iomanip>

#include "lib/dataset/data_loader.cuh"
#include "lib/modules/linear.cuh"
#include "lib/modules/mlp.cuh"
#include "lib/op/cross_entropy_with_logits.cuh"
#include "lib/op/reduce_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/utils/device_tensor.cuh"
#include "lib/utils/gpu_timer.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <path-to-mnist> steps hidden_size seed" << std::endl;
        return 1;
    }
    std::string mnist_path = argv[1];
    int steps = std::stoi(argv[2]);
    int hidden_size = argc > 3 ? std::stoi(argv[3]) : 128;
    int seed = argc > 4 ? std::stoi(argv[4]) : 0;
    int W = 28;
    int H = 28;
    int B = 32;
    int CLASSES = 10;
    std::vector<int> layer_sizes = {hidden_size, hidden_size, hidden_size, CLASSES};
    float learning_rate = 0.003f;

    auto loader = lib::mnist::DataLoader<half_t>(mnist_path, lib::mnist::Split::TRAIN, B);

    // Activations
    DeviceTensor y = make_device_tensor<half_t>(make_shape(B, CLASSES));
    DeviceTensor loss = make_device_tensor<half_t>(make_shape(B));
    DeviceTensor loss_scalar = make_device_tensor<half_t>(make_shape(_1{}));
    DeviceTensor dy = make_device_tensor<half_t>(make_shape(B, CLASSES));
    DeviceTensor dx = make_device_tensor<half_t>(make_shape(B, W * H));

    lib::module::MLP mlp(B, W * H, layer_sizes);
    mlp.init(seed);

    lib::utils::GpuTimer timer;
    timer.start();

    Tensor x = loader.get_batch_array();
    Tensor y_true = loader.get_batch_labels();

    for (int step = 1; step <= steps; ++step) {
        loader.next();  // Updates x and y_true

        mlp.forward(x, y.view());
        lib::op::cross_entropy_with_logits_bwd(y.view(), y_true, dy.view());
        mlp.backward(x, dy.view(), dx.view());

        if (step % 100 == 0) {
            lib::op::cross_entropy_with_logits_fwd(y.view(), y_true, loss.view());
            Tensor loss_expanded = lib::op::expand<1>(loss.view(), 1);  // (B) -> (B, 1)
            lib::op::mean<0>(loss_expanded, loss_scalar.view());        // (B, 1) -> (1)
            half_t loss_value = lib::utils::get_device_value(loss_scalar.data_ptr());
            std::cout << "Step: " << step << ", loss: " << loss_value << std::endl;
        }

        mlp.update(learning_rate);
    }

    timer.stop();
    const auto duration_s = timer.elapsed() / 1000.0f;
    std::cout << "Duration: " << duration_s << "s" << std::endl;

    return 0;
}