#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/modules/linear.cuh"
#include "lib/modules/mlp.cuh"
#include "lib/op/pointwise_ops.cuh"
#include "lib/utils/gpu_timer.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;
using std::string_literals::operator""s;

int main(int argc, char const* argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " B D1 D2 D3 T" << std::endl;
        return 1;
    }

    int B = atoi(argv[1]);
    int D1 = atoi(argv[2]);
    int D2 = atoi(argv[3]);
    int D3 = atoi(argv[4]);
    int n = atoi(argv[5]);
    bool print_tensors = (n <= 1 && B <= 16 && D1 <= 16 && D2 <= 16 && D3 <= 16);

    using T = half_t;

    DeviceAllocation<T> x_data(B * D1);
    DeviceAllocation<T> y_data(B * D3);
    DeviceAllocation<T> dx_data(B * D1);
    DeviceAllocation<T> dy_data(B * D3);

    Tensor x = make_tensor(make_gmem_ptr(x_data.get()), make_shape(B, D1));
    Tensor y = make_tensor(make_gmem_ptr(y_data.get()), make_shape(B, D3));
    Tensor dx = make_tensor(make_gmem_ptr(dx_data.get()), make_shape(B, D1));
    Tensor dy = make_tensor(make_gmem_ptr(dy_data.get()), make_shape(B, D3));

    lib::module::MLP mlp(B, D1, {D2, D3});
    mlp.init(0, "arange");

    lib::op::arange(x, T(0), T(1.0f / float(size(x))));
    lib::op::arange(dy, T(0), T(1.0f / float(size(dy))));

    if (print_tensors) {
        lib::utils::print_device_tensor("x", x);
        for (size_t i = 0; i < mlp.n_layers(); ++i) {
            lib::utils::print_device_tensor(std::to_string(i) + ".w"s, mlp[i].weight());
            lib::utils::print_device_tensor(std::to_string(i) + ".b"s, mlp[i].bias());
        }
    }

    lib::utils::GpuTimer timer;
    timer.start();

    for (int i = 0; i < n; ++i) {
        mlp.forward(x, y);
        mlp.backward(x, dy, dx);

        if (print_tensors) {
            lib::utils::print_device_tensor("y", y);
            lib::utils::print_device_tensor("dy", dy);
            for (size_t i = 0; i < mlp.n_layers(); ++i) {
                lib::utils::print_device_tensor(std::to_string(i) + ".dw"s, mlp[i].weight_grad());
                lib::utils::print_device_tensor(std::to_string(i) + ".db"s, mlp[i].bias_grad());
            }
            lib::utils::print_device_tensor("dx", dx);
        }
    }

    timer.stop();
    float mean_s = timer.elapsed() / n / 1000.0f;
    float tflops = mlp.tflops() / mean_s;

    std::cout << "Time: " << mean_s << " s" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;

    return 0;
}