#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/op/pointwise_ops.cuh"
#include "lib/utils/gpu_timer.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " M N n" << std::endl;
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);
    bool print_tensors = (n <= 2 && M <= 16 && N <= 16);

    using T = half_t;

    DeviceAllocation<T> x_data(M * N);
    DeviceAllocation<T> y_data(M * N);

    Tensor x = make_tensor(make_gmem_ptr(x_data.get()), make_shape(M, N));
    lib::op::normal(x);
    Tensor y = make_tensor(make_gmem_ptr(y_data.get()), make_shape(M, N));

    lib::utils::GpuTimer timer;
    timer.start();

    for (int i = 0; i < n; ++i) {
        lib::op::relu(y, x);
        if (print_tensors) {
            lib::utils::print_device_tensor("x", x);
            lib::utils::print_device_tensor("y", y);
        }
    }

    timer.stop();
    float mean_s = timer.elapsed() / n / 1000.0f;
    float tflops = M * N * 1e-12 / mean_s;

    std::cout << "Time: " << mean_s << "s" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;

    return 0;
}