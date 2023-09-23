#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/op/gemm.cuh"
#include "lib/op/pointwise_ops.cuh"
#include "lib/utils/gpu_timer.cuh"
#include "lib/utils/macros.cuh"
#include "lib/utils/print.cuh"

using namespace cute;
using namespace cutlass;

int main(int argc, char const* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " M N K T" << std::endl;
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int n = atoi(argv[4]);
    bool print_tensors = (n <= 1 && M <= 16 && N <= 16);

    using T = half_t;

    DeviceAllocation<T> a_data(M * K);
    DeviceAllocation<T> b_data(K * N);
    DeviceAllocation<T> c_data(M * N);
    DeviceAllocation<T> d_data(M * N);
    Tensor a = make_tensor(make_gmem_ptr(a_data.get()), make_shape(M, K));
    Tensor b = make_tensor(make_gmem_ptr(b_data.get()), make_shape(K, N));
    Tensor c = make_tensor(make_gmem_ptr(c_data.get()), make_shape(M, N));
    Tensor d = make_tensor(make_gmem_ptr(d_data.get()), make_shape(M, N));

    lib::op::arange(a, static_cast<T>(0), T(1.0f / static_cast<float>(M * K)));
    lib::op::arange(b, static_cast<T>(0), T(1.0f / static_cast<float>(N * K)));
    lib::op::constant(c, static_cast<T>(1));
    lib::op::constant(d);

    if (print_tensors) {
        lib::utils::print_device_tensor(a);
        lib::utils::print_device_tensor(b);
        lib::utils::print_device_tensor(c);
        lib::utils::print_device_tensor(d);
    }

    std::vector<float> times;

    DeviceAllocation<uint8_t> workspace;
    auto gemm_op = lib::op::gemm<128>(a, b, c, d, workspace);

    for (int i = 0; i < n; ++i) {
        lib::utils::GpuTimer timer;

        timer.start();
        CUTLASS_CHECK(gemm_op());

        if (print_tensors) {
            lib::utils::print_device_tensor(d);
        }
        timer.stop();

        times.push_back(timer.elapsed());
    }

    float mean_s = std::accumulate(times.begin(), times.end(), 0.0f) / times.size() / 1000.0f;
    float tflops = 2.0f * M * N * K / 1e12f / mean_s;

    std::cout << "Time: " << mean_s << " s" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;

    return 0;
}