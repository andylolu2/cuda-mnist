#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/fill.h"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/print.h"
#include "lib/utils/gpu_timer.cuh"

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

    using T = half_t;

    DeviceAllocation<T> a_data(M * K);
    DeviceAllocation<T> b_data(K * N);
    DeviceAllocation<T> c_data(M * N);
    DeviceAllocation<T> d_data(M * N);
    Tensor a = make_tensor(make_gmem_ptr(a_data.get()), make_shape(M, K));
    Tensor b = make_tensor(make_gmem_ptr(b_data.get()), make_shape(N, K));
    Tensor c = make_tensor(make_gmem_ptr(c_data.get()), make_shape(M, N));
    Tensor d = make_tensor(make_gmem_ptr(d_data.get()), make_shape(M, N));

    lib::init::arange<<<1, 64>>>(a, T(0), T(1.0f / static_cast<float>(M * K)));
    lib::init::arange<<<1, 64>>>(b, T(0), T(1.0f / static_cast<float>(N * K)));
    lib::init::arange<<<1, 64>>>(c, T(0), T(1.0f / static_cast<float>(M * N)));

    std::vector<float> times;

    for (int i = 0; i < n; ++i) {
        lib::utils::GpuTimer timer;

        timer.start();
        lib::gemm(
            M,
            N,
            K,
            a_data.get(),
            a.stride(),
            b_data.get(),
            b.stride(),
            c_data.get(),
            c.stride(),
            d_data.get(),
            d.stride());
        // lib::op::matmul(a, b, c);
        timer.stop();

        times.push_back(timer.elapsed());
    }

    float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    float tflops = 2.0f * M * N * K * 1e-12f / (mean * 1e-3f);

    std::cout << "Time: " << mean << " ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;

    return 0;
}