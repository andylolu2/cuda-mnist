#include <cutlass/numeric_types.h>
#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/fill.h"
#include "lib/matmul_bias_bwd.cuh"
#include "lib/matmul_bias_pointwise.cuh"
#include "lib/module.cuh"
#include "lib/print.h"
#include "lib/tensor_ops.cuh"

using namespace cute;

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        printf("Usage: %s M N K\n", argv[0]);
        return 0;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    cutlass::DeviceAllocation<cutlass::half_t> x_data(M * K);
    Tensor x = make_tensor(make_gmem_ptr(x_data.get()), make_shape(M, K));

    lib::module::Linear<cutlass::half_t, cutlass::half_t, float> linear(K, N);

    // cutlass::DeviceAllocation<cutlass::half_t> w_data(N * K);
    // Tensor w = make_tensor(make_gmem_ptr(w_data.get()), make_shape(N, K));

    // cutlass::DeviceAllocation<cutlass::half_t> b_data(N);
    // Tensor b = make_tensor(make_gmem_ptr(b_data.get()), make_shape(N));
    // Tensor b_expanded =
    //     make_tensor(make_gmem_ptr(b_data.get()), make_shape(M, N), make_stride(0, 1));

    cutlass::DeviceAllocation<cutlass::half_t> y_data(M * N);
    Tensor y = make_tensor(make_gmem_ptr(y_data.get()), make_shape(M, N));

    cutlass::DeviceAllocation<float> y_sum_data(1);
    Tensor y_sum = make_tensor(make_gmem_ptr(y_sum_data.get()), make_shape(1));
    Tensor y_sum_expanded =
        make_tensor(make_gmem_ptr(y_sum_data.get()), make_shape(M, N), make_stride(0, 0));

    cutlass::DeviceAllocation<float> dy_sum_data(1);
    Tensor dy_sum = make_tensor(make_gmem_ptr(dy_sum_data.get()), make_shape(1));
    Tensor dy_sum_expanded =
        make_tensor(make_gmem_ptr(dy_sum_data.get()), make_shape(M, N), make_stride(0, 0));

    cutlass::DeviceAllocation<cutlass::half_t> dy_data(M * N);
    Tensor dy = make_tensor(make_gmem_ptr(dy_data.get()), make_shape(M, N));

    cutlass::DeviceAllocation<cutlass::half_t> dx_data(M * K);
    Tensor dx = make_tensor(make_gmem_ptr(dx_data.get()), make_shape(M, K));

    cutlass::DeviceAllocation<int> y_true_data(M);
    Tensor y_true = make_tensor(make_gmem_ptr(y_true_data.get()), make_shape(M));

    // cutlass::DeviceAllocation<cutlass::half_t> dw_data(N * K);
    // Tensor dw = make_tensor(make_gmem_ptr(dw_data.get()), make_shape(N, K));

    // cutlass::DeviceAllocation<cutlass::half_t> db_data(N);
    // Tensor db = make_tensor(make_gmem_ptr(db_data.get()), make_shape(N));
    // Tensor db_expanded =
    //     make_tensor(make_gmem_ptr(db_data.get()), make_shape(M, N), make_stride(0, 1));

    lib::init::arange<<<1, 64>>>(x, -1.0_hf, 0.1_hf);
    lib::init::arange<<<1, 64>>>(y_true);

    lib::print_device_tensor("Tensor y sum", y_sum);
    lib::print_device_tensor("Tensor x", x);

    std::cout << linear << std::endl;

    // Forward pass
    // lib::op::matmul_bias(x, w, b_expanded, y);
    linear.init();
    lib::print_device_tensor("Tensor w", linear.weight());
    lib::print_device_tensor("Tensor b", linear.bias());

    linear.forward(x, y);
    lib::op::sum<<<1, 64>>>(y, y_sum_expanded);

    lib::print_device_tensor("Tensor y", y);
    lib::print_device_tensor("Tensor y sum", y_sum);

    // Set loss
    lib::init::constant<<<1, 64>>>(dy_sum, 1.0f);

    // Backward pass
    lib::op::sum_bwd<<<1, 64>>>(dy, dy_sum_expanded);
    linear.backward(x, dy, dx);
    // lib::op::matmul_bias_bwd(x, w, b_expanded, dy, dx, dw, db_expanded);

    lib::op::cross_entropy_with_logits_bwd<1><<<1, 64>>>(y, y_true, dy);

    lib::print_device_tensor("y_true", y_true);
    lib::print_device_tensor("dy", dy);
    lib::print_device_tensor("dx", dx);
    lib::print_device_tensor("dw", linear.weight_grad());
    lib::print_device_tensor("db", linear.biad_grad());

    return 0;
}