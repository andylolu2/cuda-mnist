#include <cuda_fp16.h>
#include <cudnn_frontend.h>

#include "lib/memory.h"
#include "lib/methods/matmul_bias_relu_fwd.h"
#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"

using namespace std::string_literals;
using namespace lib;

int main() {
    int64_t B = 32;
    int64_t D_in = 128;
    int64_t D_out = 8;

    cudnnHandle_t handle;
    check_cudnn_status(cudnnCreate(&handle));

    auto dtype = CUDNN_DATA_HALF;
    auto layout = lib::tensor::layout::Row;
    auto x = lib::tensor::create_cudnn(shape{B, 1, D_in}, dtype, "input"s, layout, false, false);
    auto w =
        lib::tensor::create_cudnn(shape{1, D_in, D_out}, dtype, "weight"s, layout, false, false);
    auto b = lib::tensor::create_cudnn(shape{1, 1, D_out}, dtype, "bias"s, layout, false, false);
    auto y = lib::tensor::create_cudnn(shape{B, 1, D_out}, dtype, "output"s, layout, false, false);

    memory::DeviceMemory<half> x_mem(x);
    memory::DeviceMemory<half> w_mem(w);
    memory::DeviceMemory<half> b_mem(b);
    memory::DeviceMemory<half> y_mem(y);

    matmul_bias_relu_fwd(x, x_mem.get_ptr(), w, w_mem.get_ptr(), b, b_mem.get_ptr(), y,
                         y_mem.get_ptr(), handle);

    return 0;
}