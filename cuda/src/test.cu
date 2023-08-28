#include <cuda_fp16.h>
#include <cudnn_frontend.h>

#include "lib/cache.h"
#include "lib/methods/matmul_bias_relu_bwd.h"
#include "lib/methods/matmul_bias_relu_fwd.h"
#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"

using namespace lib;

int main() {
    int64_t B = 8;
    int64_t K = 8;
    int64_t N = 8;

    cudnnHandle_t handle;
    check_cudnn_status(cudnnCreate(&handle));
    PlanCacheManager plan_cache_manager;

    auto dtype = CUDNN_DATA_HALF;
    tensor::Tensor x("x", {B, 1, K}, {K, K, 1}, dtype, false, false);
    tensor::Tensor w("w", {1, K, N}, {N * K, N, 1}, dtype, false, false);
    tensor::Tensor b("b", {1, 1, N}, {N, N, 1}, dtype, false, false);
    tensor::Tensor y("y", {B, 1, N}, {N, N, 1}, dtype, false, false);

    x.fill<half>((half)1.0);
    w.fill<half>((half)1.0);
    b.fill<half>((half)0.0);
    // y.fill<half>((half)0.0);

    // std::cout << x << std::endl;
    // std::cout << w << std::endl;
    // std::cout << y << std::endl;

    // matmul_bias_relu_fwd(x, w, b, y, handle, plan_cache);

    // std::cout << y << std::endl;

    tensor::Tensor dx("dx", {B, 1, K}, {K, K, 1}, CUDNN_DATA_FLOAT, false, false);
    tensor::Tensor dw("dw", {1, K, N}, {N * K, N, 1}, CUDNN_DATA_FLOAT, false, false);
    tensor::Tensor db("db", {B, 1, N}, {1, 1, 1}, CUDNN_DATA_FLOAT, false, false);
    tensor::Tensor dy("dy", {B, 1, N}, {N, N, 1}, CUDNN_DATA_FLOAT, false, false);

    dx.fill<float>(0.0);
    dw.fill<float>(0.0);
    db.fill<float>(0.0);
    dy.fill<float>(1.0);

    matmul_bias_relu_bwd(x, w, b, dy, dx, dw, db, handle, plan_cache_manager);

    return 0;
}