#pragma once

#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/types.h"

void matmul_bias_relu_fwd(cudnn_frontend::Tensor& x, void* x_ptr, cudnn_frontend::Tensor& w,
                          void* w_ptr, cudnn_frontend::Tensor& b, void* b_ptr,
                          cudnn_frontend::Tensor& y, void* y_ptr, cudnnHandle_t handle);