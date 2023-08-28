#pragma once

#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/tensor.h"
#include "lib/types.h"

using namespace lib;

void matmul_bias_relu_fwd(tensor::Tensor &x, tensor::Tensor &w, tensor::Tensor &b,
                          tensor::Tensor &y, cudnnHandle_t handle,
                          PlanCacheManager &plan_cache_manager);