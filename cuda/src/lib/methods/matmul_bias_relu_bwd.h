#pragma once

#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/tensor.h"
#include "lib/types.h"

using namespace lib;

void matmul_bias_relu_bwd(tensor::Tensor &x, tensor::Tensor &w, tensor::Tensor &b,
                          tensor::Tensor &dL_dy, tensor::Tensor &dL_dx, tensor::Tensor &dL_dw,
                          tensor::Tensor &dL_db, cudnnHandle_t handle,
                          PlanCacheManager &plan_cache_manager);