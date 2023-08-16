#pragma once

#include <cudnn.h>
#include <cudnn_frontend.h>

#include <string>

#include "lib/types.h"
#include "lib/utils.h"

namespace lib {
    namespace tensor {
        enum class layout { Row = 0, Column = 1 };

        cudnn_frontend::Tensor create_cudnn(shape dims, cudnnDataType_t dtype, std::string name,
                                            layout layout, bool virtual_, bool value);

        cudnn_frontend::Tensor create_cudnn(shape dims, shape strides, cudnnDataType_t dtype,
                                            std::string name, bool virtual_, bool value);

    }  // namespace tensor
}  // namespace lib
