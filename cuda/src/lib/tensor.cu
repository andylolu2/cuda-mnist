#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"

namespace lib {
    namespace tensor {
        shape generateStrides(shape dims, layout layout) {
            shape strides(dims.size(), 1);
            if (layout == layout::Row) {
                for (int i = dims.size() - 2; i >= 0; i--) {
                    strides[i] = strides[i + 1] * dims[i + 1];
                }
            } else {
                for (int i = 1; i < dims.size(); i++) {
                    strides[i] = strides[i - 1] * dims[i - 1];
                }
            }
            return strides;
        }

        cudnn_frontend::Tensor create_cudnn(shape dims, cudnnDataType_t dtype, std::string name,
                                            layout layout, bool virtual_, bool value) {
            auto strides = generateStrides(dims, layout);
            return create_cudnn(dims, strides, dtype, name, virtual_, value);
        }
        cudnn_frontend::Tensor create_cudnn(shape dims, shape strides, cudnnDataType_t dtype,
                                            std::string name, bool virtual_, bool value) {
            auto id = hashString(name);
            return cudnn_frontend::TensorBuilder()
                .setDim(dims.size(), dims.data())
                .setStrides(strides.size(), strides.data())
                .setId(hashString(name))
                .setAlignment(16)
                .setDataType(dtype)
                .setVirtual(virtual_)
                .setByValue(value)
                .build();
        }
    }  // namespace tensor
}  // namespace lib