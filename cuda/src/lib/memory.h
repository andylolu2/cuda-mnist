#pragma once

#include <cudnn_frontend.h>

#include "lib/types.h"
#include "lib/utils.h"

namespace lib {
    namespace memory {
        class DeviceMemory {
           private:
            DeviceMemory(DeviceMemory const&) = delete;  // copy is forbidden
            void* data;
            shape dims;
            size_t size;

           public:
            DeviceMemory(cudnn_frontend::Tensor& tensor) {
                shape dims(tensor.getDim(), tensor.getDim() + tensor.getDimCount());
                size_t size = 1;
                for (auto& dim : dims) {
                    size *= dim;
                }

                this->dims = dims;
                this->size = size;
                check_cuda_status(cudaMalloc((void**)&data, size * sizeof(T)));
            }

            DeviceMemory(shape dims) {
                size_t size = 1;
                for (auto& dim : dims) {
                    size *= dim;
                }

                this->dims = dims;
                this->size = size;
                check_cuda_status(cudaMalloc((void**)&data, size * sizeof(T)));
            }

            ~DeviceMemory() { check_cuda_status(cudaFree((void*)data)); }

            void* get_ptr() { return data; }
            shape get_shape() { return dims; }
            size_t get_size() { return size; }
        };
    }  // namespace memory
}  // namespace lib