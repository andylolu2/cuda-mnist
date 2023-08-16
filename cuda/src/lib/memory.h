#pragma once

#include <cudnn_frontend.h>

#include "lib/types.h"
#include "lib/utils.h"

namespace lib {
    namespace memory {
        template <typename T>
        class DeviceMemory {
           private:
            T* data;
            size_t size;
            DeviceMemory(&DeviceMemory& other) = delete;  // copy is forbidden

           public:
            DeviceMemory(cudnn_frontend::Tensor& tensor) {
                shape dims(tensor.getDim(), tensor.getDim() + tensor.getDimCount());
                size_t size = 1;
                for (auto& dim : dims) {
                    size *= dim;
                }
                this->size = size;
                check_cuda_status(cudaMalloc((void**)&data, size * sizeof(T)));
            }
            DeviceMemory(shape dims) {
                size_t size = 1;
                for (auto& dim : dims) {
                    size *= dim;
                }
                this->size = size;
                check_cuda_status(cudaMalloc((void**)&data, size * sizeof(T)));
            }
            DeviceMemory(size_t size) {
                this->size = size;
                check_cuda_status(cudaMalloc((void**)&data, size * sizeof(T)));
            }

            ~DeviceMemory() { check_cuda_status(cudaFree((void*)data)); }

            T* get_ptr() { return data; }
            size_t get_size() { return size; }
        };
    }  // namespace memory
}  // namespace lib