#pragma once

#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <string>

#include "lib/init.cuh"
#include "lib/types.h"
#include "lib/utils.h"

namespace lib {
    namespace tensor {
        class Tensor {
           private:
            std::string name;
            uint64_t id;
            lib::shape dims;
            lib::shape strides;
            cudnnDataType_t dtype;
            bool is_virtual;
            bool is_value;
            void *data;
            cudnn_frontend::Tensor cudnn_tensor;

            friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

           public:
            Tensor(std::string name, lib::shape dims, lib::shape strides, cudnnDataType_t dtype,
                   bool is_virtual, bool is_value);

            ~Tensor();

            // Getters
            std::string get_name() const;
            lib::shape get_dims() const;
            lib::shape get_strides() const;
            cudnnDataType_t get_dtype() const;
            uint64_t get_id() const;
            void *get_data() const;
            bool get_is_virtual() const;
            bool get_is_value() const;
            cudnn_frontend::Tensor &get_cudnn();

            // Derived getters
            size_t size() const;
            size_t bytes() const;

            // Methods
            template <typename T>
            void fill(T value) {
                if (is_virtual) {
                    throw std::runtime_error("Cannot fill virtual tensor");
                }

                size_t block_size = 256;
                size_t n_blocks = (size() + block_size - 1) / block_size;
                fill_kernel<T><<<n_blocks, block_size>>>(static_cast<T *>(data), value, size());
            }
            Tensor transpose(size_t dim1, size_t dim2) const;
            std::pair<uint64_t, void *> id_and_ptr() const;
        };
    }  // namespace tensor
}  // namespace lib
