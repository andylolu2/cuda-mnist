#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include "lib/init.cuh"
#include "lib/tensor.h"
#include "lib/types.h"
#include "lib/utils.h"

size_t n_elem(lib::shape dims) {
    size_t n_elem = 1;
    for (auto &dim : dims) {
        n_elem *= dim;
    }
    return n_elem;
}

namespace lib {
    namespace tensor {
        Tensor::Tensor(std::string name, lib::shape dims, lib::shape strides, cudnnDataType_t dtype,
                       bool is_virtual, bool is_value)
            : name(name),
              id(hashString(name)),
              dims(dims),
              strides(strides),
              dtype(dtype),
              is_virtual(is_virtual),
              is_value(is_value),
              cudnn_tensor(cudnn_frontend::TensorBuilder()
                               .setDim(dims.size(), dims.data())
                               .setStrides(strides.size(), strides.data())
                               .setId(id)
                               .setAlignment(16)
                               .setDataType(dtype)
                               .setVirtual(is_virtual)
                               .setByValue(is_value)
                               .build()) {
            if (is_virtual) {
                data = nullptr;
            } else {
                size_t dtype_size = cudnn_dtype_size(dtype);
                check_cuda_status(cudaMalloc((void **)&data, n_elem(dims) * dtype_size));
            }
        }

        Tensor::~Tensor() {
            if (!is_virtual) {
                check_cuda_status(cudaFree(data));
            }
        }

        std::string Tensor::get_name() const { return name; }
        uint64_t Tensor::get_id() const { return id; }
        lib::shape Tensor::get_dims() const { return dims; }
        lib::shape Tensor::get_strides() const { return strides; }
        cudnnDataType_t Tensor::get_dtype() const { return dtype; }
        void *Tensor::get_data() const { return data; }
        bool Tensor::get_is_virtual() const { return is_virtual; }
        bool Tensor::get_is_value() const { return is_value; }
        cudnn_frontend::Tensor &Tensor::get_cudnn() { return cudnn_tensor; }

        size_t Tensor::size() const { return n_elem(dims); }
        size_t Tensor::bytes() const { return n_elem(dims) * cudnn_dtype_size(dtype); }

        std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
            bool first;

            // Print metadata
            os << "Tensor(name=" << tensor.name;
            os << ", id=" << tensor.id;
            os << ", dtype=" << tensor.dtype;

            os << ", dims=[";
            first = true;
            for (auto &dim : tensor.dims) {
                if (first) {
                    first = false;
                    os << dim;
                } else {
                    os << ", " << dim;
                }
            }
            os << "], strides=[";
            first = true;
            for (auto &stride : tensor.strides) {
                if (first) {
                    first = false;
                    os << stride;
                } else {
                    os << ", " << stride;
                }
            }
            os << "])\n";

            if (tensor.is_virtual) {
                return os;
            }

            // Print data
            // 0. Synchronize device
            check_cuda_status(cudaDeviceSynchronize());

            // 1. Copy data to host
            std::vector<char> host_data(tensor.bytes());
            check_cuda_status(
                cudaMemcpy(host_data.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));

            // 2. Print data, we do this recursively
            std::function<void(std::vector<size_t>, size_t)> print_array;
            print_array = [&os, &tensor, &host_data, &print_array](std::vector<size_t> idx,
                                                                   size_t indent_level) {
                if (idx.size() == tensor.dims.size() - 1) {
                    // Base case (vector)
                    os << std::string(2 * indent_level, ' ') << "[";
                    for (size_t i = 0; i < tensor.dims[idx.size()]; i++) {
                        if (i != 0) {
                            os << ", ";
                        }
                        size_t offset = 0;
                        for (size_t j = 0; j < idx.size(); j++) {
                            offset += idx[j] * tensor.strides[j];
                        }
                        offset += i * tensor.strides[idx.size()];
                        offset *= cudnn_dtype_size(tensor.dtype);
                        os << cudnn_value_to_str((void *)(host_data.data() + offset), tensor.dtype);
                    }
                    os << "]";
                } else if (idx.size() == tensor.dims.size()) {
                    // Base case (scalar), print value
                    size_t offset = 0;
                    for (size_t i = 0; i < idx.size(); i++) {
                        offset += idx[i] * tensor.strides[i];
                        offset *= cudnn_dtype_size(tensor.dtype);
                    }
                    os << std::string(2 * indent_level, ' ');
                    os << cudnn_value_to_str((void *)(host_data.data() + offset), tensor.dtype);
                } else {
                    // Recursive case, print array
                    os << std::string(2 * indent_level, ' ') << "[\n";
                    for (size_t i = 0; i < tensor.dims[idx.size()]; i++) {
                        if (i != 0) {
                            os << ",\n";
                        }
                        idx.push_back(i);
                        print_array(idx, indent_level + 1);
                        idx.pop_back();
                    }
                    os << "\n" << std::string(2 * indent_level, ' ') << "]";
                }
            };
            print_array({}, 0);

            return os;
        }
        Tensor Tensor::transpose(size_t dim_1, size_t dim_2) const {
            lib::shape new_dims = dims;
            lib::shape new_strides = strides;
            std::swap(new_dims[dim_1], new_dims[dim_2]);
            std::swap(new_strides[dim_1], new_strides[dim_2]);
            return Tensor(name + ".T", new_dims, new_strides, dtype, is_virtual, is_value);
        }
        std::pair<uint64_t, void *> Tensor::id_and_ptr() const { return std::make_pair(id, data); }
    }  // namespace tensor
}  // namespace lib
