#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>

#include "lib/modules/linear.cuh"
#include "lib/op/pointwise_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/utils/device_tensor.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace module {
        class MLP {
            using ActivationType = half_t;
            using GradType = half_t;

            using ActivationShape = Shape<int, int>;
            using ActivationTensor =
                Tensor<ViewEngine<gmem_ptr<ActivationType>>, Layout<ActivationShape>>;
            using DActivationTensor =
                Tensor<ViewEngine<gmem_ptr<GradType>>, Layout<ActivationShape>>;

           private:
            int batch_size;
            int in_features;
            std::vector<int> feature_sizes;
            std::vector<Linear> layers;
            std::vector<DeviceTensor<ActivationType, Layout<ActivationShape>>> activations;
            std::vector<DeviceTensor<GradType, Layout<ActivationShape>>> d_activations;

           public:
            MLP(int batch_size, int in_features, std::vector<int> feature_sizes)
                : batch_size(batch_size), in_features(in_features), feature_sizes(feature_sizes) {
                // Need to reserve space for the activations_data and d_activations_data in
                // particular. Otherwise the DeviceAllocation will be moved and the pointers to
                // device will be invalid.
                // TODO: There should be a better way to handle this.
                activations.reserve(feature_sizes.size());
                d_activations.reserve(feature_sizes.size());

                for (size_t i = 0; i < feature_sizes.size(); i++) {
                    int in = i == 0 ? in_features : feature_sizes[i - 1];
                    int out = feature_sizes[i];
                    layers.emplace_back(batch_size, in, out);

                    if (i != feature_sizes.size() - 1) {
                        activations.emplace_back(
                            make_device_tensor<ActivationType>(make_shape(batch_size, out)));
                        d_activations.emplace_back(
                            make_device_tensor<GradType>(make_shape(batch_size, out)));
                    }
                }
            }
            ~MLP() = default;

            size_t n_layers() { return layers.size(); }

            Linear& operator[](int i) { return layers[i]; }

            void init(int seed = 0, std::string mode = "kaiming") {
                for (auto& layer : layers) {
                    layer.init(seed, mode);
                }
            }

            void forward(ActivationTensor const& x, ActivationTensor const& y) {
                for (size_t i = 0; i < layers.size(); i++) {
                    auto& layer = layers[i];
                    auto& x_in = i == 0 ? x : activations[i - 1].view();
                    auto& x_out = i == layers.size() - 1 ? y : activations[i].view();
                    layer.forward(x_in, x_out);
                    if (i < layers.size() - 1) {
                        lib::op::relu(x_out, x_out);
                    }
                }
            }

            void backward(
                ActivationTensor const& x,
                DActivationTensor const& dy,
                DActivationTensor const& dx) {
                for (size_t i = layers.size(); i-- > 0;) {
                    auto& layer = layers[i];
                    auto x_in = i == 0 ? x : activations[i - 1].view();
                    auto dx_in = i == 0 ? dx : d_activations[i - 1].view();
                    auto dx_out = i == layers.size() - 1 ? dy : d_activations[i].view();
                    layer.backward(x_in, dx_out, dx_in);
                    if (i > 0) {
                        lib::op::drelu(dx_in, dx_in, x_in);
                    }
                }
            }

            void update(float lr) {
                for (auto& layer : layers) {
                    layer.update(lr);
                }
            }

            /**
             * Compute the number of TFLOPs for each training step
             */
            auto tflops() {
                float tflops = 0;

                for (size_t i = 0; i < layers.size(); i++) {
                    auto& layer = layers[i];
                    tflops += layer.tflops();
                    if (i < layers.size() - 1) {
                        tflops += 2 * feature_sizes[i] * batch_size / 1e12;  // relu and drelu
                    }
                }
                return tflops;
            }
        };
    }  // namespace module
}  // namespace lib