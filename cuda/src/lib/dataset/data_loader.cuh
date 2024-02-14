#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/dataset/mnist_reader.hpp"
#include "lib/op/pointwise_ops.cuh"
#include "lib/op/tensor_ops.cuh"
#include "lib/utils/device_tensor.cuh"

using namespace cute;
using namespace cutlass;

namespace lib {
    namespace mnist {
        enum Split { TRAIN, TEST };

        template <typename T>
        class DataLoader {
            using ImageTensor =
                Tensor<ViewEngine<gmem_ptr<T>>, Layout<Shape<int, int, int>, Stride<int, int, _1>>>;
            using LabelTensor = Tensor<ViewEngine<gmem_ptr<int>>, Layout<Shape<int, _1>>>;
            using BatchImageTensor = Tensor<ViewEngine<gmem_ptr<T>>, Layout<Shape<int, int, int>>>;
            using BatchArrayTensor = Tensor<ViewEngine<gmem_ptr<T>>, Layout<Shape<int, int>>>;
            using BatchLabelTensor = LabelTensor;

           private:
            const int train_size = 60000;
            const int test_size = 10000;
            int batch_size;
            int current_idx;
            int dataset_size;
            bool train;

            std::vector<size_t> indices;

            // Device tensors that hold the entire dataset.
            DeviceTensor<T, Layout<Shape<int, int, int>, Stride<int, int, _1>>> images;
            DeviceTensor<int, Layout<Shape<int, _1>, Stride<_1, _1>>> labels;

            // Buffer that holds the current batch of images and labels.
            DeviceTensor<T, Layout<Shape<int, int, int>>> batch_images;
            DeviceTensor<int, Layout<Shape<int, _1>>> batch_labels;

            void shuffle() { std::random_shuffle(indices.begin(), indices.end()); }

           public:
            DataLoader(std::string data_dir, Split split, int batch_size_)
                : batch_size(batch_size_),
                  current_idx(0),
                  dataset_size(split == TRAIN ? train_size : test_size),
                  train(split == TRAIN),
                  indices(train ? train_size : test_size),
                  images(make_device_tensor<T>(
                      make_shape(dataset_size, 28, 28), make_stride(28 * 28, 28, _1{}))),
                  labels(make_device_tensor<int>(
                      make_shape(dataset_size, _1{}), make_stride(_1{}, _1{}))),
                  batch_images(make_device_tensor<T>(make_shape(batch_size, 28, 28))),
                  batch_labels(make_device_tensor<int>(make_shape(batch_size, _1{}))) {
                auto dataset = mnist::read_dataset(data_dir);

                // Flatten host data into a single vector for copying to device.
                std::vector<T> image_host(dataset_size * 28 * 28);
                std::vector<int> label_host(dataset_size);
                for (size_t i = 0; i < dataset_size; ++i) {
                    for (int w = 0; w < 28; ++w) {
                        for (int h = 0; h < 28; ++h) {
                            int image_idx = w * 28 + h;
                            uint8_t pixel;
                            switch (split) {
                                case TRAIN:
                                    pixel = dataset.training_images[i][image_idx];
                                    break;
                                case TEST:
                                    pixel = dataset.test_images[i][image_idx];
                                    break;
                            }
                            image_host[i * 28 * 28 + image_idx] =
                                static_cast<T>(pixel) / static_cast<T>(255);
                        }
                    }
                    switch (split) {
                        case TRAIN:
                            label_host[i] = static_cast<int>(dataset.training_labels[i]);
                            break;
                        case TEST:
                            label_host[i] = static_cast<int>(dataset.test_labels[i]);
                            break;
                    }
                }

                // Load all data into device memory.
                device_memory::copy_to_device(
                    images.data_ptr(), image_host.data(), image_host.size());
                device_memory::copy_to_device(
                    labels.data_ptr(), label_host.data(), label_host.size());

                // Fill indices with 0, 1, ..., n.
                std::iota(indices.begin(), indices.end(), 0);

                // Shuffle indices if training.
                if (train) {
                    shuffle();
                }
            }
            ~DataLoader() = default;

            /**
             * Load the next batch of images and labels into the buffer tensors.
             */
            void next() {
                if (current_idx + batch_size > dataset_size) {
                    current_idx = 0;
                    if (batch_size > size(images.view())) {
                        throw std::runtime_error("Batch size is larger than dataset size");
                    }
                    shuffle();
                }

                for (int i = 0; i < batch_size; i++) {
                    Tensor image_slice = images.view()(indices[current_idx + i], _, _);
                    Tensor batch_image_slice = batch_images.view()(i, _, _);
                    lib::op::convert(batch_image_slice, image_slice);

                    Tensor label_slice = labels.view()(indices[current_idx + i], _);
                    Tensor batch_label_slice = batch_labels.view()(i, _);
                    lib::op::convert(batch_label_slice, label_slice);
                }
                current_idx += batch_size;
            }

            auto get_batch_array() {
                return make_tensor(
                    make_gmem_ptr(batch_images.data_ptr()), make_shape(batch_size, 28 * 28));
            }

            auto get_batch_labels() {
                return make_tensor(make_gmem_ptr(batch_labels.data_ptr()), make_shape(batch_size));
            }
        };
    }  // namespace mnist
}  // namespace lib