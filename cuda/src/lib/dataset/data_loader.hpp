#pragma once

#include <cutlass/util/device_memory.h>

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/dataset/mnist_reader.hpp"
#include "lib/op/unary_pointwise.cuh"
#include "lib/op/tensor_ops.cuh"

using namespace cute;

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

            cutlass::DeviceAllocation<T> image_data;
            cutlass::DeviceAllocation<int> label_data;
            cutlass::DeviceAllocation<T> batch_image_data;
            cutlass::DeviceAllocation<int> batch_label_data;
            ImageTensor images;
            LabelTensor labels;
            BatchImageTensor batch_images;
            BatchLabelTensor batch_labels;

            void shuffle() { std::random_shuffle(indices.begin(), indices.end()); }

           public:
            DataLoader(std::string data_dir, Split split, int batch_size_)
                : batch_size(batch_size_),
                  current_idx(0),
                  dataset_size(split == TRAIN ? train_size : test_size),
                  train(split == TRAIN),
                  indices(train ? train_size : test_size),
                  image_data(dataset_size * 28 * 28),
                  label_data(dataset_size),
                  batch_image_data(batch_size * 28 * 28),
                  batch_label_data(batch_size),
                  images(make_tensor(
                      make_gmem_ptr(image_data.get()),
                      make_shape(dataset_size, 28, 28),
                      make_stride(28 * 28, 28, _1{}))),
                  labels(
                      make_tensor(make_gmem_ptr(label_data.get()), make_shape(dataset_size, _1{}))),
                  batch_images(make_tensor(
                      make_gmem_ptr(batch_image_data.get()), make_shape(batch_size, 28, 28))),
                  batch_labels(make_tensor(
                      make_gmem_ptr(batch_label_data.get()), make_shape(batch_size, _1{}))) {
                std::vector<std::vector<uint8_t>> images;
                std::vector<uint8_t> labels;

                auto dataset = mnist::read_dataset(data_dir);
                switch (split) {
                    case TRAIN:
                        images = dataset.training_images;
                        labels = dataset.training_labels;
                        break;
                    case TEST:
                        images = dataset.test_images;
                        labels = dataset.test_labels;
                        break;
                }

                // Fill indices with 0, 1, ..., n.
                std::iota(indices.begin(), indices.end(), 0);

                // Flatten host data into a single vector.
                std::vector<T> image_host(images.size() * 28 * 28);
                std::vector<int> label_host(labels.size());
                for (size_t i = 0; i < images.size(); ++i) {
                    for (int w = 0; w < 28; ++w) {
                        for (int h = 0; h < 28; ++h) {
                            int image_idx = w * 28 + h;
                            image_host[i * 28 * 28 + image_idx] = T(images[i][image_idx]) / T(255);
                        }
                    }
                    label_host[i] = int(labels[i]);
                }

                // Load all data into device memory.
                image_data.copy_from_host(image_host.data());
                label_data.copy_from_host(label_host.data());

                // Shuffle indices if training.
                if (train) {
                    shuffle();
                }
            }
            ~DataLoader() = default;

            void next() {
                if (current_idx + batch_size > dataset_size) {
                    current_idx = 0;
                    if (batch_size > images.size()) {
                        throw std::runtime_error("Batch size is larger than dataset size");
                    }
                    shuffle();
                }

                for (int i = 0; i < batch_size; i++) {
                    Tensor image_slice = images(indices[current_idx + i], _, _);
                    Tensor batch_image_slice = batch_images(i, _, _);
                    lib::op::identity(image_slice, batch_image_slice);

                    Tensor label_slice = labels(indices[current_idx + i], _);
                    Tensor batch_label_slice = batch_labels(i, _);
                    lib::op::identity(label_slice, batch_label_slice);
                }
                current_idx += batch_size;
            }

            BatchImageTensor& get_batch_images() { return batch_images; }

            auto get_batch_array() {
                return make_tensor(
                    make_gmem_ptr(batch_image_data.get()), make_shape(batch_size, 28 * 28));
            }

            auto get_batch_labels() {
                return make_tensor(make_gmem_ptr(batch_label_data.get()), make_shape(batch_size));
            }
        };
    }  // namespace mnist
}  // namespace lib