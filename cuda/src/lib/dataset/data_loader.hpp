#pragma once

#include <cute/tensor.hpp>
#include <numeric>

#include "lib/dataset/mnist_reader.hpp"

using namespace cute;

namespace lib {
    namespace mnist {
        enum Split { TRAIN, TEST };

        template <typename T>
        class DataLoader {
            using ImageTensor = Tensor<ViewEngine<T *>, Layout<Shape<int, int, int>>>;
            using LabelTensor = Tensor<ViewEngine<int *>, Layout<Shape<int>>>;

           private:
            size_t batch_size;
            size_t current_idx;

            std::vector<std::vector<uint8_t>> images;
            std::vector<uint8_t> labels;
            std::vector<size_t> indices;

            std::vector<T> image_data;
            std::vector<int> label_data;
            ImageTensor image_host;
            LabelTensor label_host;

            void shuffle() { std::random_shuffle(indices.begin(), indices.end()); }

           public:
            DataLoader(std::string data_dir, Split split, size_t batch_size)
                : batch_size(batch_size),
                  current_idx(0),
                  image_data(batch_size * 28 * 28),
                  label_data(batch_size),
                  image_host(make_tensor(
                      image_data.data(), make_shape(static_cast<int>(batch_size), 28, 28))),
                  label_host(
                      make_tensor(label_data.data(), make_shape(static_cast<int>(batch_size)))) {
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
                indices.resize(images.size());
                std::iota(indices.begin(), indices.end(), 0);

                if (split == TRAIN) {
                    shuffle();
                }
            }
            ~DataLoader() = default;

            std::tuple<ImageTensor, LabelTensor> next() {
                if (current_idx + batch_size > images.size()) {
                    current_idx = 0;
                    if (batch_size > images.size()) {
                        throw std::runtime_error("Batch size is larger than dataset size");
                    }
                    shuffle();
                }

                for (size_t i = 0; i < batch_size; ++i) {
                    for (int w = 0; w < 28; ++w) {
                        for (int h = 0; h < 28; ++h) {
                            image_host(i, w, h) = T(images[current_idx][w * 28 + h]) / T(255);
                        }
                    }
                    label_host(i) = labels[current_idx];
                    current_idx++;
                }

                return std::make_tuple(image_host, label_host);
            }
        };
    }  // namespace mnist
}  // namespace lib