#include <iomanip>

#include "lib/dataset/mnist_reader.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path-to-mnist>" << std::endl;
        return 1;
    }

    std::string mnist_path = argv[1];

    // Load MNIST dataset
    auto dataset = lib::mnist::read_dataset<std::vector, std::vector, int, uint8_t>(mnist_path);

    std::cout << "Number of training images: " << dataset.training_images.size() << std::endl;
    std::cout << "Number of training labels: " << dataset.training_labels.size() << std::endl;
    std::cout << "Number of test images: " << dataset.test_images.size() << std::endl;
    std::cout << "Number of test labels: " << dataset.test_labels.size() << std::endl;

    size_t i = 0;

    auto img = dataset.training_images[i];    // First training image [28 x 28
    auto label = dataset.training_labels[i];  // Corresponding label

    for (size_t i = 0; i < img.size(); i++) {
        if (i % 28 == 0) std::cout << std::endl;
        std::cout << std::setw(4) << (int)img[i];
    }

    std::cout << std::endl << (int)label << std::endl;

    return 0;
}