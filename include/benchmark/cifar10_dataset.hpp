/**
 * @file cifar10_dataset.hpp
 * @brief Loads CIFAR-10 image and label data from binary files into a vector of samples.
 *
 * This file includes:
 * - `CIFAR10Sample`: A structure representing a single image-label pair
 * - `CIFAR10Dataset`: A dataset loader that reads CIFAR-10 binary files
 *     - Loads 32x32 RGB images as normalized float Tensors
 *     - Converts labels into one-hot encoded vectors
 *     - Supports both training (5 batches) and test (1 batch) data
 */

#pragma once

#include "tensor.hpp"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace kernelnet {
namespace data {
/**
 * @brief Structure to hold a single CIFAR-10 sample.
 *
 * Each sample contains:
 * - image: A Tensor of shape (3 × 32 × 32), stored as a flat array (3072 elements)
 * - label: A one-hot Tensor of shape (10) for classification
 */
struct CIFAR10Sample {
    Tensor image; ///< Flattened image tensor of size 3072
    Tensor label; ///< One-hot label tensor of size 10
};

/**
 * @brief Loads and stores CIFAR-10 dataset samples from binary files.
 *
 * CIFAR-10 images are loaded from the standard binary format, normalized to [0,1],
 * and paired with one-hot encoded class labels. This class supports loading both
 * training (data_batch_1 to data_batch_5) and testing (test_batch) files.
 */
class CIFAR10Dataset {
  public:
    vector<CIFAR10Sample> samples; ///< All loaded image-label pairs

    /**
     * @brief Constructor that loads dataset samples from disk.
     *
     * @param data_dir Directory path containing CIFAR-10 binary files.
     * @param train If true, loads training data from 5 batches; else loads test batch.
     */
    CIFAR10Dataset(const string &data_dir, bool train) {
        if (train) {
            for (int i = 1; i <= 5; i++) {
                stringstream ss;
                ss << data_dir << "/data_batch_" << i << ".bin";
                loadFile(ss.str());
            }
        } else {
            string filename = data_dir + "/test_batch.bin";
            loadFile(filename);
        }
    }

    /**
     * @brief Returns the number of samples loaded in the dataset.
     * @return Number of CIFAR-10 samples available.
     */
    size_t size() const {
        return samples.size();
    }

    /**
     * @brief Accesses a sample by index.
     * @param index Index of the sample to retrieve.
     * @return A const reference to the selected CIFAR10Sample.
     */
    const CIFAR10Sample &getSample(size_t index) const {
        assert(index < samples.size());
        return samples[index];
    }

  private:
    /**
     * @brief Loads a single CIFAR-10 binary file and parses image-label records.
     *
     * Each record contains 1 byte for label and 3072 bytes for image data (RGB channels).
     * Image values are normalized to [0,1] and labels are converted to one-hot format.
     *
     * @param filename Path to the binary CIFAR-10 file to load.
     */
    void loadFile(const string &filename) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("Could not open CIFAR-10 file: " + filename);
        }

        const int num_images = 1500;            ///< Number of images to read from the file
        const int image_size = 32 * 32 * 3;     ///< Total pixels per image (3072)
        const int record_size = 1 + image_size; ///< One label byte + image data

        vector<unsigned char> buffer(record_size); ///< Temporary buffer for one record

        for (int i = 0; i < num_images; i++) {
            file.read(reinterpret_cast<char *>(buffer.data()), record_size);
            if (file.gcount() != record_size)
                break;

            unsigned char label = buffer[0];

            // Create normalized image tensor
            Tensor image(static_cast<size_t>(image_size), CPU);
            float *image_data = image.data();
            for (int j = 0; j < image_size; j++) {
                image_data[j] = static_cast<float>(buffer[j + 1]) / 255.0f;
            }

            // Create one-hot encoded label tensor
            Tensor onehot(10, CPU);
            float *onehot_data = onehot.data();
            for (int j = 0; j < 10; j++) {
                onehot_data[j] = (j == label) ? 1.0f : 0.0f;
            }

            // Store sample
            CIFAR10Sample sample;
            sample.image = image;
            sample.label = onehot;
            samples.push_back(sample);
        }

        file.close();
        cout << "Loaded " << samples.size() << " samples from " << filename << endl;
    }
};
} // namespace data
} // namespace kernelnet