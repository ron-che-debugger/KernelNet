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

// A simple structure to hold one CIFAR-10 sample.
struct CIFAR10Sample {
    Tensor image; // A Tensor of shape (3 * 32 * 32) with float values.
    Tensor label; // A Tensor of shape (10) representing a one-hot vector.
};

class CIFAR10Dataset {
  public:
    // All loaded samples.
    vector<CIFAR10Sample> samples;

    // Constructor:
    //   data_dir: directory containing the CIFAR-10 binary files.
    //   train: if true, load data_batch_1.bin ... data_batch_5.bin;
    //          if false, load test_batch.bin.
    CIFAR10Dataset(const string &data_dir, bool train) {
        if (train) {
            for (int i = 1; i <= 1; i++) {
                stringstream ss;
                ss << data_dir << "/data_batch_" << i << ".bin";
                loadFile(ss.str());
            }
        } else {
            string filename = data_dir + "/test_batch.bin";
            loadFile(filename);
        }
    }

    // Returns the number of samples in the dataset.
    size_t size() const {
        return samples.size();
    }

    // Get sample at index.
    const CIFAR10Sample &getSample(size_t index) const {
        assert(index < samples.size());
        return samples[index];
    }

  private:
    // Helper to load one CIFAR-10 binary file.
    void loadFile(const string &filename) {
        ifstream file(filename, ios::binary);
        if (!file.is_open()) {
            throw runtime_error("Could not open CIFAR-10 file: " + filename);
        }

        const int num_images = 10000;
        const int image_size = 32 * 32 * 3;     // 3072 bytes per image.
        const int record_size = 1 + image_size; // 1 byte label + image

        vector<unsigned char> buffer(record_size);

        for (int i = 0; i < num_images; i++) {
            file.read(reinterpret_cast<char *>(buffer.data()), record_size);
            if (file.gcount() != record_size)
                break;

            // The first byte is the label.
            unsigned char label = buffer[0];

            // Create an image tensor (flattened vector of floats).
            Tensor image(static_cast<size_t>(image_size), CPU);
            float *image_data = image.data();
            for (int j = 0; j < image_size; j++) {
                // Normalize from [0,255] to [0,1].
                image_data[j] = static_cast<float>(buffer[j + 1]) / 255.0f;
            }

            // Create a one-hot label tensor (10 classes).
            Tensor onehot(10, CPU);
            float *onehot_data = onehot.data();
            for (int j = 0; j < 10; j++) {
                onehot_data[j] = (j == label) ? 1.0f : 0.0f;
            }

            CIFAR10Sample sample;
            sample.image = image;
            sample.label = onehot;
            samples.push_back(sample);
        }
        file.close();
        cout << "Loaded " << samples.size() << " samples from " << filename << endl;
    }
};