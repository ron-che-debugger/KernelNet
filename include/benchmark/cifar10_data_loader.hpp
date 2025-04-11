#pragma once

#include "./benchmark/cifar10_dataset.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class CIFAR10DataLoader {
  public:
    CIFAR10Dataset &dataset;
    int batch_size;
    bool shuffle;
    vector<size_t> indices;
    size_t current_index;

    // Constructor: accepts a dataset reference, desired batch size, and a shuffle flag.
    CIFAR10DataLoader(CIFAR10Dataset &dataset, int batch_size, bool shuffle = true)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_index(0) {
        reset();
    }

    // Reset the DataLoader for a new epoch.
    void reset() {
        indices.clear();
        size_t n = dataset.size();
        for (size_t i = 0; i < n; i++) {
            indices.push_back(i);
        }
        if (shuffle) {
            random_device rd;
            mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }
        current_index = 0;
    }

    // Whether there are more batches in the current epoch.
    bool hasNext() const {
        return current_index + batch_size <= indices.size();
    }

    // Returns the next batch as a pair of Tensors: {batch_images, batch_labels}.
    // The images are concatenated along the first dimension (flattened).
    // For example, if a single image is of size (3072), and the batch size is b, the output
    // will be a Tensor with b * 3072 elements.
    // Similarly, the labels Tensor will have b * 10 elements.
    pair<Tensor, Tensor> nextBatch() {
        assert(hasNext());
        size_t start = current_index;
        size_t end = min(current_index + static_cast<size_t>(batch_size), indices.size());
        int actual_batch = end - start;

        // Assume all images have the same size and labels have fixed size 10.
        const CIFAR10Sample &firstSample = dataset.getSample(indices[0]);
        int image_size = firstSample.image.size();
        int label_size = firstSample.label.size(); // should be 10

        Tensor batch_images(static_cast<size_t>(actual_batch * image_size), firstSample.image.device());
        Tensor batch_labels(static_cast<size_t>(actual_batch * label_size), firstSample.label.device());

        float *images_data = batch_images.data();
        float *labels_data = batch_labels.data();

        for (size_t i = start; i < end; i++) {
            const CIFAR10Sample &sample = dataset.getSample(indices[i]);
            const float *srcImg = sample.image.data();
            copy(srcImg, srcImg + image_size, images_data);
            images_data += image_size;

            const float *srcLab = sample.label.data();
            copy(srcLab, srcLab + label_size, labels_data);
            labels_data += label_size;
        }

        current_index = end;
        return {batch_images, batch_labels};
    }
};