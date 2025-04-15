/**
 * @file cifar10_dataloader.hpp
 * @brief Provides a DataLoader interface for batching CIFAR-10 data samples.
 *
 * This utility wraps a CIFAR-10 dataset and supports:
 * - Randomized shuffling of sample indices for each epoch
 * - Batched retrieval of image-label pairs
 * - Concatenation of image and label data into flat tensors for model consumption
 * - Resetting for new training epochs
 */

#pragma once

#include "./benchmark/cifar10_dataset.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
namespace kernelnet {
namespace data {
/**
 * @brief Batches samples from a CIFAR-10 dataset with optional shuffling.
 *
 * This class provides an interface for mini-batch loading of CIFAR-10 samples.
 * Images and labels are flattened into contiguous tensors suitable for model input.
 */
class CIFAR10DataLoader {
  public:
    CIFAR10Dataset &dataset; ///< Reference to the underlying CIFAR-10 dataset
    int batch_size;          ///< Number of samples per batch
    bool shuffle;            ///< Whether to shuffle sample indices between epochs
    vector<size_t> indices;  ///< Shuffled indices for sample access
    size_t current_index;    ///< Current read index into the shuffled list

    /**
     * @brief Constructor for initializing the DataLoader.
     * @param dataset Reference to the CIFAR-10 dataset.
     * @param batch_size Desired number of samples per batch.
     * @param shuffle Whether to shuffle the dataset at the start of each epoch (default true).
     */
    CIFAR10DataLoader(CIFAR10Dataset &dataset, int batch_size, bool shuffle = true)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_index(0) {
        reset();
    }

    /**
     * @brief Reset the loader state to the beginning of a new epoch.
     *
     * Resets `current_index` to 0 and optionally reshuffles the indices.
     */
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

    /**
     * @brief Check if there are more batches to retrieve in the current epoch.
     * @return True if another batch is available, false otherwise.
     */
    bool hasNext() const {
        return current_index + batch_size <= indices.size();
    }

    /**
     * @brief Fetch the next mini-batch of samples.
     *
     * Each image in the batch is flattened and concatenated along the first dimension.
     * Labels are one-hot encoded and flattened similarly.
     *
     * @return A pair of Tensors: {batch_images, batch_labels}.
     *         - batch_images: Tensor of shape (batch_size * 3072)
     *         - batch_labels: Tensor of shape (batch_size * 10)
     */
    pair<Tensor, Tensor> nextBatch() {
        assert(hasNext());
        size_t start = current_index;
        size_t end = min(current_index + static_cast<size_t>(batch_size), indices.size());
        int actual_batch = end - start;

        // Use first sample to determine sizes.
        const CIFAR10Sample &firstSample = dataset.getSample(indices[0]);
        int image_size = firstSample.image.size();
        int label_size = firstSample.label.size();

        Tensor batch_images(static_cast<size_t>(actual_batch * image_size), firstSample.image.device());
        Tensor batch_labels(static_cast<size_t>(actual_batch * label_size), firstSample.label.device());

        float *images_data = batch_images.data();
        float *labels_data = batch_labels.data();

        // Copy sample data into batch tensors.
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
} // namespace data
} // namespace kernelnet