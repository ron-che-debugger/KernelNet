/**
 * @file ptb_data_loader.hpp
 * @brief Provides a DataLoader interface for batching PTB samples.
 *
 * Samples are batched and flattened into contiguous tensors.
 * The loader supports shuffling sample indices between epochs.
 */

#pragma once

#include "./benchmark/ptb_dataset.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

namespace kernelnet {
namespace data {

class PTBDataLoader {
  public:
    PTBDataset &dataset;    ///< Reference to the PTB dataset.
    int batch_size;         ///< Number of samples per batch.
    bool shuffle;           ///< Whether to shuffle at epoch start.
    vector<size_t> indices; ///< Shuffled sample indices.
    size_t current_index;   ///< Current index in the shuffled list.

    /**
     * @brief Constructor.
     * @param dataset Reference to a PTBDataset.
     * @param batch_size Number of samples per batch.
     * @param shuffle Whether to shuffle (default: true).
     */
    PTBDataLoader(PTBDataset &dataset, int batch_size, bool shuffle = true)
        : dataset(dataset), batch_size(batch_size), shuffle(shuffle), current_index(0) {
        reset();
    }

    /**
     * @brief Resets the loader for a new epoch.
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
     * @brief Checks if another batch is available in the current epoch.
     * @return True if more batches exist.
     */
    bool hasNext() const {
        return current_index + batch_size <= indices.size();
    }

    /**
     * @brief Retrieves the next mini-batch.
     *
     * Each batch consists of a pair of flattened tensors:
     * {batch_inputs, batch_targets} with shape (batch_size * sequence_length).
     *
     * @return A pair of Tensors.
     */
    pair<Tensor, Tensor> nextBatch() {
        assert(hasNext());
        size_t start = current_index;
        size_t end = min(current_index + static_cast<size_t>(batch_size), indices.size());
        int actual_batch = end - start;

        // Use the first sample to determine sequence length.
        const PTBSample &firstSample = dataset.getSample(indices[0]);
        int seq_len = firstSample.input.size();

        Tensor batch_inputs(static_cast<size_t>(actual_batch * seq_len), firstSample.input.device());
        Tensor batch_targets(static_cast<size_t>(actual_batch * seq_len), firstSample.input.device());

        float *inputs_data = batch_inputs.data();
        float *targets_data = batch_targets.data();

        for (size_t i = start; i < end; i++) {
            const PTBSample &sample = dataset.getSample(indices[i]);
            const float *src_inp = sample.input.data();
            copy(src_inp, src_inp + seq_len, inputs_data);
            inputs_data += seq_len;
            const float *src_tgt = sample.target.data();
            copy(src_tgt, src_tgt + seq_len, targets_data);
            targets_data += seq_len;
        }

        current_index = end;
        return {batch_inputs, batch_targets};
    }
};

} // namespace data
} // namespace kernelnet