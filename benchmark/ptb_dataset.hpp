/**
 * @file ptb_dataset.hpp
 * @brief Loads Penn Treebank text, builds vocabulary, and creates fixed-length sequence samples.
 *
 * This module loads a PTB text file, tokenizes the text,
 * builds a mapping from words to indices, and splits the text into samples.
 * Each sample consists of an input sequence and a target sequence (shifted by one).
 */

#pragma once

#include "tensor.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace kernelnet::tensor;

namespace kernelnet {
namespace data {

/// Structure to hold a single PTB sample.
struct PTBSample {
    Tensor input;  ///< Tensor of token indices (flat tensor of length = sequence_length)
    Tensor target; ///< Tensor of token indices (flat tensor of length = sequence_length)
};

/// Loads the PTB dataset from a text file and creates samples.
class PTBDataset {
  public:
    vector<PTBSample> samples;    ///< All generated samples
    map<string, int> word_to_idx; ///< Vocabulary: word to integer index
    vector<string> idx_to_word;   ///< Vocabulary: index to word
    int vocab_size;               ///< Number of unique tokens
    int sequence_length;          ///< Fixed sequence length for each sample

    /**
     * @brief Constructor: loads the PTB file and creates samples.
     * @param file Path to PTB text file.
     * @param sequence_length Desired sequence length.
     */
    PTBDataset(const string &file, int sequence_length) : sequence_length(sequence_length) {
        loadFile(file);
    }

    /**
     * @brief Returns the number of samples.
     * @return Number of samples.
     */
    size_t size() const {
        return samples.size();
    }

    /**
     * @brief Get a sample by index.
     * @param index Index of the sample.
     * @return Const reference to PTBSample.
     */
    const PTBSample &getSample(size_t index) const {
        assert(index < samples.size());
        return samples[index];
    }

  private:
    // Build vocabulary from tokens.
    void buildVocabulary(const vector<string> &tokens) {
        for (const auto &token : tokens) {
            if (word_to_idx.find(token) == word_to_idx.end()) {
                int idx = word_to_idx.size();
                word_to_idx[token] = idx;
                idx_to_word.push_back(token);
            }
        }
        vocab_size = word_to_idx.size();
    }

    // Load the file, tokenize, build vocabulary, and create samples.
    void loadFile(const string &filename) {
        ifstream fin(filename);
        if (!fin.is_open())
            throw runtime_error("Could not open PTB file: " + filename);

        string line;
        vector<string> tokens;
        while (getline(fin, line)) {
            istringstream iss(line);
            string word;
            while (iss >> word) {
                tokens.push_back(word);
            }
        }
        fin.close();

        // Build vocabulary.
        buildVocabulary(tokens);

        // Convert tokens to indices.
        vector<int> indices;
        for (const auto &w : tokens) {
            indices.push_back(word_to_idx[w]);
        }

        // Create samples: input is indices[i ... i+sequence_length-1] and target is indices[i+1 ... i+sequence_length]
        size_t num_samples = (indices.size() - 1) / sequence_length;
        samples.reserve(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            size_t start = i * sequence_length;
            Tensor input_tensor(static_cast<size_t>(sequence_length), CPU);
            Tensor target_tensor(static_cast<size_t>(sequence_length), CPU);
            float *inp_data = input_tensor.data();
            float *tgt_data = target_tensor.data();
            for (size_t j = 0; j < static_cast<size_t>(sequence_length); j++) {
                inp_data[j] = static_cast<float>(indices[start + j]);
                tgt_data[j] = static_cast<float>(indices[start + j + 1]);
            }
            PTBSample sample;
            sample.input = input_tensor;
            sample.target = target_tensor;
            samples.push_back(sample);
        }

        cout << "Loaded " << samples.size() << " samples from " << filename
             << ". Vocabulary size: " << vocab_size << endl;
    }
};

} // namespace data
} // namespace kernelnet