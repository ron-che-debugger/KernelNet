/**
 * @file embedding.hpp
 * @brief Defines the Embedding layer and its autograd-compatible embedding lookup function.
 *
 * This file provides:
 *  - EmbeddingLookupFunction: An internal autograd function that performs the embedding lookup
 *    and caches the input (token indices) along with the derived indices for backward computation.
 *  - Embedding: A module that converts token indices into dense embedding vectors via a learnable
 *    weight matrix.
 *
 * The embedding layer maps token indices to rows of the embedding matrix.
 */

#pragma once

#include "single_input_module.hpp"
#include <cuda_runtime.h>
#include <memory>

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;

namespace kernelnet {
namespace nn {

/**
 * @brief Internal autograd function for embedding lookup and gradient accumulation.
 *
 * This function saves the input variable (token indices) and its derived integer indices.
 * During backward propagation, it uses the saved indices to accumulate gradients for the
 * corresponding rows of the embedding weight.
 */
class EmbeddingLookupFunction : public Function {
  public:
    vector<int> indices; ///< Saved token indices derived from the input.
    int embed_dim;       ///< Dimensionality of the embeddings.
    VarPtr saved_weight; ///< Embedding weight variable.

    /**
     * @brief Applies the embedding lookup.
     *
     * Extracts token indices from the input, performs a row lookup on the embedding weight,
     * and caches the input for the backward pass.
     *
     * @param indices Input variable containing token indices.
     * @param weight Embedding weight variable.
     * @param embed_dim Dimensionality of the embedding vectors.
     * @return Output variable containing the looked-up embeddings.
     */
    static VarPtr apply(const VarPtr &indices,
                        const VarPtr &weight,
                        int embed_dim);

    /**
     * @brief Backward pass: computes the gradient with respect to the embedding weight.
     *
     * The gradient for the non-differentiable token indices is returned as a dummy tensor.
     * For each token in the batch, its gradient slice is accumulated into the corresponding
     * row of the weight gradient.
     *
     * @param grad_output Upstream gradient tensor with shape (n * embed_dim).
     * @return A vector of tensors: a dummy gradient for the indices and the gradient for the weight.
     */
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Embedding layer.
 *
 * Converts token indices into dense embedding vectors using a learnable weight matrix.
 */
class Embedding : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    int vocab_size; ///< Number of tokens.
    int embed_dim;  ///< Dimensionality of the embedding vectors.
    VarPtr weight;  ///< Learnable embedding weight matrix (flattened).

    /**
     * @brief Constructs an embedding layer.
     * @param vocab_size Number of tokens.
     * @param embed_dim Dimensionality of embedding vectors.
     * @param dev Device on which to allocate the weight (CPU or CUDA).
     */
    Embedding(int vocab_size, int embed_dim, Device dev = CPU);

    /**
     * @brief Forward pass: performs an embedding lookup.
     * @param input Input variable containing flattened token indices.
     * @return Output variable containing the corresponding dense embeddings.
     */
    VarPtr forward(const VarPtr &input) override;

    /**
     * @brief Returns the learnable parameters of the layer.
     * @return Vector containing the embedding weight variable.
     */
    vector<VarPtr> parameters() override;
};

} // namespace nn
} // namespace kernelnet