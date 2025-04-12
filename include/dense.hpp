/**
 * @file dense.hpp
 * @brief Defines a fully connected (Dense) layer for neural networks with autograd support.
 *
 * The Dense layer performs a matrix multiplication followed by bias addition:
 *     output = input × weight^T + bias
 *
 * This file includes:
 * - `Dense` class definition, inheriting from `SingleInputModule`
 * - Learnable parameters: weight and bias
 * - CUDA-compatible forward computation
 */

#pragma once

#include "single_input_module.hpp"
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

/**
 * @brief Fully connected linear layer (also known as Dense or Linear layer).
 *
 * Given an input of shape (batch_size × input_dim), this layer computes:
 *     output = input × weight^T + bias
 * where:
 * - `weight` has shape (output_dim × input_dim)
 * - `bias` has shape (output_dim)
 */
class Dense : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    VarPtr weight; ///< Learnable weight tensor of shape (output_dim × input_dim)
    VarPtr bias;   ///< Learnable bias tensor of shape (output_dim)

    int input_dim;  ///< Number of input features
    int output_dim; ///< Number of output features

    /**
     * @brief Constructs a Dense layer with given input and output dimensions.
     * @param input_dim Size of each input sample.
     * @param output_dim Size of each output sample.
     * @param device The device to allocate tensors on (CPU or CUDA).
     */
    Dense(int input_dim, int output_dim, Device device = Device::CPU);

    /// Default destructor.
    ~Dense() = default;

    /**
     * @brief Applies the linear transformation to the input variable.
     * @param input Variable with shape (batch_size × input_dim)
     * @return Output variable with shape (batch_size × output_dim)
     */
    VarPtr forward(const VarPtr &input) override;

    /**
     * @brief Returns all learnable parameters of the layer.
     * @return Vector containing weight and bias.
     */
    vector<VarPtr> parameters() override;
};