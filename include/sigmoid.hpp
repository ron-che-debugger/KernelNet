/**
 * @file sigmoid.hpp
 * @brief Defines the Sigmoid activation module and its autograd-compatible function.
 *
 * This file provides:
 * - `SigmoidFunction`: Forward and backward logic for the sigmoid function
 * - `Sigmoid`: A user-facing module wrapper for easy integration in models
 */

#pragma once

#include "single_input_module.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

/**
 * @brief Autograd-compatible implementation of the Sigmoid activation.
 *
 * Sigmoid is defined as:
 *     sigmoid(x) = 1 / (1 + exp(-x))
 *
 * Stores input and output for backward computation.
 */
class SigmoidFunction : public Function {
  public:
    VarPtr saved_input;    ///< Saved input variable
    Tensor sigmoid_output; ///< Cached forward output

    /**
     * @brief Applies the sigmoid function element-wise to the input.
     * @param input Input variable.
     * @return Output variable with sigmoid applied.
     */
    static VarPtr apply(const VarPtr &input);

    /**
     * @brief Backward pass for sigmoid.
     *
     * Computes:
     *     dL/dx = dL/dy * y * (1 - y)
     * where y = sigmoid(x)
     *
     * @param grad_output Gradient flowing from the next layer.
     * @return A vector with one tensor: the gradient with respect to the input.
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Sigmoid activation module.
 */
class Sigmoid : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    /// Constructs a Sigmoid activation module.
    Sigmoid();

    /**
     * @brief Applies sigmoid to the input variable.
     * @param input Input tensor variable.
     * @return Output variable after sigmoid.
     */
    VarPtr forward(const VarPtr &input) override;
};