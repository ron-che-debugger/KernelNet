/**
 * @file tanh.hpp
 * @brief Defines the Tanh activation function and its autograd-compatible module.
 *
 * Includes:
 * - `TanhFunction`: Autograd-aware forward and backward implementation of tanh
 * - `Tanh`: A module wrapper for integration into sequential models
 */

#pragma once

#include "single_input_module.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

/**
 * @brief Autograd-compatible implementation of the hyperbolic tangent (tanh) activation.
 *
 * The tanh function is defined as:
 *     tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 *
 * This class saves the input and output tensors for efficient backward computation.
 */
class TanhFunction : public Function {
  public:
    VarPtr saved_input; ///< Saved input variable for backward
    Tensor tanh_output; ///< Cached forward output of tanh(input)

    /**
     * @brief Applies the tanh function element-wise.
     * @param input Input variable.
     * @return Output variable after applying tanh.
     */
    static VarPtr apply(const VarPtr &input);

    /**
     * @brief Computes gradient of the tanh function.
     *
     * Uses:
     *     dL/dx = dL/dy * (1 - tanh(x)^2)
     *
     * @param grad_output Gradient from next layer.
     * @return A vector with one tensor: the gradient with respect to input.
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Tanh activation module.
 *
 * Wraps `TanhFunction` in a module interface for model usage.
 */
class Tanh : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    /// Constructs a Tanh module.
    Tanh();

    /**
     * @brief Applies the tanh function to the input.
     * @param input Input variable.
     * @return Output variable after tanh.
     */
    VarPtr forward(const VarPtr &input) override;
};