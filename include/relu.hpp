/**
 * @file relu.hpp
 * @brief Defines the ReLU activation function and its autograd support.
 *
 * This file provides:
 * - `ReLUFunction`: Autograd-compatible forward and backward for ReLU
 * - `ReLU`: A module wrapper around `ReLUFunction` for model usage
 */

#pragma once

#include "single_input_module.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;

namespace kernelnet {
namespace nn {
/**
 * @brief Autograd-compatible implementation of the ReLU activation.
 *
 * ReLU is defined as:
 *     ReLU(x) = max(0, x)
 *
 * Saves input and output for backward computation.
 */
class ReLUFunction : public Function {
  public:
    VarPtr saved_input; ///< Input variable for backward
    Tensor relu_output; ///< Cached output tensor from forward

    /**
     * @brief Applies the ReLU function.
     * @param input Input variable.
     * @return Output variable with ReLU applied.
     */
    static VarPtr apply(const VarPtr &input);

    /**
     * @brief Computes the gradient of ReLU.
     *
     * Propagates the gradient through non-zero input elements only:
     *     dL/dx = dL/dy if x > 0 else 0
     *
     * @param grad_output Gradient from the next layer.
     * @return A vector with one tensor: gradient with respect to input.
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief ReLU module wrapper for use in neural networks.
 */
class ReLU : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    /// Constructs a ReLU module.
    ReLU();

    /**
     * @brief Applies ReLU to the input variable.
     * @param input Input tensor variable.
     * @return Output after ReLU.
     */
    VarPtr forward(const VarPtr &input) override;
};
} // namespace nn
} // namespace kernelnet