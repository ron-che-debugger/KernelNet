/**
 * @file softmax.hpp
 * @brief Defines the Softmax activation and its autograd-compatible function.
 *
 * This file includes:
 * - `SoftmaxFunction`: Autograd logic for forward/backward softmax
 * - `Softmax`: A wrapper module usable in models for classification outputs
 */

#pragma once

#include "api_header.hpp"
#include "single_input_module.hpp"
#include <algorithm>
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
 * @brief Autograd-compatible softmax function.
 *
 * Applies softmax along the last dimension (class scores) of each input sample:
 *     softmax(x_i) = exp(x_i) / sum_j exp(x_j)
 *
 * Stores both input and output for backward computation.
 */
class KERNELNET_API SoftmaxFunction : public Function {
  public:
    int batch_size;        ///< Number of samples
    int num_classes;       ///< Number of classes per sample
    Tensor softmax_output; ///< Cached output of the softmax
    VarPtr saved_input;    ///< Input variable saved for backward

    /**
     * @brief Applies softmax to the input tensor.
     *
     * Assumes input shape is (batch_size × num_classes), flattened to 1D.
     *
     * @param input Input variable with shape [batch_size * num_classes].
     * @param batch_size Number of samples in the batch.
     * @param num_classes Number of class scores per sample.
     * @return Output variable after softmax.
     */
    static VarPtr apply(const VarPtr &input, int batch_size, int num_classes);

    /**
     * @brief Computes the gradient of softmax.
     *
     * Given the gradient from the next layer, computes:
     *     dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
     * where y = softmax(x)
     *
     * @param grad_output Gradient of the loss w.r.t. softmax output.
     * @return A vector with one tensor: gradient with respect to input.
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Softmax module usable in model definitions.
 *
 * Wraps `SoftmaxFunction` and exposes it through the standard module interface.
 */
class KERNELNET_API Softmax : public SingleInputModule {
    using SingleInputModule::forward;

  public:
    int batch_size;  ///< Number of input samples
    int num_classes; ///< Number of classes per input

    /**
     * @brief Constructs a Softmax module.
     * @param batch_size Number of samples.
     * @param num_classes Number of class scores per sample.
     */
    Softmax(int batch_size, int num_classes);

    /**
     * @brief Applies softmax to the input tensor.
     * @param input Variable with flattened shape [batch_size * num_classes].
     * @return Variable after softmax.
     */
    VarPtr forward(const VarPtr &input) override;
};
} // namespace nn
} // namespace kernelnet