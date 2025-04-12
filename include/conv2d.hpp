/**
 * @file conv2d.hpp
 * @brief Defines a 2D convolutional module with forward and backward CUDA-compatible operations.
 *
 * This file provides:
 * - `Conv2D` class: a learnable layer for convolution
 * - Forward and backward implementations for data, weight, and bias gradients
 * - A `Conv2DFunction` class for autograd integration
 */

#pragma once

#include "single_input_module.hpp"
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;

namespace kernelnet {
namespace nn {
/**
 * @brief Learnable 2D convolution layer with autograd support.
 */
class Conv2D : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    VarPtr weight; ///< Convolution kernel tensor: shape (out_channels, in_channels, kernel_h, kernel_w)
    VarPtr bias;   ///< Bias tensor: shape (out_channels)

    int in_channels, out_channels;
    int kernel_h, kernel_w;
    int stride, padding;
    int input_height, input_width;

    /**
     * @brief Constructs a Conv2D layer with given configuration.
     * @param in_channels Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_h Height of the convolutional kernel.
     * @param kernel_w Width of the convolutional kernel.
     * @param input_height Input tensor height (used for shape inference).
     * @param input_width Input tensor width.
     * @param stride Stride of the convolution (default 1).
     * @param padding Padding around the input (default 0).
     * @param device CPU or CUDA.
     */
    Conv2D(int in_channels, int out_channels, int kernel_h, int kernel_w,
           int input_height, int input_width, int stride = 1, int padding = 0, Device device = Device::CPU);

    /**
     * @brief Applies the convolution to the input tensor.
     * @param input A Variable containing the input tensor.
     * @return A Variable containing the output tensor.
     */
    VarPtr forward(const VarPtr &input) override;

    /**
     * @brief Returns learnable parameters (weight and bias).
     * @return Vector of learnable VarPtr parameters.
     */
    vector<VarPtr> parameters() override;
};

/**
 * @brief Forward pass for 2D convolution.
 *
 * @param input Input tensor (shape: [batch_size, in_channels, H, W]).
 * @param weight Weight tensor (shape: [out_channels, in_channels, kH, kW]).
 * @param bias Bias tensor (shape: [out_channels]).
 * @param batch_size Number of input samples.
 * @param in_channels Number of input channels.
 * @param input_height Height of input image.
 * @param input_width Width of input image.
 * @param out_channels Number of output channels.
 * @param kernel_h Height of the kernel.
 * @param kernel_w Width of the kernel.
 * @param stride Stride of convolution.
 * @param padding Zero padding size.
 * @param out_height Output height (precomputed).
 * @param out_width Output width (precomputed).
 * @return Output tensor after applying convolution.
 */
Tensor conv2d_forward(const Tensor &input, const Tensor &weight, const Tensor &bias,
                      int batch_size, int in_channels, int input_height, int input_width,
                      int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                      int out_height, int out_width);

/**
 * @brief Computes gradient with respect to bias given output gradient.
 * @param grad_output Gradient of the output.
 * @param batch_size Number of samples.
 * @param out_channels Number of output channels.
 * @param out_height Height of output tensor.
 * @param out_width Width of output tensor.
 * @return Gradient tensor for bias.
 */
Tensor conv2d_backward_bias(const Tensor &grad_output, int batch_size,
                            int out_channels, int out_height, int out_width);

/**
 * @brief Computes gradient with respect to weights.
 * @param grad_output Gradient of the output tensor.
 * @param input Original input tensor to the convolution.
 * @param batch_size Number of input samples.
 * @param in_channels Input channels.
 * @param input_height Height of input.
 * @param input_width Width of input.
 * @param out_channels Output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Stride.
 * @param padding Padding.
 * @param out_height Height of output.
 * @param out_width Width of output.
 * @return Gradient tensor for weights.
 */
Tensor conv2d_backward_weight(const Tensor &grad_output, const Tensor &input,
                              int batch_size, int in_channels, int input_height, int input_width,
                              int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                              int out_height, int out_width);

/**
 * @brief Computes gradient with respect to the input tensor.
 * @param grad_output Gradient from next layer.
 * @param weight Weight tensor of the Conv2D layer.
 * @param batch_size Number of samples.
 * @param in_channels Input channels.
 * @param input_height Height of input.
 * @param input_width Width of input.
 * @param out_channels Output channels.
 * @param kernel_h Kernel height.
 * @param kernel_w Kernel width.
 * @param stride Stride.
 * @param padding Padding.
 * @param out_height Output height.
 * @param out_width Output width.
 * @return Gradient tensor for input.
 */
Tensor conv2d_backward_input(const Tensor &grad_output, const Tensor &weight,
                             int batch_size, int in_channels, int input_height, int input_width,
                             int out_channels, int kernel_h, int kernel_w, int stride, int padding,
                             int out_height, int out_width);

/**
 * @brief Autograd wrapper for Conv2D forward/backward passes.
 *
 * Used internally by `Conv2D` to link forward results to gradient propagation.
 */
class Conv2DFunction : public Function {
  public:
    VarPtr saved_input;  ///< Saved input variable for backward.
    VarPtr saved_weight; ///< Saved weight variable for backward.
    VarPtr saved_bias;   ///< Saved bias variable for backward.

    int batch_size, in_channels, input_height, input_width;
    int out_channels, kernel_h, kernel_w;
    int stride, padding;
    int out_height, out_width;

    /**
     * @brief Forward apply function for autograd graph.
     * @param input Input variable.
     * @param weight Weight variable.
     * @param bias Bias variable.
     * @param in_channels Input channels.
     * @param input_height Input height.
     * @param input_width Input width.
     * @param out_channels Output channels.
     * @param kernel_h Kernel height.
     * @param kernel_w Kernel width.
     * @param stride Stride.
     * @param padding Padding.
     * @return Variable containing the convolution output.
     */
    static VarPtr apply(const VarPtr &input, const VarPtr &weight, const VarPtr &bias,
                        int in_channels, int input_height, int input_width,
                        int out_channels, int kernel_h, int kernel_w, int stride, int padding);

    /**
     * @brief Backward function that computes gradients for input, weight, and bias.
     * @param grad_output Gradient passed from next layer.
     * @return Vector of gradients for {input, weight, bias}.
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};
} // namespace nn
} // namespace kernelnet