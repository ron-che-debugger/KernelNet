/**
 * @file maxpool.hpp
 * @brief Defines a 2D max pooling layer and its autograd-compatible function.
 *
 * This file includes:
 * - `MaxPool2D`: A module for spatial downsampling using max pooling.
 * - `MaxPool2DFunction`: Handles autograd forward/backward logic for max pooling.
 */

#pragma once

#include "single_input_module.hpp"
#include <cfloat>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

/**
 * @brief Autograd-compatible max pooling function.
 *
 * Stores the max indices for backward gradient routing. Assumes input shape is:
 *     [batch_size, channels, input_height, input_width]
 * and outputs pooled shape:
 *     [batch_size, channels, output_height, output_width]
 */
class MaxPool2DFunction : public Function {
  public:
    VarPtr saved_input; ///< Saved input variable for backward pass

    int batch_size;    ///< Number of input samples
    int channels;      ///< Number of channels
    int input_height;  ///< Height of input tensor
    int input_width;   ///< Width of input tensor
    int kernel_size;   ///< Size of square pooling window
    int stride;        ///< Stride of pooling window
    int output_height; ///< Computed output height
    int output_width;  ///< Computed output width

    vector<int> max_indices; ///< Max indices per output for gradient routing (CPU only)

    /**
     * @brief Performs the forward max pooling operation and stores metadata for backward.
     *
     * @param input Input tensor with shape (B, C, H, W).
     * @param batch_size Number of samples.
     * @param channels Number of input channels.
     * @param input_height Height of input image.
     * @param input_width Width of input image.
     * @param kernel_size Pooling window size.
     * @param stride Stride for pooling.
     * @return Output pooled variable with shape (B, C, H_out, W_out).
     */
    static VarPtr apply(const VarPtr &input,
                        int batch_size, int channels,
                        int input_height, int input_width,
                        int kernel_size, int stride);

    /**
     * @brief Backward pass for max pooling.
     *
     * Routes gradient only to the max position per pooling window.
     *
     * @param grad_output Gradient from next layer.
     * @return Vector with a single element: gradient for input.
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief Max pooling module used in CNNs for downsampling.
 *
 * Applies max pooling on input with specified kernel size and stride.
 */
class MaxPool2D : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    int kernel_size;  ///< Size of pooling kernel (square)
    int stride;       ///< Stride for window movement
    int batch_size;   ///< Number of input samples
    int channels;     ///< Number of channels in input
    int input_height; ///< Input image height
    int input_width;  ///< Input image width

    /**
     * @brief Constructs a MaxPool2D layer.
     *
     * @param kernel_size Size of pooling window (assumed square).
     * @param stride Stride for pooling operation.
     * @param batch_size Batch size of the input.
     * @param channels Number of channels per sample.
     * @param input_height Input height.
     * @param input_width Input width.
     */
    MaxPool2D(int kernel_size, int stride,
              int batch_size, int channels,
              int input_height, int input_width);

    /**
     * @brief Applies max pooling operation.
     * @param input Input variable of shape (B × C × H × W).
     * @return Pooled output variable.
     */
    VarPtr forward(const VarPtr &input) override;
};