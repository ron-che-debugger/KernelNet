#pragma once

#include "autograd.hpp"
#include "conv2d.hpp"
#include "dense.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

// A simple test to train a CNN on a synthetic regression task using CUDA.
// The CNN consists of one Conv2D layer followed by a Dense (fully-connected) layer.
inline void runSingleConv2DDenseTests() {
    srand(42);

    // Use CUDA
    Device device = CUDA;

    // Synthetic image: 1-channel 4x4 images, batch size 2.
    const int batch_size = 2;
    const int in_channels = 1;
    const int img_height = 4;
    const int img_width = 4;

    // Conv2D layer parameters.
    const int conv_out_channels = 2;
    const int kernel_size = 3;
    int stride = 1;
    int padding = 0;
    int conv_out_height = (img_height - kernel_size + 2 * padding) / stride + 1;
    int conv_out_width = (img_width - kernel_size + 2 * padding) / stride + 1;

    // Dense layer parameters.
    const int dense_input_dim = conv_out_channels * conv_out_height * conv_out_width;
    const int dense_output_dim = 1;

    // Create synthetic input tensor on CPU and then transfer to CUDA.
    Tensor X_tensor(batch_size * in_channels * img_height * img_width, CPU);
    for (int i = 0; i < batch_size * in_channels * img_height * img_width; i++) {
        X_tensor.data()[i] = static_cast<float>(i % 16);
    }

    // Create target tensor Y on CPU (for example, sum of image pixels) and transfer to CUDA.
    Tensor Y_tensor(batch_size * dense_output_dim, CPU);
    for (int b = 0; b < batch_size; b++) {
        float sum_val = 0.0f;
        for (int i = 0; i < in_channels * img_height * img_width; i++) {
            sum_val += X_tensor.data()[b * in_channels * img_height * img_width + i];
        }
        Y_tensor.data()[b] = sum_val;
    }

    Y_tensor.toCUDA();

    X_tensor.toCUDA();
    auto X = make_shared<Variable>(X_tensor, false);

    // Build the CNN: a Conv2D layer then a Dense layer.
    Conv2D conv(in_channels, conv_out_channels, kernel_size, kernel_size, img_height, img_width, stride, padding, device);
    Dense dense(dense_input_dim, dense_output_dim, device);

    // Collect parameters.
    vector<VarPtr> params;
    vector<VarPtr> conv_params = conv.parameters();
    vector<VarPtr> dense_params = dense.parameters();
    params.insert(params.end(), conv_params.begin(), conv_params.end());
    params.insert(params.end(), dense_params.begin(), dense_params.end());

    SGD optimizer(params, 0.0001f);

    const int epochs = 2000;
    VarPtr dense_input; // Will hold the output of the conv layer.
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass through Conv2D.
        auto conv_out = conv.forward(X);
        dense_input = conv_out; // Shape: [batch_size, conv_out_channels * conv_out_height * conv_out_width]

        // Forward pass through Dense layer.
        auto pred = dense.forward(dense_input);

        // Compute MSE loss.
        auto loss = MSEFunction::apply(pred, Y_tensor);
        Tensor grad_one(loss->data.size(), loss->data.device());
        grad_one.fill(1.0f);
        loss->backward(grad_one);

        optimizer.step();
        optimizer.zero_grad();

        if (epoch % 200 == 0) {
            loss->data.toCPU();
            cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            loss->data.toCUDA();
        }
    }

    cout << "Ground Truth:" << endl;
    Y_tensor.toCPU();
    Y_tensor.print();

    auto final_pred = dense.forward(dense_input);
    final_pred->data.toCPU();
    cout << "Final Prediction:" << endl;
    final_pred->data.print();
}