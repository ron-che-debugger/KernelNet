#pragma once

#include "autograd.hpp"
#include "conv2d.hpp"
#include "optimizer.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

inline void runSingleConv2DTests() {
    srand(42);
    cout << "=== Training Conv2D layer only with CUDA ===" << endl;

    // Use CUDA.
    Device device = CUDA;

    // Input: 1-channel 4x4 image, batch size = 1.
    const int batch_size = 1;
    const int in_channels = 1;
    const int img_height = 4;
    const int img_width = 4;

    // Conv2D layer parameters.
    const int out_channels = 1; // Single output channel.
    const int kernel_h = 3;
    const int kernel_w = 3;
    int stride = 1;
    int padding = 0;
    // For a 4x4 input with 3x3 kernel, stride=1, padding=0 => output is 2x2.
    int out_height = (img_height - kernel_h + 2 * padding) / stride + 1; // 2
    int out_width = (img_width - kernel_w + 2 * padding) / stride + 1;   // 2

    // Create synthetic input tensor on CPU.
    // Values 1,2,...,16 in row-major order.
    Tensor X_tensor(batch_size * in_channels * img_height * img_width, CPU);
    for (int i = 0; i < batch_size * in_channels * img_height * img_width; i++) {
        X_tensor.data()[i] = static_cast<float>(i + 1);
    }
    // Transfer input to CUDA.
    X_tensor.toCUDA();
    auto X = make_shared<Variable>(X_tensor, false);

    // Create target tensor Y on CPU.
    // Expected output for a 3x3 all-ones kernel and zero bias:
    // [54, 63, 90, 99] arranged as a 2x2 matrix.
    Tensor Y_tensor(batch_size * out_channels * out_height * out_width, CPU);
    Y_tensor.data()[0] = 54.0f;
    Y_tensor.data()[1] = 63.0f;
    Y_tensor.data()[2] = 90.0f;
    Y_tensor.data()[3] = 99.0f;
    Y_tensor.toCUDA();

    cout << "Ground Truth Output:" << endl;
    Y_tensor.toCPU();
    Y_tensor.print();
    Y_tensor.toCUDA();

    // Build the Conv2D layer.
    Conv2D conv(in_channels, out_channels, kernel_h, kernel_w, img_height, img_width, stride, padding, device);

    // Collect Conv2D parameters and create the optimizer.
    vector<VarPtr> params = conv.parameters();
    // You might need to adjust the learning rate based on your implementation.
    SGD optimizer(params, 0.0001f);

    const int epochs = 2000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass through Conv2D.
        auto pred = conv.forward(X); // Output shape: [1, 1, 2, 2]
        // Compute MSE loss between prediction and ground truth.
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

    // Final prediction.
    auto final_pred = conv.forward(X);
    final_pred->data.toCPU();
    cout << "Final predictions:" << endl;
    final_pred->data.print();
}