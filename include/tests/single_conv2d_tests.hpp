#pragma once

#include "autograd.hpp"
#include "conv2d.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"

using namespace std;

inline void runSingleConv2DTests() {
    // ============================================================
    // --------------------- Test on CPU --------------------------
    // ============================================================
    {
        cout << "===== Running Single Conv2D Test on CPU =====" << endl;
        Device device = CPU;

        // Input: 1-channel 4x4 image, batch size = 1.
        const int batch_size = 1;
        const int in_channels = 1;
        const int img_height = 4;
        const int img_width = 4;
        size_t input_size = batch_size * in_channels * img_height * img_width;

        // Create synthetic input tensor on CPU.
        // Values 1,2,...,16 in row-major order.
        Tensor X_tensor(input_size, CPU);
        for (int i = 0; i < static_cast<int>(input_size); i++) {
            X_tensor.data()[i] = static_cast<float>(i + 1);
        }
        // No transfer needed for CPU.
        auto X = make_shared<Variable>(X_tensor, false, "input");

        // Create target tensor Y on CPU.
        // Expected output for a 3x3 kernel (all-ones kernel) and zero bias:
        // For a 4x4 input, valid convolution (stride=1, padding=0) produces a 2x2 output.
        // The expected output is:
        // [54, 63, 90, 99]
        Tensor Y_tensor(batch_size * 1 * 2 * 2, CPU);
        Y_tensor.data()[0] = 54.0f;
        Y_tensor.data()[1] = 63.0f;
        Y_tensor.data()[2] = 90.0f;
        Y_tensor.data()[3] = 99.0f;

        // Build the Conv2D layer on CPU.
        // Conv2D parameters: in_channels=1, out_channels=1, kernel=3x3, stride=1, padding=0.
        Conv2D conv(in_channels, 1, 3, 3, img_height, img_width, 1, 0, device);

        // Collect parameters and create the optimizer.
        vector<VarPtr> params = conv.parameters();
        // Learning rate might need adjustment.
        SGD optimizer(params, 0.0001f);

        const int epochs = 2000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            auto pred = conv.forward(X); // Output shape: [1, 1, 2, 2]
            auto loss = MSEFunction::apply(pred, Y_tensor);
            Tensor grad_one(loss->data.size(), loss->data.device());
            grad_one.fill(1.0f);
            loss->backward(grad_one);

            optimizer.step();
            optimizer.zero_grad();

            if (epoch % 200 == 0) {
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }

        cout << "Ground Truth (CPU):" << endl;
        Y_tensor.print();

        auto final_pred = conv.forward(X);
        final_pred->data.toCPU();
        cout << "Final Prediction (CPU):" << endl;
        final_pred->data.print();
        cout << endl;
    }

    // ============================================================
    // --------------------- Test on CUDA -------------------------
    // ============================================================
    {
        cout << "===== Running Single Conv2D Test on CUDA =====" << endl;
        Device device = CUDA;

        // Input: 1-channel 4x4 image, batch size = 1.
        const int batch_size = 1;
        const int in_channels = 1;
        const int img_height = 4;
        const int img_width = 4;
        size_t input_size = batch_size * in_channels * img_height * img_width;

        // Create synthetic input tensor on CPU then transfer to CUDA.
        Tensor X_tensor(input_size, CPU);
        for (int i = 0; i < static_cast<int>(input_size); i++) {
            X_tensor.data()[i] = static_cast<float>(i + 1);
        }
        X_tensor.toCUDA();
        auto X = make_shared<Variable>(X_tensor, false, "input");

        // Create target tensor Y on CPU and then transfer to CUDA.
        // Expected output: [54, 63, 90, 99] arranged as a 2x2 matrix.
        Tensor Y_tensor(batch_size * 1 * 2 * 2, CPU);
        Y_tensor.data()[0] = 54.0f;
        Y_tensor.data()[1] = 63.0f;
        Y_tensor.data()[2] = 90.0f;
        Y_tensor.data()[3] = 99.0f;
        Y_tensor.toCUDA();

        // Build the Conv2D layer on CUDA.
        Conv2D conv(in_channels, 1, 3, 3, img_height, img_width, 1, 0, device);

        // Collect parameters and create the optimizer.
        vector<VarPtr> params = conv.parameters();
        SGD optimizer(params, 0.0001f);

        const int epochs = 2000;
        for (int epoch = 0; epoch < epochs; epoch++) {
            auto pred = conv.forward(X); // Output shape: [1, 1, 2, 2]
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

        cout << "Ground Truth (CUDA):" << endl;
        Y_tensor.toCPU();
        Y_tensor.print();

        auto final_pred = conv.forward(X);
        final_pred->data.toCPU();
        cout << "Final Prediction (CUDA):" << endl;
        final_pred->data.print();
        cout << endl;
    }
}