#pragma once

#include "autograd.hpp"
#include "conv2d.hpp"
#include "dense.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"

using namespace std;

inline void runSingleConv2DDenseTests() {
    srand(42);

    // --------------------- Test on CPU ---------------------
    {
        cout << "===== Running Single Conv2D + Dense NN Test on CPU =====" << endl;
        Device device = CPU;

        // Synthetic image: 1-channel 4x4 images, batch size 2.
        const int batch_size = 2;
        const int in_channels = 1;
        const int img_height = 4;
        const int img_width = 4;
        size_t input_size = batch_size * in_channels * img_height * img_width;

        // Conv2D layer parameters.
        const int conv_out_channels = 2;
        const int kernel_size = 3;
        int stride = 1;
        int padding = 0;
        int conv_out_height = (img_height - kernel_size + 2 * padding) / stride + 1; // 2
        int conv_out_width = (img_width - kernel_size + 2 * padding) / stride + 1;   // 2

        // Dense layer parameters.
        const int dense_input_dim = conv_out_channels * conv_out_height * conv_out_width;
        const int dense_output_dim = 1;

        // Create synthetic input tensor on CPU.
        Tensor X_tensor(batch_size * in_channels * img_height * img_width, CPU);
        for (int i = 0; i < batch_size * in_channels * img_height * img_width; i++) {
            X_tensor.data()[i] = static_cast<float>(i % 16);
        }
        // Create target tensor Y on CPU.
        // Here we compute the sum of image pixels for each image.
        Tensor Y_tensor(batch_size * dense_output_dim, CPU);
        for (int b = 0; b < batch_size; b++) {
            float sum_val = 0.0f;
            for (int i = 0; i < in_channels * img_height * img_width; i++) {
                sum_val += X_tensor.data()[b * in_channels * img_height * img_width + i];
            }
            Y_tensor.data()[b] = sum_val;
        }
        // Build the network on CPU.
        Conv2D conv(in_channels, conv_out_channels, kernel_size, kernel_size,
                    img_height, img_width, stride, padding, device);
        Dense dense(dense_input_dim, dense_output_dim, device);

        // Collect parameters.
        vector<VarPtr> params;
        vector<VarPtr> conv_params = conv.parameters();
        vector<VarPtr> dense_params = dense.parameters();
        params.insert(params.end(), conv_params.begin(), conv_params.end());
        params.insert(params.end(), dense_params.begin(), dense_params.end());
        SGD optimizer(params, 0.0001f);

        const int epochs = 2000;
        VarPtr dense_input; // Will hold conv output.
        // Training loop on CPU.
        for (int epoch = 0; epoch < epochs; epoch++) {
            auto conv_out = conv.forward(make_shared<Variable>(X_tensor, false, "input"));
            dense_input = conv_out;
            auto pred = dense.forward(dense_input);
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

        cout << "Ground Truth (CPU): ";
        Y_tensor.print();

        auto final_pred = dense.forward(dense_input);
        final_pred->data.toCPU();
        cout << "Final Prediction (CPU): ";
        final_pred->data.print();
    }

    // --------------------- Test on CUDA ---------------------
    {
        cout << "\n===== Running Single Conv2D + Dense NN Test on CUDA =====" << endl;
        Device device = CUDA;

        // Synthetic image: 1-channel 4x4 images, batch size 2.
        const int batch_size = 2;
        const int in_channels = 1;
        const int img_height = 4;
        const int img_width = 4;
        size_t input_size = batch_size * in_channels * img_height * img_width;

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

        // Create synthetic input tensor on CPU.
        Tensor X_tensor(batch_size * in_channels * img_height * img_width, CPU);
        for (int i = 0; i < batch_size * in_channels * img_height * img_width; i++) {
            X_tensor.data()[i] = static_cast<float>(i % 16);
        }
        // Create target tensor Y on CPU.
        Tensor Y_tensor(batch_size * dense_output_dim, CPU);
        for (int b = 0; b < batch_size; b++) {
            float sum_val = 0.0f;
            // Use the CPU data before transferring X to CUDA.
            for (int i = 0; i < in_channels * img_height * img_width; i++) {
                sum_val += X_tensor.data()[b * in_channels * img_height * img_width + i];
            }
            Y_tensor.data()[b] = sum_val;
        }
        // Now transfer both input and target to CUDA.
        X_tensor.toCUDA();
        Y_tensor.toCUDA();
        auto X = make_shared<Variable>(X_tensor, false, "input");

        // Build the network on CUDA.
        Conv2D conv(in_channels, conv_out_channels, kernel_size, kernel_size,
                    img_height, img_width, stride, padding, device);
        Dense dense(dense_input_dim, dense_output_dim, device);

        // Collect parameters.
        vector<VarPtr> params;
        vector<VarPtr> conv_params = conv.parameters();
        vector<VarPtr> dense_params = dense.parameters();
        params.insert(params.end(), conv_params.begin(), conv_params.end());
        params.insert(params.end(), dense_params.begin(), dense_params.end());
        SGD optimizer(params, 0.0001f);

        const int epochs = 2000;
        VarPtr dense_input; // Will hold conv output.
        // Training loop on CUDA.
        for (int epoch = 0; epoch < epochs; epoch++) {
            auto conv_out = conv.forward(X);
            dense_input = conv_out;
            auto pred = dense.forward(dense_input);
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

        cout << "Ground Truth (CUDA): ";
        {
            Tensor target_cpu = Y_tensor;
            if (target_cpu.device() != CPU) {
                target_cpu.toCPU();
            }
            target_cpu.print();
        }

        auto final_pred = dense.forward(dense_input);
        final_pred->data.toCPU();
        cout << "Final Prediction (CUDA): ";
        final_pred->data.print();
    }
}