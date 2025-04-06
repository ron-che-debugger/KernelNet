#pragma once
#include "autograd.hpp"
#include "conv2d.hpp"
#include "dense.hpp"
#include "maxpool.hpp"
#include "optimizer.hpp"
#include "softmax.hpp"
#include "tensor.hpp"

using namespace std;

inline void runSimpleCnnTests() {
    // --- Input Setup ---
    // Create an input: 1 batch, 1 channel, 8x8 image filled with ones.
    int batch_size = 1, in_channels = 1, height = 8, width = 8;
    size_t input_size = batch_size * in_channels * height * width;
    Tensor input(input_size, CPU);
    float *input_data = input.data();
    for (size_t i = 0; i < input_size; i++) {
        input_data[i] = 1.0f;
    }
    input.toCUDA(); // Transfer input to CUDA.
    VarPtr input_var = make_shared<Variable>(input, false);

    // --- Target Setup ---
    // For a 2-class classification, define target as a one-hot vector.
    int num_classes = 2;
    Tensor target(num_classes, CPU);
    // Ground truth: class 1, i.e. [0, 1]
    target.data()[0] = 0.0f;
    target.data()[1] = 1.0f;
    target.toCUDA();

    // --- Network Construction ---
    // Conv2D Layer 1: in_channels=1, out_channels=1, kernel=3x3, stride=1, padding=1 (maintains 8x8).
    Conv2D conv1(1, 1, 3, 3, height, width, 1, 1, CUDA);
    // MaxPool2D Layer 1: kernel=2, stride=2 reduces 8x8 -> 4x4.
    MaxPool2D pool1(2, 2, batch_size, 1, height, width);

    // Conv2D Layer 2: now input is 4x4; parameters: in_channels=1, out_channels=1, kernel=3x3, stride=1, padding=1.
    Conv2D conv2(1, 1, 3, 3, 4, 4, 1, 1, CUDA);
    // MaxPool2D Layer 2: reduces 4x4 -> 2x2.
    MaxPool2D pool2(2, 2, batch_size, 1, 4, 4);

    // Dense Layer: flatten the 2x2 output (4 elements) to output_dim=2.
    Dense dense(4, 2, CUDA);

    // Softmax Layer.
    Softmax softmax(batch_size, num_classes);

    // --- Optimizer Setup ---
    // Collect parameters from conv1, conv2, and dense layers.
    vector<VarPtr> params;
    {
        vector<VarPtr> p = conv1.parameters();
        params.insert(params.end(), p.begin(), p.end());
        p = conv2.parameters();
        params.insert(params.end(), p.begin(), p.end());
        p = dense.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    float learning_rate = 0.01f;
    SGD optimizer(params, learning_rate);

    // --- Training Loop ---
    int num_epochs = 2000;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass.
        VarPtr x = conv1.forward(input_var);
        x = pool1.forward(x);
        x = conv2.forward(x);
        x = pool2.forward(x);
        x = dense.forward(x);
        VarPtr predictions = softmax.forward(x);

        // Compute loss using MSE as a surrogate loss.
        VarPtr loss = MSEFunction::apply(predictions, target);

        // Backward pass.
        loss->backward(loss->data);

        // Update parameters.
        optimizer.step();
        optimizer.zero_grad();

        if (epoch % 100 == 0) {
            loss->data.toCPU();
            cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            loss->data.toCUDA();
        }
    }

    // --- Final Prediction ---
    // Run a final forward pass.
    VarPtr final_pred = softmax.forward(
        dense.forward(
            pool2.forward(
                conv2.forward(
                    pool1.forward(
                        conv1.forward(input_var))))));
    Tensor final_output = final_pred->data;
    if (final_output.device() != CPU) {
        final_output.toCPU();
    }
    const float *final_data = final_output.data();
    cout << "Final Prediction: ";
    for (int i = 0; i < num_classes; i++) {
        cout << final_data[i] << " ";
    }
    cout << endl;

    // Print Ground Truth again.
    cout << "Ground Truth: ";
    {
        Tensor target_cpu = target;
        if (target_cpu.device() != CPU) {
            target_cpu.toCPU();
        }
        for (int i = 0; i < num_classes; i++) {
            cout << target_cpu.data()[i] << " ";
        }
        cout << endl;
    }
}