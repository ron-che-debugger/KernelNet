#pragma once
#include "kernelnet.hpp"

using namespace std;

inline void runSimpleCnnTests() {
    // ------------------- Test on CPU -------------------
    cout << "===== Running Simple CNN Test on CPU =====" << endl;
    {
        Device dev = CPU;
        int batch_size = 1, in_channels = 1, height = 8, width = 8, num_classes = 2;
        size_t input_size = batch_size * in_channels * height * width;
        Tensor input(input_size, CPU);
        float *input_data = input.data();
        for (size_t i = 0; i < input_size; i++) {
            input_data[i] = 1.0f;
        }
        // For CPU, we leave input on CPU.
        VarPtr input_var = make_shared<Variable>(input, false, "input");

        // Target: one-hot vector [0, 1].
        Tensor target(num_classes, CPU);
        target.data()[0] = 0.0f;
        target.data()[1] = 1.0f;

        // Build network layers on CPU.
        Conv2D conv1(1, 1, 3, 3, height, width, 1, 1, dev);
        MaxPool2D pool1(2, 2, batch_size, 1, height, width);
        // conv2 processes a 4x4 feature map.
        Conv2D conv2(1, 1, 3, 3, 4, 4, 1, 1, dev);
        MaxPool2D pool2(2, 2, batch_size, 1, 4, 4);
        Dense dense(4, 2, dev);
        Softmax softmax(batch_size, num_classes);

        // Collect parameters.
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

        // Training loop.
        int num_epochs = 2000;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            VarPtr x = conv1.forward(input_var);
            x = pool1.forward(x);
            x = conv2.forward(x);
            x = pool2.forward(x);
            x = dense.forward(x);
            VarPtr predictions = softmax.forward(x);

            VarPtr loss = MSEFunction::apply(predictions, target);
            loss->backward(loss->data);
            optimizer.step();
            optimizer.zero_grad();

            if (epoch % 100 == 0) {
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }

        VarPtr final_pred = softmax.forward(
            dense.forward(
                pool2.forward(
                    conv2.forward(
                        pool1.forward(
                            conv1.forward(input_var))))));
        Tensor final_output = final_pred->data;
        const float *final_data = final_output.data();
        cout << "Final Prediction (CPU): ";
        for (int i = 0; i < num_classes; i++) {
            cout << final_data[i] << " ";
        }
        cout << endl;

        cout << "Ground Truth (CPU): ";
        for (int i = 0; i < num_classes; i++) {
            cout << target.data()[i] << " ";
        }
        cout << endl;
    }

    // ------------------- Test on CUDA -------------------
    cout << "\n===== Running Simple CNN Test on CUDA =====" << endl;
    {
        Device dev = CUDA;
        int batch_size = 1, in_channels = 1, height = 8, width = 8, num_classes = 2;
        size_t input_size = batch_size * in_channels * height * width;
        Tensor input(input_size, CPU);
        float *input_data = input.data();
        for (size_t i = 0; i < input_size; i++) {
            input_data[i] = 1.0f;
        }
        input.toCUDA();
        VarPtr input_var = make_shared<Variable>(input, false, "input");

        // Target: one-hot vector [0, 1].
        Tensor target(num_classes, CPU);
        target.data()[0] = 0.0f;
        target.data()[1] = 1.0f;
        target.toCUDA();

        // Build network layers on CUDA.
        Conv2D conv1(1, 1, 3, 3, height, width, 1, 1, dev);
        MaxPool2D pool1(2, 2, batch_size, 1, height, width);
        Conv2D conv2(1, 1, 3, 3, 4, 4, 1, 1, dev);
        MaxPool2D pool2(2, 2, batch_size, 1, 4, 4);
        Dense dense(4, 2, dev);
        Softmax softmax(batch_size, num_classes);

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

        int num_epochs = 2000;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            VarPtr x = conv1.forward(input_var);
            x = pool1.forward(x);
            x = conv2.forward(x);
            x = pool2.forward(x);
            x = dense.forward(x);
            VarPtr predictions = softmax.forward(x);

            VarPtr loss = MSEFunction::apply(predictions, target);
            loss->backward(loss->data);
            optimizer.step();
            optimizer.zero_grad();

            if (epoch % 100 == 0) {
                loss->data.toCPU();
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
                loss->data.toCUDA();
            }
        }

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
        cout << "Final Prediction (CUDA): ";
        for (int i = 0; i < num_classes; i++) {
            cout << final_data[i] << " ";
        }
        cout << endl;

        target.toCPU();
        cout << "Ground Truth (CUDA): ";
        for (int i = 0; i < num_classes; i++) {
            cout << target.data()[i] << " ";
        }
        cout << endl;
    }
}