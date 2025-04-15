#pragma once

#include "api_header.hpp"

using namespace std;

inline void runSingleDenseLayerTests() {
    srand(42);

    // -------------------- CPU Test --------------------
    {
        cout << "===== Running Single Dense Layer Test on CPU =====" << endl;
        Device device = CPU;
        const int batch_size = 4;
        const int input_dim = 2;
        const int output_dim = 1;

        // Create input tensor X on CPU.
        Tensor X_tensor(batch_size * input_dim, CPU);
        float X_data[] = {1.0f, 2.0f,
                          2.0f, 1.0f,
                          3.0f, 0.0f,
                          0.0f, 4.0f};
        memcpy(X_tensor.data(), X_data, sizeof(X_data));
        // No transfer needed for CPU.
        auto X = make_shared<Variable>(X_tensor, false, "input");

        // Create target tensor Y on CPU.
        Tensor Y_tensor(batch_size * output_dim, CPU);
        float Y_data[] = {3 * 1.0f + 2 * 2.0f,  // 3+4 = 7
                          3 * 2.0f + 2 * 1.0f,  // 6+2 = 8
                          3 * 3.0f + 2 * 0.0f,  // 9+0 = 9
                          3 * 0.0f + 2 * 4.0f}; // 0+8 = 8
        memcpy(Y_tensor.data(), Y_data, sizeof(Y_data));

        // Build a Dense layer on CPU.
        Dense dense(input_dim, output_dim, device);
        vector<VarPtr> params = dense.parameters();
        SGD optimizer(params, 0.001f);

        const int epochs = 5000;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto pred = dense.forward(X);
            auto loss = MSEFunction::apply(pred, Y_tensor);
            Tensor grad_one(loss->data.size(), loss->data.device());
            grad_one.fill(1.0f);
            loss->backward(grad_one);
            optimizer.step();
            optimizer.zero_grad();

            if (epoch % 500 == 0) {
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }

        cout << "Ground Truth (CPU):" << endl;
        Y_tensor.print();

        auto final_pred = dense.forward(X);
        final_pred->data.toCPU();
        cout << "Final Prediction (CPU):" << endl;
        final_pred->data.print();
    }

    // -------------------- CUDA Test --------------------
    {
        cout << "\n===== Running Single Dense Layer Test on CUDA =====" << endl;
        Device device = CUDA;
        const int batch_size = 4;
        const int input_dim = 2;
        const int output_dim = 1;

        // Create input tensor X on CPU, then transfer to CUDA.
        Tensor X_tensor(batch_size * input_dim, CPU);
        float X_data[] = {1.0f, 2.0f,
                          2.0f, 1.0f,
                          3.0f, 0.0f,
                          0.0f, 4.0f};
        memcpy(X_tensor.data(), X_data, sizeof(X_data));
        X_tensor.toCUDA();
        auto X = make_shared<Variable>(X_tensor, false, "input");

        // Create target tensor Y on CPU, then transfer to CUDA.
        Tensor Y_tensor(batch_size * output_dim, CPU);
        float Y_data[] = {3 * 1.0f + 2 * 2.0f,  // 3+4 = 7
                          3 * 2.0f + 2 * 1.0f,  // 6+2 = 8
                          3 * 3.0f + 2 * 0.0f,  // 9+0 = 9
                          3 * 0.0f + 2 * 4.0f}; // 0+8 = 8
        memcpy(Y_tensor.data(), Y_data, sizeof(Y_data));
        Y_tensor.toCUDA();

        // Build a Dense layer on CUDA.
        Dense dense(input_dim, output_dim, device);
        vector<VarPtr> params = dense.parameters();
        SGD optimizer(params, 0.001f);

        const int epochs = 5000;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto pred = dense.forward(X);
            auto loss = MSEFunction::apply(pred, Y_tensor);
            Tensor grad_one(loss->data.size(), loss->data.device());
            grad_one.fill(1.0f);
            loss->backward(grad_one);
            optimizer.step();
            optimizer.zero_grad();

            if (epoch % 500 == 0) {
                loss->data.toCPU();
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
                loss->data.toCUDA();
            }
        }

        cout << "Ground Truth (CUDA):" << endl;
        Y_tensor.toCPU();
        Y_tensor.print();
        Y_tensor.toCUDA();

        auto final_pred = dense.forward(X);
        final_pred->data.toCPU();
        cout << "Final Prediction (CUDA):" << endl;
        final_pred->data.print();
    }
}