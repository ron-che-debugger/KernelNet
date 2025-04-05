#ifndef SINGLE_DENSE_LAYER_TESTS_HPP
#define SINGLE_DENSE_LAYER_TESTS_HPP

#include "autograd.hpp"
#include "dense.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

// A simple test to train a single dense layer on a synthetic regression task.
inline void runSingleDenseLayerTests() {
    srand(42);
    cout << "=== Training a Simple Dense Network on GPU (CUDA) ===" << endl;

    Device device = CUDA;
    const int batch_size = 4;
    const int input_dim = 2;
    const int output_dim = 1;

    // Create input tensor X on CPU then transfer to CUDA.
    Tensor X_tensor(batch_size * input_dim, CPU);
    float X_data[] = {1.0f, 2.0f,
                      2.0f, 1.0f,
                      3.0f, 0.0f,
                      0.0f, 4.0f};
    memcpy(X_tensor.data(), X_data, sizeof(X_data));
    X_tensor.toCUDA();
    auto X = make_shared<Variable>(X_tensor, false);

    // Create target tensor Y on CPU then transfer to CUDA.
    Tensor Y_tensor(batch_size * output_dim, CPU);
    float Y_data[] = {3 * 1.0f + 2 * 2.0f,
                      3 * 2.0f + 2 * 1.0f,
                      3 * 3.0f + 2 * 0.0f,
                      3 * 0.0f + 2 * 4.0f};
    memcpy(Y_tensor.data(), Y_data, sizeof(Y_data));
    Y_tensor.toCUDA();

    // Build a Dense layer.
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

    auto final_pred = dense.forward(X);
    final_pred->data.toCPU();
    cout << "Final predictions:" << endl;
    final_pred->data.print();
}

#endif