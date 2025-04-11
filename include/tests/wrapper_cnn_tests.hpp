#pragma once

#include "autograd.hpp"
#include "conv2d.hpp"     // Convolution layer (inherits from SingleInputModule)
#include "dense.hpp"      // Dense layer (inherits from SingleInputModule)
#include "maxpool.hpp"    // Max Pool layer (inherits from SingleInputModule)
#include "optimizer.hpp"  // SGD optimizer
#include "sequential.hpp" // Our new sequential container
#include "softmax.hpp"    // Softmax layer (inherits from SingleInputModule)
#include "tensor.hpp"
#include "trainer.hpp" // Our trainer module
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

inline void runWrapperCnnTests() {
    // ------------------- Test on CPU -------------------
    cout << "===== Running Wrapper CNN Test on CPU =====" << endl;
    {
        Device dev = CPU;
        int batch_size = 1, in_channels = 1, height = 8, width = 8, num_classes = 2;
        size_t input_size = batch_size * in_channels * height * width;
        Tensor input(input_size, CPU);
        float *input_data = input.data();
        for (size_t i = 0; i < input_size; i++) {
            input_data[i] = 1.0f;
        }
        // Wrap input tensor in a Variable.
        VarPtr input_var = make_shared<Variable>(input, false, "input");

        // Create target tensor: one-hot vector [0, 1].
        Tensor target(num_classes, CPU);
        target.data()[0] = 0.0f;
        target.data()[1] = 1.0f;
        VarPtr target_var = make_shared<Variable>(target, false, "target");

        // Build network layers.
        // For Conv2D, we use padding so that the spatial dims remain unchanged.
        auto conv1 = make_shared<Conv2D>(1, 1, 3, 3, height, width, 1, 1, dev);
        auto pool1 = make_shared<MaxPool2D>(2, 2, batch_size, 1, height, width);
        // After pool1, image dims become (4,4).
        auto conv2 = make_shared<Conv2D>(1, 1, 3, 3, 4, 4, 1, 1, dev);
        auto pool2 = make_shared<MaxPool2D>(2, 2, batch_size, 1, 4, 4);
        auto dense = make_shared<Dense>(4, 2, dev);
        auto softmax = make_shared<Softmax>(batch_size, num_classes);

        // Build a Sequential container by listing the layers.
        // Note that we store the model as a Sequential pointer so that we can call
        // the convenience single-argument forward.
        shared_ptr<Sequential> model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
            conv1, pool1, conv2, pool2, dense, softmax});

        // Create SGD optimizer using the parameters collected from the Sequential container.
        vector<VarPtr> params = model->parameters();
        float learning_rate = 0.01f;
        SGD optimizer(params, learning_rate);

        // Create a Trainer.
        Trainer trainer(model, optimizer);

        // In this simple test, we train on a single sample.
        vector<VarPtr> inputs = {input_var};
        vector<VarPtr> targets = {target_var};

        // Define number of epochs.
        int num_epochs = 2000;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            trainer.trainEpoch(inputs, targets);

            if (epoch % 100 == 0) {
                VarPtr prediction = model->forward(input_var); // calls the convenience overload
                // Calculate loss for logging.
                VarPtr loss = MSEFunction::apply(prediction, target);
                loss->data.toCPU();
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }

        // Final forward pass.
        VarPtr final_pred = model->forward(input_var);
        final_pred->data.toCPU();
        cout << "Final Prediction (CPU): ";
        for (int i = 0; i < num_classes; i++) {
            cout << final_pred->data.data()[i] << " ";
        }
        cout << endl;

        cout << "Ground Truth (CPU): ";
        for (int i = 0; i < target.size(); i++) {
            cout << target.data()[i] << " ";
        }
        cout << endl;
    }

    // ------------------- Test on CUDA -------------------
    cout << "\n===== Running Wrapper CNN Test on CUDA =====" << endl;
    {
        Device dev = CUDA;
        int batch_size = 1, in_channels = 1, height = 8, width = 8, num_classes = 2;
        size_t input_size = batch_size * in_channels * height * width;
        Tensor input(input_size, CPU);
        float *input_data = input.data();
        for (size_t i = 0; i < input_size; i++) {
            input_data[i] = 1.0f;
        }
        // Move input to CUDA.
        input.toCUDA();
        VarPtr input_var = make_shared<Variable>(input, false, "input");

        // Create target: one-hot vector [0, 1].
        Tensor target(num_classes, CPU);
        target.data()[0] = 0.0f;
        target.data()[1] = 1.0f;
        target.toCUDA();
        VarPtr target_var = make_shared<Variable>(target, false, "target");

        // Build network layers on CUDA.
        auto conv1 = make_shared<Conv2D>(1, 1, 3, 3, height, width, 1, 1, dev);
        auto pool1 = make_shared<MaxPool2D>(2, 2, batch_size, 1, height, width);
        auto conv2 = make_shared<Conv2D>(1, 1, 3, 3, 4, 4, 1, 1, dev);
        auto pool2 = make_shared<MaxPool2D>(2, 2, batch_size, 1, 4, 4);
        auto dense = make_shared<Dense>(4, 2, dev);
        auto softmax = make_shared<Softmax>(batch_size, num_classes);

        shared_ptr<Sequential> model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
            conv1, pool1, conv2, pool2, dense, softmax});

        // Collect parameters from the model.
        vector<VarPtr> params = model->parameters();
        float learning_rate = 0.01f;
        SGD optimizer(params, learning_rate);

        // Create Trainer.
        Trainer trainer(model, optimizer);
        vector<VarPtr> inputs = {input_var};
        vector<VarPtr> targets = {target_var};

        int num_epochs = 2000;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            trainer.trainEpoch(inputs, targets);

            if (epoch % 100 == 0) {
                VarPtr prediction = model->forward(input_var);
                VarPtr loss = MSEFunction::apply(prediction, target);
                loss->data.toCPU();
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }

        VarPtr final_pred = model->forward(input_var);
        final_pred->data.toCPU();
        cout << "Final Prediction (CUDA): ";
        for (int i = 0; i < num_classes; i++) {
            cout << final_pred->data.data()[i] << " ";
        }
        cout << endl;

        target.toCPU();
        cout << "Ground Truth (CUDA): ";
        for (int i = 0; i < target.size(); i++) {
            cout << target.data()[i] << " ";
        }
        cout << endl;
    }
}