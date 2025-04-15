/**
 * @file cifar10_test.cpp
 * @brief Trains and evaluates a simple CNN on the CIFAR-10 dataset using a custom autograd engine.
 *
 * This test pipeline includes:
 * - Loading CIFAR-10 training and test data from binary files
 * - Batching and shuffling samples via a DataLoader
 * - Building a convolutional neural network with Conv2D, MaxPool, Dense, and Softmax layers
 * - Using SGD optimizer and cross-entropy loss for training
 * - Running multiple epochs and reporting average loss per epoch
 * - Evaluating final model accuracy on the test set
 */

#pragma once

#include "kernelnet.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace std::chrono;

/**
 * @brief Entry point for running CIFAR-10 training and evaluation.
 *
 * Constructs a basic convolutional network and trains it using the custom autograd framework.
 * After training, accuracy is evaluated on the held-out test set.
 */
void runCIFAR10Tests() {
    Device dev = CPU;

    int batch_size = 256;
    int num_epochs = 100;
    int num_classes = 10;
    int image_height = 32, image_width = 32, in_channels = 3;

    // --- Load CIFAR-10 Data ---
    CIFAR10Dataset trainDataset("data/cifar10/train", true);
    CIFAR10Dataset testDataset("data/cifar10/test", false);
    CIFAR10DataLoader trainLoader(trainDataset, batch_size, true);
    CIFAR10DataLoader testLoader(testDataset, batch_size, false);

    // --- Define Model Architecture ---
    // conv1: input channels 3 → 16, output dims remain 32×32
    auto conv1 = make_shared<Conv2D>(in_channels, 16, 3, 3, image_height, image_width, 1, 1, dev);
    auto pool1 = make_shared<MaxPool2D>(2, 2, batch_size, 16, image_height, image_width); // → 16×16

    auto conv2 = make_shared<Conv2D>(16, 32, 3, 3, image_height / 2, image_width / 2, 1, 1, dev);
    auto pool2 = make_shared<MaxPool2D>(2, 2, batch_size, 32, image_height / 2, image_width / 2); // → 8×8

    auto dense = make_shared<Dense>(32 * 8 * 8, num_classes, dev);
    auto softmax = make_shared<Softmax>(batch_size, num_classes);

    // Assemble model into a Sequential container
    shared_ptr<Sequential> model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
        conv1, pool1, conv2, pool2, dense, softmax});

    // --- Set Optimizer ---
    vector<VarPtr> params = model->parameters();
    float learning_rate = 0.01f;
    SGD optimizer(params, learning_rate);

    // --- Define Loss Function ---
    LossFunction loss_fn = [num_classes](const VarPtr &prediction, const Tensor &target) {
        return CrossEntropyLossFunction::apply(prediction, target, num_classes);
    };

    // --- Create Trainer ---
    Trainer trainer(model, optimizer, loss_fn);

    // --- Training Loop ---
    auto start = high_resolution_clock::now();
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int batches = 0;

        while (trainLoader.hasNext()) {
            auto batch = trainLoader.nextBatch();
            if (dev == CUDA) {
                batch.first.toCUDA();
                batch.second.toCUDA();
            }

            VarPtr input_var = make_shared<Variable>(batch.first, false, "input_batch");
            VarPtr target_var = make_shared<Variable>(batch.second, false, "target_batch");

            vector<VarPtr> inputs = {input_var};
            vector<VarPtr> targets = {target_var};
            trainer.trainEpoch(inputs, targets);

            VarPtr prediction = model->forward(input_var);
            VarPtr loss = loss_fn(prediction, batch.second);
            float batch_loss = loss->data.sum();

            epoch_loss += batch_loss;
            batches++;
        }

        trainLoader.reset();
        cout << "Epoch " << epoch << " Average Loss: " << epoch_loss / batches << endl;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "Custom architecture training completed in " << (duration / 1000.0) << " seconds" << endl;

    // --- Evaluate on Test Set ---
    int correct = 0, total = 0;
    while (testLoader.hasNext()) {
        auto batch = testLoader.nextBatch();

        if (dev == CUDA) {
            batch.first.toCUDA();
            batch.second.toCUDA();
        }

        VarPtr input_var = make_shared<Variable>(batch.first, false, "test_input");
        VarPtr prediction = model->forward(input_var);

        vector<int> pred_labels = prediction->data.argmax(1, num_classes);
        vector<int> true_labels = batch.second.argmax(1, num_classes);

        for (size_t i = 0; i < pred_labels.size(); ++i) {
            if (pred_labels[i] == true_labels[i])
                correct++;
            total++;
        }
    }

    cout << "KenelNet Test Accuracy: " << (100.0 * correct / total) << "%" << endl;
}