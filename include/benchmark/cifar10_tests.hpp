#pragma once

#include "./benchmark/cifar10_data_loader.hpp" // New module for batching, shuffling, etc.
#include "./benchmark/cifar10_dataset.hpp"     // New module to load CIFAR-10 from disk.
#include "autograd.hpp"                        // Our autograd system.
#include "conv2d.hpp"                          // Convolution layer (SingleInputModule).
#include "dense.hpp"                           // Dense layer (SingleInputModule).
#include "maxpool.hpp"                         // MaxPool layer (SingleInputModule).
#include "optimizer.hpp"                       // SGD optimizer.
#include "sequential.hpp"                      // Our Sequential container.
#include "softmax.hpp"                         // Softmax layer (SingleInputModule).
#include "tensor.hpp"                          // Our Tensor abstraction.
#include "trainer.hpp"                         // Our Trainer module.
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace std::chrono;

void runCIFAR10Tests() {
    // Set device (for this example, we run on CPU; later you can set dev = CUDA)
    Device dev = CPU;
    int batch_size = 32;
    int num_epochs = 10;  // For demonstration purposes.
    int num_classes = 10; // CIFAR-10 has 10 classes.
    int image_height = 32, image_width = 32, in_channels = 3;

    // --- Data Loading ---
    // Assume we have implemented CIFAR10Dataset and DataLoader.
    // CIFAR10Dataset loads images (as Tensor) and labels (as one-hot or indices) from disk.
    CIFAR10Dataset trainDataset("data/cifar10/train", /*train=*/true);
    CIFAR10Dataset testDataset("data/cifar10/test", /*train=*/false);
    CIFAR10DataLoader trainLoader(trainDataset, batch_size, /*shuffle=*/true);
    CIFAR10DataLoader testLoader(testDataset, batch_size, /*shuffle=*/false);

    // --- Build CNN Model ---
    // Our simple CNN:
    // conv1: from 3 channels to 16, kernel 3x3, stride=1, pad=1 (keeps spatial dims 32x32).
    auto conv1 = make_shared<Conv2D>(in_channels, 16, 3, 3, image_height, image_width, 1, 1, dev);
    // After pool1: dims become 16 x 16.
    auto pool1 = make_shared<MaxPool2D>(2, 2, batch_size, 16, image_height, image_width);
    // conv2: 16 -> 32 channels, dims stay 16x16.
    auto conv2 = make_shared<Conv2D>(16, 32, 3, 3, image_height / 2, image_width / 2, 1, 1, dev);
    // After pool2: dims become 8 x 8.
    auto pool2 = make_shared<MaxPool2D>(2, 2, batch_size, 32, image_height / 2, image_width / 2);
    // Dense: we flatten the output (32 * 8 * 8 = 2048) and map to 10 classes.
    auto dense = make_shared<Dense>(32 * 8 * 8, num_classes, dev);
    // Softmax layer for classification.
    auto softmax = make_shared<Softmax>(batch_size, num_classes);

    // Build the Sequential model.
    shared_ptr<Sequential> model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
        conv1, pool1, conv2, pool2, dense, softmax});

    // --- Set Optimizer ---
    vector<VarPtr> params = model->parameters();
    float learning_rate = 0.0001f;
    SGD optimizer(params, learning_rate);

    // --- Create Trainer ---
    LossFunction loss_fn = [num_classes](const VarPtr &prediction, const Tensor &target) {
        // 10 is the number of classes we want to use for averaging.
        return CrossEntropyLossFunction::apply(prediction, target, num_classes);
    };
    
    Trainer trainer(model, optimizer, loss_fn);

    // --- Training Loop ---
    auto start = high_resolution_clock::now();
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int batches = 0;
        while (trainLoader.hasNext()) {
            // Each batch returns a pair: {input_batch, target_batch}
            auto batch = trainLoader.nextBatch();
            // batch.first: Tensor input, shape: [batch_size, channels, height, width]
            // batch.second: Tensor target, format as required (e.g., class indices or one-hot)
            VarPtr input_var = make_shared<Variable>(batch.first, false, "input_batch");
            VarPtr target_var = make_shared<Variable>(batch.second, false, "target_batch");
            vector<VarPtr> inputs = {input_var};
            vector<VarPtr> targets = {target_var};
            trainer.trainEpoch(inputs, targets);
            VarPtr prediction = model->forward(input_var);
            VarPtr loss = loss_fn(prediction, batch.second);
            float batch_avg_loss = loss->data.sum() / batch_size;
            epoch_loss += batch_avg_loss;
            batches++;
            // cout << "Finished computing a batch." << endl;
        }
        trainLoader.reset(); // Reset loader for next epoch.
        cout << "Epoch " << epoch << " Average Loss: " << epoch_loss / batches << endl;

        if (epoch == 0 || epoch % 5 == 0) { // Adjust frequency as needed.
            // Get one batch without training.
            auto sample_batch = trainLoader.nextBatch();
            VarPtr sample_input = make_shared<Variable>(sample_batch.first, false, "sample_input");
            VarPtr sample_pred = model->forward(sample_input);
            cout << "Prediction tensor at epoch " << epoch << ":" << endl;
            sample_pred->data.toCPU();
            sample_pred->data.print();
            // Optionally, reset the loader so that training continues correctly.
            trainLoader.reset();
        }
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "Custom architecture training completed in " << duration << " ms" << endl;

    // --- Evaluate on Test Set (Accuracy) ---
    int correct = 0, total = 0;
    while (testLoader.hasNext()) {
        auto batch = testLoader.nextBatch();
        VarPtr input_var = make_shared<Variable>(batch.first, false, "test_input");
        VarPtr prediction = model->forward(input_var);

        int pred_label = prediction->data.argmax();
        int true_label = batch.second.argmax();
        if (pred_label == true_label)
            correct++;
        total++;
    }
    cout << "Custom architecture Test Accuracy: " << (100.0 * correct / total) << "%" << endl;
}