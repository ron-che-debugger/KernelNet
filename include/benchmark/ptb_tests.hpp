#pragma once

#include "kernelnet.hpp"

#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace std::chrono;

/**
 * Helper function: converts a tensor of token indices to a one-hot encoded tensor.
 */
inline Tensor onehot(const Tensor &indices, int num_classes) {
    size_t n = indices.size();
    Tensor onehot_tensor(n * num_classes, indices.device());
    const float *indices_data = indices.data();
    float *onehot_data = onehot_tensor.data();
    for (size_t i = 0; i < n; i++) {
        int idx = static_cast<int>(indices_data[i]);
        for (int j = 0; j < num_classes; j++) {
            onehot_data[i * num_classes + j] = (j == idx) ? 1.0f : 0.0f;
        }
    }
    return onehot_tensor;
}

int runPTBTests() {
    // --- Hyperparameters and Device Setup ---
    Device dev = CPU; // Change to CUDA if desired.
    int batch_size = 64;
    int num_epochs = 20;
    int sequence_length = 35;
    int embed_dim = 128;
    int hidden_dim = 256;

    // --- Load PTB Data ---
    PTBDataset trainDataset("data/ptb/ptb.train.txt", sequence_length);
    PTBDataset testDataset("data/ptb/ptb.test.txt", sequence_length);
    PTBDataLoader trainLoader(trainDataset, batch_size, true);
    PTBDataLoader testLoader(testDataset, batch_size, false);

    int vocab_size = trainDataset.vocab_size;
    cout << "Vocabulary Size: " << vocab_size << endl;

    // --- Build Model ---
    // Model architecture: Embedding → LSTM (unrolled) → Dense → Softmax.
    auto embedding = make_shared<Embedding>(vocab_size, embed_dim, dev);
    auto lstm = make_shared<LSTM>(batch_size, sequence_length, embed_dim, hidden_dim, dev);
    auto dense = make_shared<Dense>(hidden_dim, vocab_size, dev);
    auto softmax = make_shared<Softmax>(batch_size * sequence_length, vocab_size);
    cout << "pass 0" << endl;
    // Assemble the model using the Sequential container.
    auto model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
        embedding,
        lstm,
        dense,
        softmax});

    // --- Setup Optimizer ---
    vector<VarPtr> params = model->parameters();
    float learning_rate = 0.01f;
    SGD optimizer(params, learning_rate);
    cout << "pass 1" << endl;
    // --- Define Loss Function Lambda ---
    // Our loss function takes a prediction (VarPtr) and a target tensor and returns a VarPtr.
    LossFunction loss_fn = [vocab_size](const VarPtr &prediction, const Tensor &target) {
        Tensor onehot_target = onehot(target, vocab_size);
        return CrossEntropyLossFunction::apply(prediction, onehot_target, vocab_size);
    };
    cout << "pass 2" << endl;
    // --- Create Trainer ---
    // Trainer accepts model, optimizer, and the loss function.
    Trainer trainer(model, optimizer, loss_fn);
    cout << "pass 3" << endl;
    // --- Training Loop ---
    auto start = high_resolution_clock::now();
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        int batches = 0;
        while (trainLoader.hasNext()) {
            cout << "pass 4" << endl;
            auto batch = trainLoader.nextBatch();
            cout << "pass 5" << endl;
            if (dev == CUDA) {
                batch.first.toCUDA();
                batch.second.toCUDA();
            }
            cout << "pass 6" << endl;
            // Wrap input into a Variable (target is passed as Tensor).
            VarPtr input_var = make_shared<Variable>(batch.first, false, "input_batch");
            VarPtr target_var = make_shared<Variable>(batch.second, false, "target_batch");
            cout << "pass 6.5" << endl;
            // Trainer.trainEpoch() takes a vector of input Variables and a vector of target Tensors.
            vector<VarPtr> inputs = {input_var};
            vector<VarPtr> targets = {target_var};

            trainer.trainEpoch(inputs, targets);
            cout << "pass 7" << endl;
            // For logging, compute loss separately:
            VarPtr prediction = model->forward(input_var);
            cout << "pass 8" << endl;
            VarPtr loss = loss_fn(prediction, batch.second);
            cout << "pass 9" << endl;
            float batch_loss = loss->data.sum();
            epoch_loss += batch_loss;
            batches++;
        }
        trainLoader.reset();
        cout << "Epoch " << epoch << " Average Loss: " << (epoch_loss / batches) << endl;
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();
    cout << "LSTM training completed in " << (duration / 1000.0) << " seconds." << endl;

    // --- Evaluation: Compute Perplexity on Validation Set ---
    float total_loss = 0.0f;
    int total_tokens = 0;
    while (testLoader.hasNext()) {
        auto batch = testLoader.nextBatch();
        if (dev == CUDA) {
            batch.first.toCUDA();
            batch.second.toCUDA();
        }
        VarPtr input_var = make_shared<Variable>(batch.first, false, "valid_input");
        VarPtr prediction = model->forward(input_var);
        VarPtr loss = loss_fn(prediction, batch.second);
        total_loss += loss->data.sum();
        total_tokens += batch.second.size();
    }
    float avg_loss = total_loss / total_tokens;
    float perplexity = exp(avg_loss);
    cout << "Validation Perplexity: " << perplexity << endl;

    return 0;
}