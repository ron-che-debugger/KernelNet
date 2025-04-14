#pragma once

#include "kernelnet.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include <initializer_list>

using namespace std;

inline void runEmbeddingLSTMTests() {
    // Use a fixed random seed for reproducibility.
    srand(42);

    // -------------------- CPU Test --------------------
    {
        cout << "===== Running Embedding+LSTM Test on CPU =====" << endl;
        Device dev = CPU;
        const int batch_size = 8;
        const int sequence_length = 10;
        const int vocab_size = 50;   // small vocabulary for synthetic test
        const int embed_dim = 16;
        const int hidden_dim = 32;

        // Create synthetic input data on CPU.
        size_t num_tokens = batch_size * sequence_length;
        Tensor input_indices(num_tokens, dev);
        float* input_data = input_indices.data();
        for (size_t i = 0; i < num_tokens; i++) {
            // random index between 0 and (vocab_size-1)
            input_data[i] = static_cast<float>(rand() % vocab_size);
        }

        // Build the model: Embedding layer + LSTM layer.
        auto embedding = make_shared<Embedding>(vocab_size, embed_dim, dev);
        auto lstm = make_shared<LSTM>(batch_size, sequence_length, embed_dim, hidden_dim, dev);
        auto model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
            embedding,
            lstm
        });

        // Setup optimizer.
        vector<VarPtr> params = model->parameters();
        float learning_rate = 0.01f;
        SGD optimizer(params, learning_rate);

        // Dummy loss function that sums all elements of the prediction.
        LossFunction loss_fn = [](const VarPtr &prediction, const Tensor &target) -> VarPtr {
            return SumFunction::apply(prediction);
        };

        // Create the trainer.
        Trainer trainer(model, optimizer, loss_fn);

        // Training loop.
        int num_epochs = 5;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            // Wrap the input tensor in a Variable (indices do not require gradients).
            VarPtr input_var = make_shared<Variable>(input_indices, false, "input_indices");

            // Create a dummy target tensor (its values are ignored by loss_fn).
            Tensor dummy_target = input_indices;
            VarPtr target_var = make_shared<Variable>(dummy_target, false, "target");

            vector<VarPtr> inputs = {input_var};
            vector<VarPtr> targets = {target_var};

            // Perform a training epoch.
            trainer.trainEpoch(inputs, targets);

            // Compute prediction and loss for logging.
            VarPtr prediction = model->forward(input_var);
            VarPtr loss = loss_fn(prediction, dummy_target);
            cout << "Epoch " << epoch << " loss (CPU): " << loss->data.sum() << endl;
        }
    }

    // -------------------- CUDA Test --------------------
    {
        cout << "\n===== Running Embedding+LSTM Test on CUDA =====" << endl;
        Device dev = CUDA;
        const int batch_size = 8;
        const int sequence_length = 10;
        const int vocab_size = 50;   // small vocabulary for synthetic test
        const int embed_dim = 16;
        const int hidden_dim = 32;

        // Create synthetic input data on CPU, then convert to CUDA.
        size_t num_tokens = batch_size * sequence_length;
        Tensor input_indices(num_tokens, CPU);
        float* input_data = input_indices.data();
        for (size_t i = 0; i < num_tokens; i++) {
            // random index between 0 and (vocab_size-1)
            input_data[i] = static_cast<float>(rand() % vocab_size);
        }
        input_indices.toCUDA();  // careful with the conversion: transfer tensor to CUDA memory

        // Build the model on CUDA.
        auto embedding = make_shared<Embedding>(vocab_size, embed_dim, dev);
        auto lstm = make_shared<LSTM>(batch_size, sequence_length, embed_dim, hidden_dim, dev);
        auto model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
            embedding,
            lstm
        });

        // Setup optimizer.
        vector<VarPtr> params = model->parameters();
        float learning_rate = 0.01f;
        SGD optimizer(params, learning_rate);

        // Dummy loss function.
        LossFunction loss_fn = [](const VarPtr &prediction, const Tensor &target) -> VarPtr {
            return SumFunction::apply(prediction);
        };

        // Create the trainer.
        Trainer trainer(model, optimizer, loss_fn);

        // Training loop.
        int num_epochs = 5;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            // Wrap the CUDA tensor in a Variable.
            VarPtr input_var = make_shared<Variable>(input_indices, false, "input_indices");

            // Create a dummy target tensor. (It is on CUDA because input_indices is now in CUDA memory.)
            Tensor dummy_target = input_indices;
            VarPtr target_var = make_shared<Variable>(dummy_target, false, "target");

            vector<VarPtr> inputs = {input_var};
            vector<VarPtr> targets = {target_var};

            trainer.trainEpoch(inputs, targets);

            // Forward pass: move the prediction back to CPU for logging.
            VarPtr prediction = model->forward(input_var);
            prediction->data.toCPU();
            VarPtr loss = loss_fn(prediction, dummy_target);
            cout << "Epoch " << epoch << " loss (CUDA): " << loss->data.sum() << endl;
        }
    }
}