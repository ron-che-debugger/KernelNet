#pragma once

#include "api_header.hpp"

using namespace std;

inline void runSingleEmbeddingTests() {
    srand(42);

    // -------------------- CPU Test --------------------
    {
        cout << "===== Running Embedding Test on CPU =====" << endl;
        Device device = CPU;
        const int vocab_size = 5;
        const int embed_dim = 3;
        const int batch_size = 4; // Number of token indices

        // Create an input tensor (flattened indices) on CPU.
        // We'll use indices: {1, 3, 0, 4}
        Tensor input_tensor(batch_size, CPU);
        float indices_data[] = {1.0f, 3.0f, 0.0f, 4.0f};
        memcpy(input_tensor.data(), indices_data, sizeof(indices_data));
        auto input = make_shared<Variable>(input_tensor, false, "embedding_input");

        // Build an Embedding layer on CPU.
        Embedding embedding(vocab_size, embed_dim, device);
        // Overwrite weight with a known pattern for testing.
        // For instance, set each row i in the weight matrix to [i, i, i]
        Tensor weight_tensor(vocab_size * embed_dim, CPU);
        float *w_data = weight_tensor.data();
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < embed_dim; j++) {
                w_data[i * embed_dim + j] = float(i);
            }
        }
        // Replace the default weight with our controlled weight.
        embedding.weight->data = weight_tensor;

        // Forward pass.
        auto output = embedding.forward(input);
        output->data.toCPU();
        cout << "Embedding output (CPU):" << endl;
        output->data.print(); // Expected : 1 1 1 3 3 3 0 0 0 4 4 4

        // Backward pass.
        // Create a grad_output tensor of ones (same shape as output).
        Tensor grad_output(output->data.size(), CPU);
        grad_output.fill(1.0f);
        output->backward(grad_output);

        // Print the weight gradient.
        cout << "Weight gradient (CPU):" << endl;
        embedding.weight->grad.toCPU();
        embedding.weight->grad.print(); // Expected : 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1
    }

    // -------------------- CUDA Test --------------------
    {
        cout << "\n===== Running Embedding Test on CUDA =====" << endl;
        Device device = CUDA;
        const int vocab_size = 5;
        const int embed_dim = 3;
        const int batch_size = 4; // Number of token indices

        // Create an input tensor (flattened indices) on CPU and transfer to CUDA.
        Tensor input_tensor(batch_size, CPU);
        float indices_data[] = {1.0f, 3.0f, 0.0f, 4.0f};
        memcpy(input_tensor.data(), indices_data, sizeof(indices_data));
        input_tensor.toCUDA();
        auto input = make_shared<Variable>(input_tensor, false, "embedding_input");

        // Build an Embedding layer on CUDA.
        Embedding embedding(vocab_size, embed_dim, device);
        // Overwrite the embedding weight with a known pattern.
        Tensor weight_tensor(vocab_size * embed_dim, CPU);
        float *w_data = weight_tensor.data();
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < embed_dim; j++) {
                w_data[i * embed_dim + j] = float(i);
            }
        }
        weight_tensor.toCUDA();
        embedding.weight->data = weight_tensor;

        // Forward pass.
        auto output = embedding.forward(input);
        output->data.toCPU();
        cout << "Embedding output (CUDA):" << endl;
        output->data.print(); // Expected : 1 1 1 3 3 3 0 0 0 4 4 4

        // Backward pass.
        // Create a grad_output tensor of ones, then move to CUDA.
        Tensor grad_output(output->data.size(), CPU);
        grad_output.fill(1.0f);
        grad_output.toCUDA();
        output->backward(grad_output);

        // Print the weight gradient.
        embedding.weight->grad.toCPU();
        cout << "Weight gradient (CUDA):" << endl;
        embedding.weight->grad.print(); // Expected : 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1
    }
}