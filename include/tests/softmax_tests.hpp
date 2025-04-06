#pragma once
#include "softmax.hpp"

using namespace std;

// Test function for Softmax using CUDA.
inline void runSoftmaxTests() {
    // Define dimensions: 1 sample, 3 classes.
    int batch_size = 1, num_classes = 3;
    size_t size = batch_size * num_classes;

    // Create a CPU tensor with known values.
    // For example, using values [1, 2, 3] for one sample.
    Tensor input(size, CPU);
    float *in_data = input.data();
    in_data[0] = 1.0f;
    in_data[1] = 2.0f;
    in_data[2] = 3.0f;

    // Transfer input to CUDA.
    input.toCUDA();

    // Wrap the input tensor in a Variable.
    VarPtr input_var = make_shared<Variable>(input, false);

    // Create a Softmax module with the given dimensions.
    Softmax softmax(batch_size, num_classes);

    // Run the forward pass.
    VarPtr output_var = softmax.forward(input_var);
    Tensor output = output_var->data;

    // Transfer the output to CPU for validation.
    if (output.device() != CPU) {
        output.toCPU();
    }

    // Expected output:
    // Compute using softmax formula:
    // For input [1, 2, 3]:
    // max = 3,
    // exp(1-3)=exp(-2) ~ 0.135335, exp(2-3)=exp(-1) ~ 0.367879, exp(3-3)=exp(0)=1,
    // sum ~ 1.503214, so probabilities ~ [0.09003057, 0.24472847, 0.66524096]
    vector<float> expected = {0.09003057f, 0.24472847f, 0.66524096f};

    cout << "Expected Output (Ground Truth): ";
    for (size_t i = 0; i < expected.size(); ++i) {
        cout << expected[i] << " ";
    }
    cout << endl;

    const float *out_data = output.data();
    cout << "Softmax output: ";
    for (size_t i = 0; i < size; i++) {
        cout << out_data[i] << " ";
    }
    cout << endl;
}