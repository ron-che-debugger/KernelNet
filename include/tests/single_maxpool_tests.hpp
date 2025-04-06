#pragma once
#include "maxpool.hpp"
#include "tensor.hpp"

using namespace std;

// Test function for MaxPool2D using CUDA.
inline void runSingleMaxpoolTests() {
    // Define input dimensions: 1 batch, 1 channel, 4x4 spatial dimensions.
    int batch_size = 1, channels = 1, height = 4, width = 4;
    size_t size = batch_size * channels * height * width;

    // Create a CPU tensor with known values.
    Tensor input(size, CPU);
    float *input_data = input.data();
    // Fill with values 1 through 16 (row-major order):
    // [ 1,  2,  3,  4 ]
    // [ 5,  6,  7,  8 ]
    // [ 9, 10, 11, 12 ]
    // [13, 14, 15, 16 ]
    for (size_t i = 0; i < size; i++) {
        input_data[i] = static_cast<float>(i + 1);
    }

    // Transfer input tensor to CUDA device.
    input.toCUDA();

    // Create a MaxPool2D module with kernel_size=2, stride=2.
    MaxPool2D maxpool(2, 2, batch_size, channels, height, width);

    // Wrap the input tensor in a Variable.
    VarPtr input_var = make_shared<Variable>(input, false);

    // Run the forward pass.
    VarPtr output_var = maxpool.forward(input_var);
    Tensor output = output_var->data;

    // Make sure to transfer the output to CPU for validation.
    if (output.device() != CPU) {
        output.toCPU();
    }

    // Expected output: a 1x1x2x2 tensor with values:
    // [6, 8,
    // 14, 16]
    vector<float> expected = {6.0f, 8.0f, 14.0f, 16.0f};

    const float *out_data = output.data();
    int output_size = output.size();

    cout << "MaxPool Output: ";
    for (int i = 0; i < output_size; ++i) {
        cout << out_data[i] << " ";
    }
    cout << endl;

    // Validate the output against expected values.
    for (int i = 0; i < output_size; ++i) {
        assert(fabs(out_data[i] - expected[i]) < 1e-5);
    }

    cout << "MaxPool CUDA test passed!" << endl;
}
