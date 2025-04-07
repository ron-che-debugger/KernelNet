#pragma once
#include "sigmoid.hpp"
#include "tensor.hpp"

using namespace std;

inline void runSigmoidTests() {
    // Define size for our test (3 values).
    size_t size = 3;
    // Expected sigmoid outputs for inputs [0, 1, -1]:
    // sigmoid(0) = 0.5
    // sigmoid(1) ≈ 0.7310586
    // sigmoid(-1) ≈ 0.2689414
    vector<float> expected = {0.5f, 0.7310586f, 0.2689414f};

    // ------------------- CPU Test -------------------
    cout << "===== Running Sigmoid Test on CPU =====" << endl;
    // Create a CPU tensor with known values.
    Tensor input_cpu(size, CPU);
    float *cpu_data = input_cpu.data();
    cpu_data[0] = 0.0f;
    cpu_data[1] = 1.0f;
    cpu_data[2] = -1.0f;

    // Wrap input in a Variable (remains on CPU).
    VarPtr input_var_cpu = make_shared<Variable>(input_cpu, false, "sigmoid_input");

    // Create Sigmoid module.
    Sigmoid sigmoid_cpu;

    // Run forward pass.
    VarPtr output_var_cpu = sigmoid_cpu.forward(input_var_cpu);
    Tensor output_cpu = output_var_cpu->data; // Already on CPU.

    cout << "Expected Output (CPU): ";
    for (size_t i = 0; i < expected.size(); ++i) {
        cout << expected[i] << " ";
    }
    cout << endl;

    cout << "Sigmoid Output (CPU): ";
    const float *cpu_out = output_cpu.data();
    for (size_t i = 0; i < size; i++) {
        cout << cpu_out[i] << " ";
    }
    cout << endl
         << endl;

    // ------------------- CUDA Test -------------------
    cout << "===== Running Sigmoid Test on CUDA =====" << endl;
    // Create a CPU tensor with the same values then transfer to CUDA.
    Tensor input_cuda(size, CPU);
    float *cuda_data = input_cuda.data();
    cuda_data[0] = 0.0f;
    cuda_data[1] = 1.0f;
    cuda_data[2] = -1.0f;
    input_cuda.toCUDA();

    VarPtr input_var_cuda = make_shared<Variable>(input_cuda, false, "sigmoid_input");

    Sigmoid sigmoid_cuda;

    VarPtr output_var_cuda = sigmoid_cuda.forward(input_var_cuda);
    Tensor output_cuda = output_var_cuda->data;
    if (output_cuda.device() != CPU) {
        output_cuda.toCPU();
    }

    cout << "Expected Output (CUDA): ";
    for (size_t i = 0; i < expected.size(); ++i) {
        cout << expected[i] << " ";
    }
    cout << endl;

    cout << "Sigmoid Output (CUDA): ";
    const float *cuda_out = output_cuda.data();
    for (size_t i = 0; i < size; i++) {
        cout << cuda_out[i] << " ";
    }
    cout << endl;
}