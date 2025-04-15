#pragma once
#include "api_header.hpp"

using namespace std;

inline void runSoftmaxTests() {
    // Define dimensions: 1 sample, 3 classes.
    int batch_size = 1, num_classes = 3;
    size_t size = batch_size * num_classes;

    vector<float> expected = {0.09003057f, 0.24472847f, 0.66524096f};

    // ------------------- CPU Test -------------------
    cout << "===== Running Softmax Test on CPU =====" << endl;

    // Create a CPU tensor with known values.
    Tensor input_cpu(size, CPU);
    float *cpu_data = input_cpu.data();
    cpu_data[0] = 1.0f;
    cpu_data[1] = 2.0f;
    cpu_data[2] = 3.0f;

    // Wrap input in a Variable (remains on CPU).
    VarPtr input_var_cpu = make_shared<Variable>(input_cpu, false, "softmax_input");

    // Create Softmax module.
    Softmax softmax_cpu(batch_size, num_classes);

    // Run forward pass.
    VarPtr output_var_cpu = softmax_cpu.forward(input_var_cpu);
    Tensor output_cpu = output_var_cpu->data; // Already on CPU.

    cout << "Expected Output (Ground Truth, CPU): ";
    for (size_t i = 0; i < expected.size(); ++i) {
        cout << expected[i] << " ";
    }
    cout << endl;

    cout << "Softmax Output (CPU): ";
    const float *cpu_out = output_cpu.data();
    for (size_t i = 0; i < size; i++) {
        cout << cpu_out[i] << " ";
    }
    cout << endl
         << endl;

    // ------------------- CUDA Test -------------------
    cout << "===== Running Softmax Test on CUDA =====" << endl;

    // Create a CPU tensor with the same values then transfer to CUDA.
    Tensor input_cuda(size, CPU);
    float *cuda_data = input_cuda.data();
    cuda_data[0] = 1.0f;
    cuda_data[1] = 2.0f;
    cuda_data[2] = 3.0f;
    input_cuda.toCUDA();

    VarPtr input_var_cuda = make_shared<Variable>(input_cuda, false, "softmax_input");

    Softmax softmax_cuda(batch_size, num_classes);

    VarPtr output_var_cuda = softmax_cuda.forward(input_var_cuda);
    Tensor output_cuda = output_var_cuda->data;
    if (output_cuda.device() != CPU) {
        output_cuda.toCPU();
    }

    cout << "Expected Output (Ground Truth, CUDA): ";
    for (size_t i = 0; i < expected.size(); ++i) {
        cout << expected[i] << " ";
    }
    cout << endl;

    cout << "Softmax Output (CUDA): ";
    const float *cuda_out = output_cuda.data();
    for (size_t i = 0; i < size; i++) {
        cout << cuda_out[i] << " ";
    }
    cout << endl;
}