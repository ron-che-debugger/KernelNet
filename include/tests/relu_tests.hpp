#pragma once
#include "kernelnet.hpp"

using namespace std;

inline void runReluTests() {
    // ------------------- CPU Test -------------------
    cout << "===== Running ReLU Test on CPU =====" << endl;
    // Create a CPU tensor with a mix of negative, zero, and positive values.
    // For example, a tensor of size 5.
    size_t size = 5;
    Tensor input_cpu(size, CPU);
    float *cpu_data = input_cpu.data();
    cpu_data[0] = -1.0f;
    cpu_data[1] = 0.0f;
    cpu_data[2] = 1.0f;
    cpu_data[3] = -2.0f;
    cpu_data[4] = 2.0f;

    // Wrap the tensor in a Variable (on CPU).
    VarPtr input_var_cpu = make_shared<Variable>(input_cpu, false, "relu_input_cpu");

    // Create a ReLU module instance.
    ReLU relu_cpu;

    // Perform forward pass.
    VarPtr output_var_cpu = relu_cpu.forward(input_var_cpu);
    Tensor output_cpu = output_var_cpu->data; // Already on CPU.

    cout << "Expected Output (CPU): 0 0 1 0 2" << endl;
    cout << "ReLU Output (CPU): ";
    const float *cpu_out = output_cpu.data();
    for (size_t i = 0; i < size; i++) {
        cout << cpu_out[i] << " ";
    }
    cout << "\n\n";

    // ------------------- CUDA Test -------------------
    cout << "===== Running ReLU Test on CUDA =====" << endl;
    // Create a CPU tensor and initialize the same values, then transfer it to CUDA.
    Tensor input_cuda(size, CPU);
    float *cuda_data = input_cuda.data();
    cuda_data[0] = -1.0f;
    cuda_data[1] = 0.0f;
    cuda_data[2] = 1.0f;
    cuda_data[3] = -2.0f;
    cuda_data[4] = 2.0f;
    input_cuda.toCUDA();

    VarPtr input_var_cuda = make_shared<Variable>(input_cuda, false, "relu_input_cuda");
    ReLU relu_cuda;
    VarPtr output_var_cuda = relu_cuda.forward(input_var_cuda);
    Tensor output_cuda = output_var_cuda->data;
    if (output_cuda.device() != CPU) {
        output_cuda.toCPU();
    }

    cout << "Expected Output (CUDA): 0 0 1 0 2" << endl;
    cout << "ReLU Output (CUDA): ";
    const float *cuda_out = output_cuda.data();
    for (size_t i = 0; i < size; i++) {
        cout << cuda_out[i] << " ";
    }
    cout << endl;
}
