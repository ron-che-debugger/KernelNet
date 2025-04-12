#pragma once
#include "kernelnet.hpp"

using namespace std;

inline void runSingleMaxpoolTests() {
    // Define input dimensions: 1 batch, 1 channel, 4x4 spatial dimensions.
    int batch_size = 1, channels = 1, height = 4, width = 4;
    size_t size = batch_size * channels * height * width;

    // Expected output (ground truth) for a 2x2 pooling (kernel=2, stride=2)
    // The 4x4 input (row-major) is:
    // [ 1,  2,  3,  4 ]
    // [ 5,  6,  7,  8 ]
    // [ 9, 10, 11, 12 ]
    // [13, 14, 15, 16 ]
    // Pooling 2x2 windows (top-left, top-right, bottom-left, bottom-right) yields:
    // [max(1,2,5,6),  max(3,4,7,8),
    //  max(9,10,13,14), max(11,12,15,16)]
    // That is: [6, 8, 14, 16].
    vector<float> expected = {6.0f, 8.0f, 14.0f, 16.0f};

    // ---------- CPU Test ----------
    cout << "===== Running MaxPool2D Test on CPU =====" << endl;

    // Create a CPU tensor with known values.
    Tensor input_cpu(size, CPU);
    float *cpu_data = input_cpu.data();
    for (size_t i = 0; i < size; i++) {
        cpu_data[i] = static_cast<float>(i + 1);
    }

    // Wrap the CPU tensor in a Variable.
    VarPtr input_var_cpu = make_shared<Variable>(input_cpu, false);

    // Create a MaxPool2D module (using CPU branch).
    MaxPool2D maxpool_cpu(2, 2, batch_size, channels, height, width);
    VarPtr output_var_cpu = maxpool_cpu.forward(input_var_cpu);
    Tensor output_cpu = output_var_cpu->data;

    // Ensure output is on CPU.
    if (output_cpu.device() != CPU) {
        output_cpu.toCPU();
    }

    cout << "Expected Output (Ground Truth): ";
    for (size_t i = 0; i < expected.size(); ++i) {
        cout << expected[i] << " ";
    }
    cout << endl;

    cout << "CPU MaxPool Output: ";
    const float *cpu_out_data = output_cpu.data();
    int cpu_output_size = output_cpu.size();
    for (int i = 0; i < cpu_output_size; ++i) {
        cout << cpu_out_data[i] << " ";
    }
    cout << endl;

    // ---------- CUDA Test ----------
    cout << "\n===== Running MaxPool2D Test on CUDA =====" << endl;

    // Create a CPU tensor with known values (we'll then transfer it to CUDA).
    Tensor input_cuda(size, CPU);
    float *cuda_data = input_cuda.data();
    for (size_t i = 0; i < size; i++) {
        cuda_data[i] = static_cast<float>(i + 1);
    }
    // Transfer to CUDA.
    input_cuda.toCUDA();

    // Wrap the CUDA tensor in a Variable.
    VarPtr input_var_cuda = make_shared<Variable>(input_cuda, false);

    // Create a MaxPool2D module (this will run the CUDA branch).
    MaxPool2D maxpool_cuda(2, 2, batch_size, channels, height, width);
    VarPtr output_var_cuda = maxpool_cuda.forward(input_var_cuda);
    Tensor output_cuda = output_var_cuda->data;

    // Transfer the output to CPU for validation.
    if (output_cuda.device() != CPU) {
        output_cuda.toCPU();
    }

    cout << "Expected Output (Ground Truth): ";
    for (size_t i = 0; i < expected.size(); ++i) {
        cout << expected[i] << " ";
    }
    cout << endl;

    cout << "CUDA MaxPool Output: ";
    const float *cuda_out_data = output_cuda.data();
    int cuda_output_size = output_cuda.size();
    for (int i = 0; i < cuda_output_size; ++i) {
        cout << cuda_out_data[i] << " ";
    }
    cout << endl;
}