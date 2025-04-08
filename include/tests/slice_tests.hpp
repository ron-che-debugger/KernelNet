#pragma once
#include "slice.hpp"
#include "tensor.hpp"
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

inline void runSliceTests() {
    // Define the input dimensions: 2 samples (batch_size), 5 features per sample.
    int batch_size = 2;
    int total_width = 5;
    int slice_start = 1;
    int slice_end = 4;
    int slice_length = slice_end - slice_start; // should be 3
    size_t input_size = batch_size * total_width;

    // Expected output after slicing: For each row, elements at indices 1, 2, and 3.
    // For row 0: [11, 12, 13], row 1: [21, 22, 23]
    vector<float> expected = {
        11.0f, 12.0f, 13.0f,
        21.0f, 22.0f, 23.0f};

    // --------------- CPU Test ---------------
    cout << "===== Running Slice Test on CPU =====" << endl;

    // Create a CPU tensor with known values.
    // We create a tensor of size 2x5: row0 = [10, 11, 12, 13, 14], row1 = [20, 21, 22, 23, 24]
    Tensor input_cpu(input_size, CPU);
    float *cpu_data = input_cpu.data();
    // Row 0
    cpu_data[0] = 10.0f;
    cpu_data[1] = 11.0f;
    cpu_data[2] = 12.0f;
    cpu_data[3] = 13.0f;
    cpu_data[4] = 14.0f;
    // Row 1
    cpu_data[5] = 20.0f;
    cpu_data[6] = 21.0f;
    cpu_data[7] = 22.0f;
    cpu_data[8] = 23.0f;
    cpu_data[9] = 24.0f;

    // Wrap input in a Variable.
    VarPtr input_var_cpu = make_shared<Variable>(input_cpu, false, "slice_input");

    // Run the forward pass using the SliceFunction.
    // We assume the input is interpreted as shape [batch_size, total_width].
    VarPtr output_var_cpu = SliceFunction::apply(input_var_cpu, batch_size, slice_start, slice_end);
    Tensor output_cpu = output_var_cpu->data; // Already on CPU.

    cout << "Expected Slice Output (CPU): ";
    for (auto val : expected) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Slice Output (CPU): ";
    const float *cpu_out = output_cpu.data();
    for (size_t i = 0; i < batch_size * slice_length; i++) {
        cout << cpu_out[i] << " ";
    }
    cout << endl
         << endl;

    // --------------- CUDA Test ---------------
    cout << "===== Running Slice Test on CUDA =====" << endl;

    // Create a CPU tensor with the same values then transfer to CUDA.
    Tensor input_cuda(input_size, CPU);
    float *cuda_data = input_cuda.data();
    // Row 0
    cuda_data[0] = 10.0f;
    cuda_data[1] = 11.0f;
    cuda_data[2] = 12.0f;
    cuda_data[3] = 13.0f;
    cuda_data[4] = 14.0f;
    // Row 1
    cuda_data[5] = 20.0f;
    cuda_data[6] = 21.0f;
    cuda_data[7] = 22.0f;
    cuda_data[8] = 23.0f;
    cuda_data[9] = 24.0f;
    input_cuda.toCUDA();

    VarPtr input_var_cuda = make_shared<Variable>(input_cuda, false, "slice_input");

    // Call the slice forward pass.
    VarPtr output_var_cuda = SliceFunction::apply(input_var_cuda, batch_size, slice_start, slice_end);
    Tensor output_cuda = output_var_cuda->data;
    // Ensure output is on CPU for printing.
    if (output_cuda.device() != CPU) {
        output_cuda.toCPU();
    }

    cout << "Expected Slice Output (CUDA): ";
    for (auto val : expected) {
        cout << val << " ";
    }
    cout << endl;

    cout << "Slice Output (CUDA): ";
    const float *cuda_out = output_cuda.data();
    for (size_t i = 0; i < batch_size * slice_length; i++) {
        cout << cuda_out[i] << " ";
    }
    cout << endl;
}