#pragma once
#include "api_header.hpp"

#include <iostream>
#include <memory>
#include <vector>

using namespace std;

inline void runSingleLSTMCellTests() {
    // Define dimensions.
    int batch_size = 1;
    int input_dim = 1;
    int hidden_dim = 1;
    size_t input_size = batch_size * input_dim;
    size_t hidden_size = batch_size * hidden_dim;

    // Expected outputs:
    // With weights and biases set to zero, for input=1, h_prev=1, c_prev=1:
    //    i = f = o = sigmoid(0) = 0.5 and g = tanh(0) = 0.0.
    // Therefore:
    //    c_new = 0.5 * c_prev + 0.5 * 0 = 0.5, and
    //    h_new = 0.5 * tanh(c_new) = 0.5 * tanh(0.5) ≈ 0.5 * 0.46211716 = 0.23105858.
    float expected_h = 0.23105858f;
    float expected_c = 0.5f;

    // ------------- CPU Test -------------
    cout << "===== Running LSTM Forward Test on CPU =====" << endl;
    // Create input, previous hidden and previous cell tensors on CPU.
    Tensor input_cpu(input_size, CPU);
    Tensor h_prev_cpu(hidden_size, CPU);
    Tensor c_prev_cpu(hidden_size, CPU);
    float *input_cpu_data = input_cpu.data();
    float *h_prev_cpu_data = h_prev_cpu.data();
    float *c_prev_cpu_data = c_prev_cpu.data();
    // Set values.
    input_cpu_data[0] = 1.0f;
    h_prev_cpu_data[0] = 1.0f;
    c_prev_cpu_data[0] = 1.0f;

    // Wrap them in Variables.
    VarPtr input_var_cpu = make_shared<Variable>(input_cpu, false, "lstm_input_cpu");
    VarPtr h_prev_var_cpu = make_shared<Variable>(h_prev_cpu, false, "lstm_h_prev_cpu");
    VarPtr c_prev_var_cpu = make_shared<Variable>(c_prev_cpu, false, "lstm_c_prev_cpu");

    // Create an LSTMCell on CPU.
    LSTMCell lstm_cpu(input_dim, hidden_dim, CPU);

    // Override weights and biases to zeros.
    {
        // Create zero Tensors.
        Tensor zero_w_ih(input_dim * 4 * hidden_dim, CPU);
        zero_w_ih.fill(0.0f);
        Tensor zero_w_hh(hidden_dim * 4 * hidden_dim, CPU);
        zero_w_hh.fill(0.0f);
        Tensor zero_b_ih(4 * hidden_dim, CPU);
        zero_b_ih.fill(0.0f);
        Tensor zero_b_hh(4 * hidden_dim, CPU);
        zero_b_hh.fill(0.0f);
        // Replace parameters.
        lstm_cpu.weight_ih = make_shared<Variable>(zero_w_ih, true, "weight_ih");
        lstm_cpu.weight_hh = make_shared<Variable>(zero_w_hh, true, "weight_hh");
        lstm_cpu.bias_ih = make_shared<Variable>(zero_b_ih, true, "bias_ih");
        lstm_cpu.bias_hh = make_shared<Variable>(zero_b_hh, true, "bias_hh");
    }

    // Run forward pass.
    // Pack inputs into a vector.
    vector<VarPtr> lstm_inputs_cpu = {input_var_cpu, h_prev_var_cpu, c_prev_var_cpu};
    vector<VarPtr> lstm_outputs_cpu = lstm_cpu.forward(lstm_inputs_cpu);
    LSTMState state_cpu;
    state_cpu.h = lstm_outputs_cpu[0];
    state_cpu.c = lstm_outputs_cpu[1];

    cout << "Expected h (CPU): " << expected_h << ", Expected c (CPU): " << expected_c << endl;
    cout << "Computed h (CPU): ";
    state_cpu.h->data.print();
    cout << "Computed c (CPU): ";
    state_cpu.c->data.print();
    cout << endl;

    // ------------- CUDA Test -------------
    cout << "===== Running LSTM Forward Test on CUDA =====" << endl;
    // Create input, h_prev, c_prev as CPU tensors then convert to CUDA.
    Tensor input_cuda(input_size, CPU);
    Tensor h_prev_cuda(hidden_size, CPU);
    Tensor c_prev_cuda(hidden_size, CPU);
    float *input_cuda_data = input_cuda.data();
    float *h_prev_cuda_data = h_prev_cuda.data();
    float *c_prev_cuda_data = c_prev_cuda.data();
    input_cuda_data[0] = 1.0f;
    h_prev_cuda_data[0] = 1.0f;
    c_prev_cuda_data[0] = 1.0f;
    input_cuda.toCUDA();
    h_prev_cuda.toCUDA();
    c_prev_cuda.toCUDA();

    VarPtr input_var_cuda = make_shared<Variable>(input_cuda, false, "lstm_input_cuda");
    VarPtr h_prev_var_cuda = make_shared<Variable>(h_prev_cuda, false, "lstm_h_prev_cuda");
    VarPtr c_prev_var_cuda = make_shared<Variable>(c_prev_cuda, false, "lstm_c_prev_cuda");

    // Create an LSTMCell on CUDA.
    LSTMCell lstm_cuda(input_dim, hidden_dim, CUDA);
    // Override parameters to zeros.
    {
        Tensor zero_w_ih(input_dim * 4 * hidden_dim, CPU);
        zero_w_ih.fill(0.0f);
        zero_w_ih.toCUDA();
        Tensor zero_w_hh(hidden_dim * 4 * hidden_dim, CPU);
        zero_w_hh.fill(0.0f);
        zero_w_hh.toCUDA();
        Tensor zero_b_ih(4 * hidden_dim, CPU);
        zero_b_ih.fill(0.0f);
        zero_b_ih.toCUDA();
        Tensor zero_b_hh(4 * hidden_dim, CPU);
        zero_b_hh.fill(0.0f);
        zero_b_hh.toCUDA();
        lstm_cuda.weight_ih = make_shared<Variable>(zero_w_ih, true, "weight_ih");
        lstm_cuda.weight_hh = make_shared<Variable>(zero_w_hh, true, "weight_hh");
        lstm_cuda.bias_ih = make_shared<Variable>(zero_b_ih, true, "bias_ih");
        lstm_cuda.bias_hh = make_shared<Variable>(zero_b_hh, true, "bias_hh");
    }

    // Run forward pass on CUDA.
    vector<VarPtr> lstm_inputs_cuda = {input_var_cuda, h_prev_var_cuda, c_prev_var_cuda};
    vector<VarPtr> lstm_outputs_cuda = lstm_cuda.forward(lstm_inputs_cuda);
    LSTMState state_cuda;
    state_cuda.h = lstm_outputs_cuda[0];
    state_cuda.c = lstm_outputs_cuda[1];

    // Move results to CPU for printing.
    state_cuda.h->data.toCPU();
    state_cuda.c->data.toCPU();

    cout << "Expected h (CUDA): " << expected_h << ", Expected c (CUDA): " << expected_c << endl;
    cout << "Computed h (CUDA): ";
    state_cuda.h->data.print();
    cout << "Computed c (CUDA): ";
    state_cuda.c->data.print();
    cout << endl;
}