#pragma once
#include "autograd.hpp"
#include "lstm.hpp"
#include "single_input_module.hpp"
#include "tensor.hpp"
#include <cassert>

using namespace std;

class LSTM : public SingleInputModule {
  public:
    using SingleInputModule::forward;
    int batch_size, sequence_length, input_dim, hidden_dim;
    Device device;
    LSTMCell cell;

    LSTM(int batch_size, int sequence_length, int input_dim, int hidden_dim, Device device)
        : batch_size(batch_size), sequence_length(sequence_length), input_dim(input_dim),
          hidden_dim(hidden_dim), device(device), cell(input_dim, hidden_dim, device) {}

    // Unroll the LSTM over the sequence.
    // The input is assumed to be a flattened sequence tensor of shape (batch_size * sequence_length * input_dim).
    // For each time step, we extract a slice of length input_dim for each sample.
    virtual VarPtr forward(const VarPtr &input) override {
        // Initialize hidden and cell states to zeros.
        Tensor h0(batch_size * hidden_dim, device);
        Tensor c0(batch_size * hidden_dim, device);
        h0.fill(0.0f);
        c0.fill(0.0f);
        VarPtr h_prev = make_shared<Variable>(h0, true, "h0");
        VarPtr c_prev = make_shared<Variable>(c0, true, "c0");

        VarPtr current_hidden;
        // For each time step, extract the corresponding slice and update states.
        for (int t = 0; t < sequence_length; t++) {
            int offset = t * input_dim; // Each time step has input_dim values per sample.
            VarPtr x_t = SliceFunction::apply(input, batch_size, offset, offset + input_dim);
            vector<VarPtr> lstm_inputs = {x_t, h_prev, c_prev};
            vector<VarPtr> lstm_outputs = cell.forward(lstm_inputs);
            h_prev = lstm_outputs[0];
            c_prev = lstm_outputs[1];
            current_hidden = h_prev;
        }
        return current_hidden;
    }
};