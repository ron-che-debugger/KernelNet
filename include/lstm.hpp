#pragma once
#include "autograd.hpp"
#include "module.hpp"
#include "sigmoid.hpp"
#include "slice.hpp"
#include "tanh.hpp"
#include "tensor.hpp"
#include <cmath>
#include <tuple>
#include <vector>

using namespace std;

// Structure to hold the LSTM cell's outputs.
struct LSTMState {
    VarPtr h; // New hidden state.
    VarPtr c; // New cell state.
};

// The merged LSTM cell function: it performs the forward pass and saves
// all intermediate values as hard pointers for autograd.
class LSTMCellFunction : public Function {
  public:
    // Saved inputs and parameters.
    VarPtr saved_input;
    VarPtr saved_h_prev;
    VarPtr saved_c_prev;
    VarPtr saved_weight_ih;
    VarPtr saved_weight_hh;
    VarPtr saved_bias_ih;
    VarPtr saved_bias_hh;

    // Saved intermediate gate tensors.
    VarPtr saved_i_gate;     // raw input gate.
    VarPtr saved_f_gate;     // raw forget gate.
    VarPtr saved_g_gate;     // raw candidate.
    VarPtr saved_o_gate;     // raw output gate.
    VarPtr saved_i_gate_act; // i = sigmoid(saved_i_gate)
    VarPtr saved_f_gate_act; // f = sigmoid(saved_f_gate)
    VarPtr saved_g_gate_act; // g = tanh(saved_g_gate)
    VarPtr saved_o_gate_act; // o = sigmoid(saved_o_gate)
    VarPtr saved_c_new;      // c_new = f ⊙ c_prev + i ⊙ g (before tanh for h)

    int input_dim, hidden_dim, batch_size;

    // Forward pass.
    // Returns a pair {h_new, c_new} (only h_new is stored as output).
    static pair<VarPtr, VarPtr> apply(const VarPtr &input,
                                      const VarPtr &h_prev,
                                      const VarPtr &c_prev,
                                      const VarPtr &weight_ih,
                                      const VarPtr &weight_hh,
                                      const VarPtr &bias_ih,
                                      const VarPtr &bias_hh,
                                      int input_dim,
                                      int hidden_dim);

    // Backward pass.
    // Expects a single Tensor grad_output of shape (B, 2H),
    // where the first half is grad_h (dL/dh_new) and the second half is grad_c (dL/dc_new).
    // Returns gradients for [input, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh].
    virtual vector<Tensor> backward(const Tensor &grad_output) override;
};

// The LSTMCell module: holds learnable parameters and calls the LSTMCellFunction.
class LSTMCell : public Module {
  public:
    int input_dim, hidden_dim;
    Device device;

    // Learnable parameters.
    VarPtr weight_ih; // (input_dim, 4 * hidden_dim)
    VarPtr weight_hh; // (hidden_dim, 4 * hidden_dim)
    VarPtr bias_ih;   // (4 * hidden_dim)
    VarPtr bias_hh;   // (4 * hidden_dim)

    // Constructor.
    LSTMCell(int input_dim, int hidden_dim, Device device = CPU);

    // Forward pass: returns new hidden and cell states.
    // Update the forward pass to conform to the unified interface.
    // Expects a vector of exactly three inputs: {input, h_prev, c_prev}.
    // Returns a vector of two outputs: {h_new, c_new}.
    vector<VarPtr> forward(const vector<VarPtr> &inputs) override;

    vector<VarPtr> parameters() override;
};