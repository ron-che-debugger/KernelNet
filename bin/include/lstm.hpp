/**
 * @file lstm.hpp
 * @brief Defines the LSTMCell and LSTMCellFunction for sequence modeling.
 *
 * This file includes:
 * - `LSTMCellFunction`: A low-level autograd-enabled function for LSTM computation
 * - `LSTMCell`: A high-level module that stores parameters and invokes LSTMCellFunction
 * - `generateSequenceData`: A helper to create synthetic sequence regression data
 */

#pragma once

#include "api_header.hpp"
#include "autograd.hpp"
#include "module.hpp"
#include "sigmoid.hpp"
#include "tanh.hpp"
#include "tensor.hpp"
#include <cmath>
#include <tuple>
#include <vector>

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;

namespace kernelnet {
namespace nn {
/**
 * @brief Structure to hold the output states of an LSTM cell.
 */
KERNELNET_API struct LSTMState {
    VarPtr h; ///< New hidden state
    VarPtr c; ///< New cell state
};

/**
 * @brief Autograd-compatible LSTM cell function.
 *
 * Internally handles all four gates:
 * - Input gate `i`
 * - Forget gate `f`
 * - Output gate `o`
 * - Cell candidate `g`
 *
 * Saves intermediate activations for backward pass.
 */
class KERNELNET_API LSTMCellFunction : public Function {
  public:
    // Saved inputs and parameters.
    VarPtr saved_input, saved_h_prev, saved_c_prev;
    VarPtr saved_weight_ih, saved_weight_hh;
    VarPtr saved_bias_ih, saved_bias_hh;

    // Saved raw gates and activations.
    VarPtr saved_i_gate, saved_f_gate, saved_g_gate, saved_o_gate;
    VarPtr saved_i_gate_act, saved_f_gate_act, saved_g_gate_act, saved_o_gate_act;
    VarPtr saved_c_new;

    int input_dim;  ///< Input feature size
    int hidden_dim; ///< Hidden state size
    int batch_size; ///< Batch size

    /**
     * @brief Forward pass of the LSTM cell.
     * @param input Input tensor at time t.
     * @param h_prev Hidden state from t-1.
     * @param c_prev Cell state from t-1.
     * @param weight_ih Input-to-hidden weights.
     * @param weight_hh Hidden-to-hidden weights.
     * @param bias_ih Input-to-hidden bias.
     * @param bias_hh Hidden-to-hidden bias.
     * @param input_dim Size of input vector.
     * @param hidden_dim Size of hidden vector.
     * @return A pair of {h_new, c_new}.
     */
    static pair<VarPtr, VarPtr> apply(const VarPtr &input,
                                      const VarPtr &h_prev,
                                      const VarPtr &c_prev,
                                      const VarPtr &weight_ih,
                                      const VarPtr &weight_hh,
                                      const VarPtr &bias_ih,
                                      const VarPtr &bias_hh,
                                      int input_dim,
                                      int hidden_dim);

    /**
     * @brief Backward pass of the LSTM cell.
     * @param grad_output Concatenated gradient of output and cell state: [dL/dh, dL/dc]
     * @return Gradients for {input, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh}
     */
    vector<Tensor> backward(const Tensor &grad_output) override;
};

/**
 * @brief A learnable LSTM cell module.
 *
 * This class stores parameters (weights and biases), and provides a high-level
 * interface for computing LSTM output from input + previous states.
 */
class KERNELNET_API LSTMCell : public Module {
  public:
    int input_dim;  ///< Input feature size
    int hidden_dim; ///< Hidden state size
    Device device;  ///< Execution device

    // Parameters
    VarPtr weight_ih; ///< Weight matrix from input: shape (input_dim, 4 * hidden_dim)
    VarPtr weight_hh; ///< Weight matrix from hidden: shape (hidden_dim, 4 * hidden_dim)
    VarPtr bias_ih;   ///< Input bias: shape (4 * hidden_dim)
    VarPtr bias_hh;   ///< Hidden bias: shape (4 * hidden_dim)

    /**
     * @brief Constructs a new LSTM cell.
     * @param input_dim Size of input vector at each time step.
     * @param hidden_dim Size of hidden state.
     * @param device Device to allocate parameters on.
     */
    LSTMCell(int input_dim, int hidden_dim, Device device = CPU);

    /**
     * @brief Applies the LSTM cell for a single time step.
     * @param inputs A vector of 3 elements: {input, h_prev, c_prev}
     * @return A vector of 2 elements: {h_new, c_new}
     */
    vector<VarPtr> forward(const vector<VarPtr> &inputs) override;

    /**
     * @brief Returns all trainable parameters in the LSTM cell.
     * @return A vector of {weight_ih, weight_hh, bias_ih, bias_hh}
     */
    vector<VarPtr> parameters() override;
};

/**
 * @brief Generates synthetic sequence-to-value training data.
 *
 * Each sample is a sequence of integers [1, 2, ..., sequence_length]
 * repeated across all batches. The target for each sequence is the sum of all its elements.
 *
 * @param batch_size Number of sequences in the batch.
 * @param sequence_length Number of time steps per sequence.
 * @param input_dim Number of features per time step.
 * @param input Output tensor for the input (flattened as batch × sequence × dim).
 * @param target Output tensor for the summed target values.
 */
inline void generateSequenceData(int batch_size, int sequence_length, int input_dim,
                                 Tensor &input, Tensor &target) {
    int total_elems = batch_size * sequence_length * input_dim;
    input = Tensor(total_elems, CPU);
    target = Tensor(batch_size, CPU);

    float *in_data = input.data();
    float *tgt_data = target.data();

    for (int b = 0; b < batch_size; b++) {
        float sum = 0.0f;
        for (int t = 0; t < sequence_length; t++) {
            for (int d = 0; d < input_dim; d++) {
                int idx = b * (sequence_length * input_dim) + t * input_dim + d;
                in_data[idx] = static_cast<float>(t + 1); // Value = timestep (1-based)
                sum += in_data[idx];
            }
        }
        tgt_data[b] = sum;
    }
}
} // namespace nn
} // namespace kernelnet