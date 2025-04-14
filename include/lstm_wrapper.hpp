/**
 * @file lstm_wrapper.hpp
 * @brief Defines an LSTM module that unrolls an LSTM cell over a sequence.
 *
 * This module wraps a single `LSTMCell` and performs forward computation across
 * all time steps of a sequence. It accepts a flattened input tensor of shape:
 *     [batch_size * sequence_length * input_dim]
 *
 * For each time step:
 * - A slice of shape [batch_size, input_dim] is extracted
 * - The LSTM cell is applied
 * - Hidden and cell states are updated
 *
 * The final hidden state is returned.
 */

#pragma once

#include "autograd.hpp"
#include "lstm.hpp"
#include "single_input_module.hpp"
#include "tensor.hpp"
#include <cassert>

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;

namespace kernelnet {
namespace nn {
/**
 * @brief Multi-step LSTM module that unrolls a single LSTMCell across time steps.
 */
class LSTM : public SingleInputModule {
  public:
    using SingleInputModule::forward;

    int batch_size;      ///< Number of samples in a batch
    int sequence_length; ///< Number of time steps
    int input_dim;       ///< Number of input features per time step
    int hidden_dim;      ///< Hidden state size
    Device device;       ///< CPU or CUDA

    LSTMCell cell; ///< Internal LSTM cell used at each time step

    /**
     * @brief Constructs an LSTM module using the provided dimensions and device.
     *
     * @param batch_size Number of sequences per batch.
     * @param sequence_length Number of time steps per sequence.
     * @param input_dim Size of the input vector at each time step.
     * @param hidden_dim Size of the hidden state.
     * @param device Execution device (CPU or CUDA).
     */
    LSTM(int batch_size, int sequence_length, int input_dim, int hidden_dim, Device device)
        : batch_size(batch_size), sequence_length(sequence_length), input_dim(input_dim),
          hidden_dim(hidden_dim), device(device), cell(input_dim, hidden_dim, device) {}

    /**
     * @brief Forward pass through the entire sequence using the internal LSTMCell.
     *
     * Accepts a flattened input tensor of shape:
     *     [batch_size * sequence_length * input_dim]
     *
     * @return A Variable with concatenated hidden states of shape
     *         [batch_size * sequence_length, hidden_dim].
     */
    VarPtr forward(const VarPtr &input) override {
        // Initialize hidden and cell states to zeros.
        Tensor h0(batch_size * hidden_dim, device);
        Tensor c0(batch_size * hidden_dim, device);
        h0.fill(0.0f);
        c0.fill(0.0f);
        VarPtr h_prev = make_shared<Variable>(h0, true, "h0");
        VarPtr c_prev = make_shared<Variable>(c0, true, "c0");

        // Container for hidden states.
        vector<VarPtr> hidden_states;

        // Unroll the LSTM over each time step.
        for (int t = 0; t < sequence_length; t++) {
            // Compute offset for current time step (each time step has input_dim elements per batch row).
            int offset = t * input_dim;

            // Slice out x_t with shape [batch_size, input_dim].
            VarPtr x_t = SliceFunction::apply(input, batch_size, offset, offset + input_dim);

            // Apply the LSTM cell.
            vector<VarPtr> lstm_inputs = {x_t, h_prev, c_prev};
            vector<VarPtr> lstm_outputs = cell.forward(lstm_inputs);

            // Update hidden and cell states.
            h_prev = lstm_outputs[0];
            c_prev = lstm_outputs[1];

            // Save the hidden state for this time step.
            hidden_states.push_back(h_prev);
        }

        // Concatenate hidden states along axis 0 to produce a tensor
        // of shape [batch_size * sequence_length, hidden_dim].
        VarPtr output = ConcatFunction::apply(hidden_states);
        return output;
    }
};
} // namespace nn
} // namespace kernelnet