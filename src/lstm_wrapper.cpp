#include "lstm_wrapper.hpp"

namespace kernelnet {
namespace nn {

LSTM::LSTM(int batch_size, int sequence_length, int input_dim, int hidden_dim, Device device)
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
VarPtr LSTM::forward(const VarPtr &input) {
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

/**
 * @brief Returns the trainable parameters of the LSTM module.
 *
 * The parameters are exactly the ones stored in the internal LSTMCell.
 *
 * @return A vector containing {weight_ih, weight_hh, bias_ih, bias_hh}.
 */
vector<VarPtr> LSTM::parameters() {
    return cell.parameters();
}

} // namespace nn
} // namespace kernelnet