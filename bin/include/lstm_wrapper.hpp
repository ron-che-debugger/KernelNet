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

#include "api_header.hpp"
#include "autograd.hpp"
#include "lstm.hpp"
#include "single_input_module.hpp"
#include "tensor.hpp"
#include <cassert>
#include <memory>
#include <string>
#include <vector>

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
class KERNELNET_API LSTM : public SingleInputModule {
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
    LSTM(int batch_size, int sequence_length, int input_dim, int hidden_dim, Device device);

    /**
     * @brief Forward pass through the entire sequence using the internal LSTMCell.
     *
     * Accepts a flattened input tensor of shape:
     *     [batch_size * sequence_length * input_dim]
     *
     * @return A Variable with concatenated hidden states of shape
     *         [batch_size * sequence_length, hidden_dim].
     */
    virtual VarPtr forward(const VarPtr &input) override;

    /**
     * @brief Returns the trainable parameters of the LSTM module.
     *
     * The parameters are exactly the ones stored in the internal LSTMCell.
     *
     * @return A vector containing {weight_ih, weight_hh, bias_ih, bias_hh}.
     */
    virtual vector<VarPtr> parameters() override;
};

} // namespace nn
} // namespace kernelnet