/**
 * @file trainer.hpp
 * @brief Defines a simple Trainer class for model training with SGD and autograd.
 *
 * The trainer performs:
 * - Forward pass through the model
 * - Loss computation using a pluggable loss function
 * - Backward pass to compute gradients
 * - Parameter update using SGD
 */

#pragma once

#include "api_header.hpp"
#include "autograd.hpp"
#include "module.hpp"
#include "optimizer.hpp"
#include "sequential.hpp"
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

using namespace std;
using namespace kernelnet;
using namespace kernelnet::tensor;
using namespace kernelnet::autograd;
using namespace kernelnet::nn;
using namespace optim;

namespace kernelnet {
namespace trainer {

/**
 * @brief Alias for a loss function.
 *
 * Accepts:
 *   - prediction: VarPtr output from the model
 *   - target: ground truth Tensor
 * Returns:
 *   - a scalar loss as VarPtr
 */
using LossFunction = function<VarPtr(const VarPtr &, const Tensor &)>;

/**
 * @brief Trainer class that handles forward, backward, and optimization steps.
 */
class KERNELNET_API Trainer {
  public:
    shared_ptr<Sequential> model; ///< The model to be trained
    SGD optimizer;                ///< Optimizer (e.g., SGD)
    LossFunction loss_fn;         ///< Loss function (e.g., MSE or CrossEntropy)

    /**
     * @brief Constructs a Trainer object.
     *
     * @param model Shared pointer to a Sequential model.
     * @param optimizer SGD optimizer instance.
     * @param loss_fn Optional loss function (defaults to MSE).
     */
    Trainer(const shared_ptr<Sequential> &model, const SGD &optimizer,
            LossFunction loss_fn = MSEFunction::apply);

    /**
     * @brief Trains the model for one epoch on the provided data.
     *
     * Performs the following steps for each input-target pair:
     * - Forward pass
     * - Loss computation
     * - Backward pass
     * - Optimizer step
     * - Gradient reset
     *
     * @param inputs Vector of input VarPtr (batch of samples).
     * @param targets Vector of ground truth VarPtr (same size as inputs).
     */
    void trainEpoch(const vector<VarPtr> &inputs, const vector<VarPtr> &targets);
};

} // namespace trainer
} // namespace kernelnet