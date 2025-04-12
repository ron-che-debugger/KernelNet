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

#include "autograd.hpp"
#include "module.hpp"
#include "optimizer.hpp"
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

using namespace std;

/**
 * @brief Alias for a loss function.
 *
 * Accepts:
 *   - prediction: VarPtr output from the model
 *   - target: ground truth Tensor
 * Returns:
 *   - a scalar loss as VarPtr
 */
using LossFunction = std::function<VarPtr(const VarPtr &, const Tensor &)>;

/**
 * @brief Trainer class that handles forward, backward, and optimization steps.
 */
class Trainer {
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
            LossFunction loss_fn = MSEFunction::apply)
        : model(model), optimizer(optimizer), loss_fn(loss_fn) {}

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
    void trainEpoch(const vector<VarPtr> &inputs, const vector<VarPtr> &targets) {
        assert(inputs.size() == targets.size() && "Mismatched number of inputs and targets");

        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass.
            VarPtr prediction = model->forward(inputs[i]);

            // Compute loss.
            VarPtr loss = loss_fn(prediction, targets[i]->data);

            // Backward pass to compute gradients.
            loss->backward(loss->data);

            // Parameter update step.
            optimizer.step();

            // Clear gradients before the next sample.
            optimizer.zero_grad();
        }
    }
};