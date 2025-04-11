#pragma once

#include "autograd.hpp"
#include "module.hpp"
#include "optimizer.hpp"
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

using namespace std;

// Define a loss function type alias:
// It takes a prediction (VarPtr) and a target Tensor reference,
// and returns a VarPtr representing the loss.
using LossFunction = std::function<VarPtr(const VarPtr &, const Tensor &)>;

class Trainer {
  public:
    // The model (e.g., Sequential) must follow the unified Module interface.
    shared_ptr<Sequential> model;
    // Optimizer, e.g. SGD.
    SGD optimizer;
    // Loss function to compute the loss during training.
    LossFunction loss_fn;

    // Constructor: takes a model, an optimizer, and optionally a loss function.
    // Default loss function is MSEFunction::apply.
    Trainer(const shared_ptr<Sequential> &model, const SGD &optimizer,
            LossFunction loss_fn = MSEFunction::apply)
        : model(model), optimizer(optimizer), loss_fn(loss_fn) {}

    // Train the model for one epoch.
    // Here, inputs and targets are assumed to be provided as vectors of VarPtr.
    // For each sample, we perform a forward pass, compute the loss,
    // backpropagate the loss, update parameters, and zero the gradients.
    void trainEpoch(const vector<VarPtr> &inputs, const vector<VarPtr> &targets) {
        assert(inputs.size() == targets.size() && "Mismatched number of inputs and targets");
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass:
            VarPtr prediction = model->forward(inputs[i]);
            // Compute the loss using the configured loss function.
            // We assume that the target variable stores a tensor in its 'data' member.
            VarPtr loss = loss_fn(prediction, targets[i]->data);

            // Backward pass to compute gradients.
            loss->backward(loss->data);

            // Update the model parameters.
            optimizer.step();

            // Zero out gradients for the next iteration.
            optimizer.zero_grad();
        }
    }
};