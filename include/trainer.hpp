#pragma once

#include "module.hpp"
#include "optimizer.hpp"
#include <cassert>

using namespace std;

class Trainer {
  public:
    // The model (e.g., Sequential) must follow the unified Module interface.
    shared_ptr<Module> model;
    // Optimizer, e.g. SGD.
    SGD optimizer;

    // Constructor: takes a model and an optimizer.
    Trainer(const shared_ptr<Module> &model, const SGD &optimizer)
        : model(model), optimizer(optimizer) {}

    // Train the model for one epoch.
    // Here, inputs and targets are assumed to be provided as vectors of VarPtr.
    // For each sample, we perform a forward pass, compute the loss,
    // backpropagate the loss, update parameters, and zero the gradients.
    void trainEpoch(const vector<VarPtr> &inputs, const vector<VarPtr> &targets) {
        assert(inputs.size() == targets.size() && "Mismatched number of inputs and targets");
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass:
            // Since the base Module::forward expects a vector, wrap the single input in braces.
            vector<VarPtr> prediction_vec = model->forward({inputs[i]});
            VarPtr prediction = prediction_vec.front();

            // Compute the loss (using MSE as an example).
            // Note: MSEFunction::apply is assumed to take (prediction, target Tensor).
            // If target is a Variable, use target->data.
            VarPtr loss = MSEFunction::apply(prediction, targets[i]->data);

            // Backward pass to compute gradients.
            loss->backward(loss->data);

            // Update the model parameters.
            optimizer.step();

            // Zero out gradients for the next iteration.
            optimizer.zero_grad();
        }
    }
};
